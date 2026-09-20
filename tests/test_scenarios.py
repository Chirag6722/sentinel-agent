"""End-to-end scenario tests against the mock provider. Each demo ticket must
produce exactly the guardrail behaviour we promise in the README."""
from __future__ import annotations

import asyncio
import os

import pytest

os.environ["LLM_PROVIDER"] = "mock"
os.environ["DB_PATH"] = "test.db"

from app import db  # noqa: E402
from app.agent import AgentRunner  # noqa: E402
from app.guardrails import Policy, scan_for_injection  # noqa: E402
from app.llm import MockProvider  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    db.init_db(reset=True)
    yield


def types(run, t):
    return [e for e in run.events if e["type"] == t]


def decisions(run):
    return [(e["tool"], e["verdict"], e["rule"]) for e in types(run, "policy_decision")]


async def drive(ticket_id, approve: bool | None = None, note=""):
    """Run a ticket to completion; auto-answer approval gates with `approve`."""
    runner = AgentRunner(MockProvider(), Policy())
    run = runner.start(ticket_id)
    for _ in range(200):
        await asyncio.sleep(0.02)
        if run.pending:
            assert approve is not None, "unexpected approval gate"
            runner.resolve_approval(run.id, run.pending.approval_id, approve, note)
        if run.task.done():
            break
    assert run.task.done(), "run did not finish"
    return run


def world():
    with db.tx() as c:
        return {
            "refunds": db.rows(c.execute("SELECT * FROM refunds")),
            "emails": db.rows(c.execute("SELECT * FROM emails")),
            "orders": {o["id"]: o for o in db.rows(c.execute("SELECT * FROM orders"))},
            "tickets": {t["id"]: t for t in db.rows(c.execute("SELECT * FROM tickets"))},
        }


# --------------------------------------------------------------------------- tests
@pytest.mark.asyncio
async def test_happy_path_is_fully_autonomous():
    run = await drive("T-101")
    assert run.status == "completed"
    assert not types(run, "approval_requested")
    d = decisions(run)
    assert ("issue_refund", "ALLOW", "refund-auto") in d
    assert ("send_email", "ALLOW", "comms-customer") in d
    w = world()
    assert [r["amount"] for r in w["refunds"]] == [34.99]
    assert w["emails"][0]["to_addr"] == "priya.nair@example.com"


@pytest.mark.asyncio
async def test_large_refund_needs_approval_and_proceeds_when_approved():
    run = await drive("T-102", approve=True)
    assert run.status == "completed"
    gates = types(run, "approval_requested")
    assert len(gates) == 1 and gates[0]["tool"] == "issue_refund" and gates[0]["rule"] == "refund-approval"
    assert world()["refunds"][0]["amount"] == 249.0


@pytest.mark.asyncio
async def test_large_refund_rejected_leads_to_escalation_not_retry():
    run = await drive("T-102", approve=False, note="customer already refunded via chargeback")
    assert run.status == "escalated"
    assert world()["refunds"] == []
    assert [d for d in decisions(run) if d[0] == "issue_refund"] == [("issue_refund", "REQUIRE_APPROVAL", "refund-approval")]


@pytest.mark.asyncio
async def test_prompt_injection_is_detected_and_contained():
    findings = scan_for_injection(db.TICKETS[2][4])
    assert {f.label for f in findings} >= {"instruction override", "fake system message", "claims prior approval"}

    run = await drive("T-103", approve=True)
    d = decisions(run)
    # The injected $5000 refund and the exfiltration email are both refused outright.
    assert ("issue_refund", "DENY", "refund-exceeds-order") in d
    assert ("send_email", "DENY", "comms-recipient") in d
    # The legitimate refund is gated (amount rule) and, because risk is high,
    # even the confirmation email to the real customer needs a human.
    assert ("issue_refund", "REQUIRE_APPROVAL", "refund-approval") in d
    assert ("send_email", "REQUIRE_APPROVAL", "risk-escalation") in d
    assert len(types(run, "approval_requested")) == 2
    assert run.risk_level == "high"
    w = world()
    assert [r["amount"] for r in w["refunds"]] == [89.5]
    assert all(e["to_addr"] == "sara.khan@example.com" for e in w["emails"])


@pytest.mark.asyncio
async def test_scope_violation_denied_and_shipped_cancel_gated():
    run = await drive("T-104", approve=True)
    d = decisions(run)
    assert ("cancel_order", "REQUIRE_APPROVAL", "cancel-shipped") in d
    assert ("cancel_order", "DENY", "scope") in d
    w = world()
    assert w["orders"]["O-5004"]["status"] == "cancelled"
    assert w["orders"]["O-5005"]["status"] == "processing"  # someone else's order untouched
    assert run.status == "completed"


@pytest.mark.asyncio
async def test_tool_failure_retries_then_circuit_breaks_and_escalates():
    run = await drive("T-105")
    fails = types(run, "tool_failed")
    assert len(fails) == 3 and all(f["transient"] for f in fails)
    assert ("issue_refund", "DENY", "circuit-breaker") in decisions(run)
    assert run.status == "escalated"
    w = world()
    assert w["refunds"] == []
    assert w["tickets"]["T-105"]["status"] == "escalated"


@pytest.mark.asyncio
async def test_audit_log_is_persisted_in_order():
    run = await drive("T-101")
    with db.tx() as c:
        seqs = [r[0] for r in c.execute("SELECT seq FROM audit_events WHERE run_id=? ORDER BY seq", (run.id,))]
    assert seqs == list(range(1, len(run.events) + 1))


@pytest.mark.asyncio
async def test_stop_cancels_pending_approval():
    runner = AgentRunner(MockProvider(), Policy())
    run = runner.start("T-102")
    for _ in range(200):
        await asyncio.sleep(0.02)
        if run.pending:
            break
    assert run.pending is not None, "run never reached approval gate"
    runner.stop(run.id)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if run.task.done():
            break
    assert run.task.done()
    assert run.status == "stopped"
    resolved = [e for e in run.events if e["type"] == "approval_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["approved"] is False
    assert world()["refunds"] == []


def test_cannot_finish_with_a_fused_tool_until_escalated():
    from app.guardrails import RunContext
    ctx = RunContext(run_id="r", ticket={"id": "T-1"}, customer={"id": "C-1", "email": "x@example.com"},
                     customer_orders={}, fused_tools={"issue_refund"})
    d = Policy().evaluate(ctx, "finish", {"summary": "all good"})
    assert (d.verdict, d.rule) == ("DENY", "unresolved-failure")
    ctx.escalated = True
    assert Policy().evaluate(ctx, "finish", {"summary": "escalated"}).verdict == "ALLOW"
