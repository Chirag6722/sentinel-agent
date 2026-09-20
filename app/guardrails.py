"""Policy engine: decides, for every tool call the model proposes, whether it
runs automatically, needs a human, or is refused.

Design principles
-----------------
* The model never executes anything; it only *proposes* tool calls.
* Every decision is deterministic code with a named rule and a human-readable
  reason, so the audit trail explains itself.
* Guardrails are layered:
    1. Input trust     - prompt-injection scanning of untrusted text, raises run risk
    2. Budget          - step / tool-call caps, loop detection, circuit breaker
    3. Scope           - actions must target the ticket's own customer/orders
    4. Money           - auto / approval / hard-deny thresholds, per-run cap
    5. Comms           - outbound email only to the customer or internal domain,
                         no secrets/PII leakage in the body
    6. Risk escalation - when the input looked hostile, every WRITE needs a human
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from . import config as _cfg
from .tools import TOOL_KIND

Verdict = Literal["ALLOW", "REQUIRE_APPROVAL", "DENY"]


@dataclass
class Decision:
    verdict: Verdict
    rule: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "rule": self.rule, "reason": self.reason}


# ---------------------------------------------------------------------------
# 1. Input trust: prompt-injection heuristics
# ---------------------------------------------------------------------------
# Patterns that appear in instruction-hijack attempts far more than in genuine
# support requests. Heuristic, not a classifier: false positives only raise the
# bar to "ask a human", they never block a legitimate customer outright.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore (all |any )?(previous|prior|above|earlier) instructions", "instruction override"),
    (r"\b(system|admin|developer|management) (notice|override|message|instruction)s?\b", "fake system message"),
    (r"\b(to|for) (the )?(ai|assistant|agent|model|llm)\b", "addresses the AI directly"),
    (r"\bpre-?approved\b", "claims prior approval"),
    (r"\bwithout (asking|approval|confirmation|verification)\b", "asks to skip approval"),
    (r"\bdo not (mention|tell|reveal|disclose)\b", "asks for concealment"),
    (r"\b(send|forward|email)\b[^.\n]{0,60}\b(account|password|card|ssn|details)\b", "data exfiltration"),
    (r"\byou (must|are required to|have to)\b", "coercive phrasing"),
    (r"\$\s?\d{4,}|\b\d{4,}\s?(usd|dollars)\b", "unusually large amount"),
]


@dataclass
class InjectionFinding:
    label: str
    excerpt: str
    span: tuple[int, int]


def scan_for_injection(text: str) -> list[InjectionFinding]:
    findings: list[InjectionFinding] = []
    for pattern, label in INJECTION_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            findings.append(InjectionFinding(label, text[m.start():m.end()], (m.start(), m.end())))
    return findings


def risk_from_findings(findings: list[InjectionFinding]) -> str:
    n = len(findings)
    if n == 0:
        return "low"
    if n <= 2:
        return "elevated"
    return "high"


# ---------------------------------------------------------------------------
# 5. Output / comms checks
# ---------------------------------------------------------------------------
SECRET_PATTERNS = [
    (r"\b(?:\d[ -]?){13,19}\b", "card-number-like digits"),
    (r"\b(api[_ -]?key|password|secret|token)\b\s*[:=]", "credential"),
    (r"\bsk-[A-Za-z0-9]{8,}\b", "API key"),
]


def scan_email_body(body: str) -> list[str]:
    return [label for pat, label in SECRET_PATTERNS if re.search(pat, body, flags=re.IGNORECASE)]


# ---------------------------------------------------------------------------
# Run context the policy reasons about
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    run_id: str
    ticket: dict[str, Any]
    customer: dict[str, Any]
    customer_orders: dict[str, dict[str, Any]]   # order_id -> order row (live snapshot)
    risk_level: str = "low"
    tool_calls: int = 0
    refunded_total: float = 0.0
    refund_count: int = 0
    failures: dict[str, int] = field(default_factory=dict)      # tool -> failure count
    history: list[str] = field(default_factory=list)            # fingerprints of proposed calls
    denied_history: list[str] = field(default_factory=list)
    fused_tools: set[str] = field(default_factory=set)         # tripped circuit breakers
    escalated: bool = False

    @property
    def customer_order_ids(self) -> set[str]:
        return set(self.customer_orders)


def fingerprint(name: str, args: dict[str, Any]) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
class Policy:
    def __init__(self) -> None:
        pass

    @property
    def s(self):
        return _cfg.settings

    def describe(self) -> list[dict[str, str]]:
        """Human-readable rule list for the UI."""
        s = self.s
        return [
            {"id": "input-trust", "text": "Customer text is untrusted; injection heuristics raise the run's risk level."},
            {"id": "budget", "text": f"Max {s.max_steps} model steps / {s.max_tool_calls} tool calls per run; repeated identical calls are blocked."},
            {"id": "circuit-breaker", "text": f"A tool that fails {s.circuit_breaker_failures}x is fused for the rest of the run."},
            {"id": "unresolved-failure", "text": "A run with a fused tool cannot finish until a human has been looped in via escalate_to_human."},
            {"id": "scope", "text": "Write actions may only target the ticket's own customer and their orders."},
            {"id": "refund-auto", "text": f"Refunds <= ${s.refund_auto_limit:.0f} run automatically."},
            {"id": "refund-approval", "text": f"Refunds ${s.refund_auto_limit:.0f}-${s.refund_approval_limit:.0f} need human approval."},
            {"id": "refund-deny", "text": f"Refunds > ${s.refund_approval_limit:.0f}, > order total, or > ${s.refund_run_cap:.0f} cumulative are refused."},
            {"id": "cancel", "text": "Cancelling a shipped order needs approval; delivered orders cannot be cancelled."},
            {"id": "comms", "text": f"Email only to the ticket's customer or @{s.internal_email_domain}; bodies are scanned for secrets."},
            {"id": "risk-escalation", "text": "If risk is elevated/high, every write action needs human approval."},
        ]

    # -- entry point ---------------------------------------------------------
    def evaluate(self, ctx: RunContext, name: str, args: dict[str, Any]) -> Decision:
        kind = TOOL_KIND.get(name)
        if kind is None:
            return Decision("DENY", "unknown-tool", f"'{name}' is not an allowed tool")

        # 2. Budget --------------------------------------------------------
        if ctx.tool_calls >= self.s.max_tool_calls:
            return Decision("DENY", "budget", f"tool-call budget of {self.s.max_tool_calls} exhausted")
        fp = fingerprint(name, args)
        if ctx.history.count(fp) >= 2:
            return Decision("DENY", "loop-detection", "identical call already made twice this run")
        if ctx.failures.get(name, 0) >= self.s.circuit_breaker_failures:
            return Decision("DENY", "circuit-breaker",
                            f"'{name}' failed {ctx.failures[name]}x this run; fused. Escalate instead.")

        if kind == "READ":
            return Decision("ALLOW", "read-only", "read-only tool")
        if kind == "META":
            # A run whose refund/cancel tool got fused has an unresolved failure:
            # it may not be closed as "done" - a human must be looped in first.
            if name == "finish" and ctx.fused_tools and not ctx.escalated:
                return Decision("DENY", "unresolved-failure",
                                f"{', '.join(sorted(ctx.fused_tools))} failed and was fused; call escalate_to_human before finishing")
            return Decision("ALLOW", "control-flow", "control-flow tool")

        # 3. Scope ---------------------------------------------------------
        scope = self._check_scope(ctx, name, args)
        if scope:
            return scope

        # 4/5. Tool-specific rules ------------------------------------------
        d = {
            "issue_refund": self._refund,
            "cancel_order": self._cancel,
            "send_email": self._email,
            "add_ticket_note": lambda c, a: Decision("ALLOW", "internal-note", "internal note, not customer-visible"),
        }[name](ctx, args)

        # 6. Risk escalation -------------------------------------------------
        if d.verdict == "ALLOW" and ctx.risk_level != "low" and name != "add_ticket_note":
            return Decision("REQUIRE_APPROVAL", "risk-escalation",
                            f"run risk is {ctx.risk_level} (suspicious input); write actions need a human")
        return d

    # -- helpers --------------------------------------------------------------
    def _check_scope(self, ctx: RunContext, name: str, args: dict) -> Decision | None:
        if name in ("issue_refund", "cancel_order"):
            oid = str(args.get("order_id", ""))
            if oid not in ctx.customer_order_ids:
                return Decision("DENY", "scope",
                                f"order {oid} does not belong to ticket customer {ctx.customer['id']}")
        if name == "add_ticket_note" and str(args.get("ticket_id")) != ctx.ticket["id"]:
            return Decision("DENY", "scope", "note must be on the current ticket")
        return None

    def _refund(self, ctx: RunContext, a: dict) -> Decision:
        try:
            amount = float(a.get("amount", 0))
        except (TypeError, ValueError):
            return Decision("DENY", "refund-invalid", "amount is not a number")
        s = self.s
        if amount <= 0:
            return Decision("DENY", "refund-invalid", "amount must be positive")
        order = ctx.customer_orders.get(str(a.get("order_id")))
        refundable = order["total"] - order.get("refunded", 0.0) if order else 0.0
        if amount > refundable + 1e-9:
            return Decision("DENY", "refund-exceeds-order",
                            f"${amount:.2f} exceeds refundable ${refundable:.2f} on {a.get('order_id')}")
        if amount > s.refund_approval_limit:
            return Decision("DENY", "refund-hard-cap", f"${amount:.2f} exceeds hard cap ${s.refund_approval_limit:.0f}")
        if ctx.refunded_total + amount > s.refund_run_cap:
            return Decision("DENY", "refund-run-cap",
                            f"cumulative refunds would reach ${ctx.refunded_total + amount:.2f} > ${s.refund_run_cap:.0f}")
        if amount > s.refund_auto_limit:
            return Decision("REQUIRE_APPROVAL", "refund-approval",
                            f"${amount:.2f} is above the ${s.refund_auto_limit:.0f} auto-refund limit")
        return Decision("ALLOW", "refund-auto", f"${amount:.2f} is within the auto-refund limit")

    def _cancel(self, ctx: RunContext, a: dict) -> Decision:
        order = ctx.customer_orders.get(str(a.get("order_id")))
        status = order["status"] if order else "unknown"
        if status in ("delivered", "cancelled"):
            return Decision("DENY", "cancel-final-state", f"order is {status}; cancellation impossible")
        if status == "shipped":
            return Decision("REQUIRE_APPROVAL", "cancel-shipped", "order already shipped; recall costs money")
        return Decision("ALLOW", "cancel-unshipped", f"order is {status}, safe to cancel")

    def _email(self, ctx: RunContext, a: dict) -> Decision:
        to = str(a.get("to", "")).strip().lower()
        cust = str(ctx.customer.get("email", "")).lower()
        if to != cust and not to.endswith("@" + self.s.internal_email_domain):
            return Decision("DENY", "comms-recipient",
                            f"recipient {to} is neither the customer ({cust}) nor internal")
        leaks = scan_email_body(str(a.get("subject", "")) + "\n" + str(a.get("body", "")))
        if leaks:
            return Decision("DENY", "comms-leak", "email body contains " + ", ".join(leaks))
        return Decision("ALLOW", "comms-customer", "email to the ticket's own customer")
