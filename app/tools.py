"""Tools the agent can call. Every tool is a plain function over the SQLite
store; the *policy engine* (guardrails.py) decides whether a call may run,
these functions only do the work.

Tools are classified by side effect so the policy can reason about them:
  READ   - no side effects
  WRITE  - changes customer-visible state (money, orders, outbound comms)
  META   - control flow (escalate / finish)
"""
from __future__ import annotations

from typing import Any, Callable

from . import db


class ToolError(Exception):
    """Permanent failure - retrying will not help."""


class TransientToolError(ToolError):
    """Temporary failure (gateway timeout etc.) - retry is reasonable."""


# ---------------------------------------------------------------------------
# Schemas (OpenAI / Groq function-calling format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_ticket", "description": "Fetch the support ticket being worked on.",
        "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]}}},
    {"type": "function", "function": {
        "name": "lookup_customer", "description": "Fetch a customer profile and the list of their orders.",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_order", "description": "Fetch one order: items, total, status, amount already refunded.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Refund money to the customer for an order. Amount in USD, must not exceed the order's remaining refundable total.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "amount": {"type": "number", "description": "USD amount to refund"},
            "reason": {"type": "string", "description": "Short reason shown on the customer's statement"}},
            "required": ["order_id", "amount", "reason"]}}},
    {"type": "function", "function": {
        "name": "cancel_order", "description": "Cancel an order. Orders in status processing are cancelled directly; shipped orders can still be recalled (may need approval); delivered orders cannot be cancelled.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                       "required": ["order_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "send_email", "description": "Send an email to the customer.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "add_ticket_note", "description": "Add an internal note to the ticket (not visible to the customer).",
        "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}, "note": {"type": "string"}},
                       "required": ["ticket_id", "note"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the ticket to a human agent. Use when you cannot safely complete the request, when a policy blocks you, or when something looks suspicious.",
        "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}, "reason": {"type": "string"}},
                       "required": ["ticket_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "End the run with a one-paragraph summary of what was done. Call this exactly once when the ticket is handled.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]

TOOL_KIND: dict[str, str] = {
    "get_ticket": "READ", "lookup_customer": "READ", "get_order": "READ",
    "issue_refund": "WRITE", "cancel_order": "WRITE", "send_email": "WRITE",
    "add_ticket_note": "WRITE",  # internal, low risk, but still a write
    "escalate_to_human": "META", "finish": "META",
}


# ---------------------------------------------------------------------------
# Implementations. Each takes (args, run_id) and returns a JSON-able dict.
# ---------------------------------------------------------------------------
def _get_ticket(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        t = db.row(c.execute("SELECT * FROM tickets WHERE id=?", (a["ticket_id"],)).fetchone())
    if not t:
        raise ToolError(f"ticket {a['ticket_id']} not found")
    t.pop("tag", None)  # scenario label is for the UI, never for the model
    # Ticket text is untrusted user content: wrap it so the model sees the boundary.
    t["body"] = f"<untrusted_customer_message>\n{t['body']}\n</untrusted_customer_message>"
    return t


def _lookup_customer(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        cu = db.row(c.execute("SELECT * FROM customers WHERE id=?", (a["customer_id"],)).fetchone())
        if not cu:
            raise ToolError(f"customer {a['customer_id']} not found")
        cu["orders"] = db.rows(c.execute(
            "SELECT id,items,total,status,refunded FROM orders WHERE customer_id=?", (a["customer_id"],)))
    return cu


def _get_order(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        o = db.row(c.execute("SELECT * FROM orders WHERE id=?", (a["order_id"],)).fetchone())
    if not o:
        raise ToolError(f"order {a['order_id']} not found")
    o.pop("gateway_profile", None)
    o["refundable"] = round(o["total"] - o["refunded"], 2)
    return o


def _issue_refund(a: dict, run_id: str) -> dict:
    amount = float(a["amount"])
    if amount <= 0:
        raise ToolError("amount must be positive")
    with db.tx() as c:
        o = db.row(c.execute("SELECT * FROM orders WHERE id=?", (a["order_id"],)).fetchone())
        if not o:
            raise ToolError(f"order {a['order_id']} not found")
        if amount > o["total"] - o["refunded"] + 1e-9:
            raise ToolError(f"amount {amount:.2f} exceeds refundable {o['total'] - o['refunded']:.2f}")
        # Simulated payment gateway. The 'flaky' profile always times out: this
        # is the failure-recovery demo, the agent must give up and escalate.
        if o["gateway_profile"] == "flaky":
            raise TransientToolError("payment gateway timeout (HTTP 504) - refund NOT processed")
        c.execute("UPDATE orders SET refunded = refunded + ? WHERE id=?", (amount, a["order_id"]))
        c.execute("INSERT INTO refunds (order_id,amount,reason,run_id,created_at) VALUES (?,?,?,?,?)",
                  (a["order_id"], amount, a.get("reason", ""), run_id, db.now_iso()))
        rid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    return {"refund_id": f"RF-{rid:04d}", "order_id": a["order_id"], "amount": amount, "status": "processed"}


def _cancel_order(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        o = db.row(c.execute("SELECT * FROM orders WHERE id=?", (a["order_id"],)).fetchone())
        if not o:
            raise ToolError(f"order {a['order_id']} not found")
        if o["status"] in ("delivered", "cancelled"):
            raise ToolError(f"order is already {o['status']}; cannot cancel")
        c.execute("UPDATE orders SET status='cancelled' WHERE id=?", (a["order_id"],))
    return {"order_id": a["order_id"], "status": "cancelled", "previous_status": o["status"]}


def _send_email(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        c.execute("INSERT INTO emails (to_addr,subject,body,run_id,created_at) VALUES (?,?,?,?,?)",
                  (a["to"], a["subject"], a["body"], run_id, db.now_iso()))
    return {"status": "queued", "to": a["to"]}


def _add_ticket_note(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        c.execute("INSERT INTO ticket_notes (ticket_id,note,author,run_id,created_at) VALUES (?,?,?,?,?)",
                  (a["ticket_id"], a["note"], "agent", run_id, db.now_iso()))
    return {"status": "added"}


def _escalate(a: dict, run_id: str) -> dict:
    with db.tx() as c:
        c.execute("UPDATE tickets SET status='escalated' WHERE id=?", (a["ticket_id"],))
        c.execute("INSERT INTO ticket_notes (ticket_id,note,author,run_id,created_at) VALUES (?,?,?,?,?)",
                  (a["ticket_id"], "ESCALATED: " + str(a["reason"]), "agent", run_id, db.now_iso()))
    return {"status": "escalated", "queue": "tier-2-human"}


def _finish(a: dict, run_id: str) -> dict:
    return {"status": "done"}


IMPLS: dict[str, Callable[[dict, str], dict]] = {
    "get_ticket": _get_ticket, "lookup_customer": _lookup_customer, "get_order": _get_order,
    "issue_refund": _issue_refund, "cancel_order": _cancel_order, "send_email": _send_email,
    "add_ticket_note": _add_ticket_note, "escalate_to_human": _escalate, "finish": _finish,
}


def execute(name: str, args: dict, run_id: str) -> dict:
    if name not in IMPLS:
        raise ToolError(f"unknown tool {name}")
    return IMPLS[name](args, run_id)
