"""FastAPI app: REST + SSE for the guardrailed support agent, and the static UI."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .agent import AgentRunner
from .config import reload_settings, settings
from .guardrails import Policy, scan_for_injection
from .llm import make_provider

STATIC = Path(__file__).resolve().parent.parent / "static"
runner: AgentRunner


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    db.init_db()
    # Runs that were live when the previous process died can never resume: mark them.
    with db.tx() as c:
        c.execute("UPDATE runs SET status='stopped', summary='server restarted mid-run', finished_at=? "
                  "WHERE status IN ('running','awaiting_approval')", (db.now_iso(),))
    runner = AgentRunner(make_provider(), Policy())
    yield


app = FastAPI(title="Sentinel - guardrailed support agent", lifespan=lifespan)


# ----------------------------------------------------------------------------- reads
@app.get("/api/meta")
def meta():
    return {
        "provider": runner.provider.name,
        "policy": runner.policy.describe(),
        "limits": {
            "refund_auto_limit": settings.refund_auto_limit,
            "refund_approval_limit": settings.refund_approval_limit,
            "refund_run_cap": settings.refund_run_cap,
            "max_steps": settings.max_steps, "max_tool_calls": settings.max_tool_calls,
        },
    }


@app.get("/api/tickets")
def tickets():
    with db.tx() as c:
        rows = db.rows(c.execute(
            "SELECT t.*, cu.name AS customer_name, cu.email AS customer_email "
            "FROM tickets t JOIN customers cu ON cu.id = t.customer_id ORDER BY t.id"))
    for t in rows:
        t["findings"] = [{"label": f.label, "span": list(f.span)} for f in scan_for_injection(t["body"])]
    return rows


@app.get("/api/world")
def world():
    """Current state of the synthetic shop, so the UI can show side effects."""
    with db.tx() as c:
        return {
            "orders": db.rows(c.execute("SELECT id,customer_id,items,total,status,refunded FROM orders ORDER BY id")),
            "refunds": db.rows(c.execute("SELECT * FROM refunds ORDER BY id DESC")),
            "emails": db.rows(c.execute("SELECT * FROM emails ORDER BY id DESC")),
            "notes": db.rows(c.execute("SELECT * FROM ticket_notes ORDER BY id DESC")),
        }


@app.get("/api/runs")
def list_runs():
    with db.tx() as c:
        return db.rows(c.execute("SELECT * FROM runs ORDER BY started_at DESC"))


def _run_view(run_id: str) -> dict:
    run = runner.runs.get(run_id)
    if run:
        return {"id": run.id, "ticket_id": run.ticket_id, "provider": run.provider, "status": run.status,
                "risk_level": run.risk_level, "summary": run.summary, "events": run.events,
                "pending": None if not run.pending else {
                    "approval_id": run.pending.approval_id, "tool": run.pending.tool,
                    "args": run.pending.args, "rule": run.pending.rule, "reason": run.pending.reason}}
    with db.tx() as c:
        r = db.row(c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())
        if not r:
            raise HTTPException(404, "run not found")
        evs = db.rows(c.execute("SELECT seq,ts,type,payload FROM audit_events WHERE run_id=? ORDER BY seq", (run_id,)))
    r["events"] = [{"seq": e["seq"], "ts": e["ts"], "type": e["type"], "run_id": run_id, **json.loads(e["payload"])} for e in evs]
    r["pending"] = None
    return r


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _run_view(run_id)


@app.get("/api/runs/{run_id}/audit.json")
def audit_export(run_id: str):
    v = _run_view(run_id)
    return JSONResponse(v, headers={"Content-Disposition": f'attachment; filename="{run_id}-audit.json"'})


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str):
    run = runner.runs.get(run_id)
    if not run:
        raise HTTPException(404, "run not live")
    q: asyncio.Queue = asyncio.Queue()
    run.subscribers.append(q)

    async def gen():
        try:
            for ev in list(run.events):  # replay
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if run.status in ("completed", "escalated", "stopped", "error"):
                        break
                    continue
                yield f"data: {json.dumps(ev, default=str)}\n\n"
                if ev["type"] == "status" and ev["status"] in ("completed", "escalated", "stopped", "error"):
                    break
        finally:
            if q in run.subscribers:
                run.subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------- writes
# These must be `async def`: they touch asyncio tasks/futures and have to run on the
# event loop, not in Starlette's threadpool.
class StartRun(BaseModel):
    ticket_id: str


@app.post("/api/runs")
async def start_run(body: StartRun):
    with db.tx() as c:
        if not c.execute("SELECT 1 FROM tickets WHERE id=?", (body.ticket_id,)).fetchone():
            raise HTTPException(404, "ticket not found")
    run = runner.start(body.ticket_id)
    return {"run_id": run.id}


class ApprovalBody(BaseModel):
    approved: bool
    note: str = ""


@app.post("/api/runs/{run_id}/approvals/{approval_id}")
async def resolve(run_id: str, approval_id: str, body: ApprovalBody):
    try:
        runner.resolve_approval(run_id, approval_id, body.approved, body.note)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@app.post("/api/runs/{run_id}/stop")
async def stop(run_id: str):
    if run_id not in runner.runs:
        raise HTTPException(404, "run not live")
    runner.stop(run_id)
    return {"ok": True}


@app.post("/api/policy/reload")
async def policy_reload():
    reload_settings()
    runner.policy = Policy()
    return {
        "ok": True,
        "limits": {
            "refund_auto_limit": settings.refund_auto_limit,
            "refund_approval_limit": settings.refund_approval_limit,
            "refund_run_cap": settings.refund_run_cap,
            "max_steps": settings.max_steps,
            "max_tool_calls": settings.max_tool_calls,
        },
    }


@app.post("/api/reset")
async def reset():
    for r in list(runner.runs.values()):
        if r.task and not r.task.done():
            r.task.cancel()
    runner.runs.clear()
    db.reset_demo()
    return {"ok": True}


# ------------------------------------------------------------------------------- ui
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
