/* Sentinel UI: tickets -> run agent -> live audit timeline + approval gate. */
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (n) => "$" + Number(n).toFixed(2);
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

let tickets = [], selected = null, currentRun = null, es = null;

// ------------------------------------------------------------------ boot
async function boot() {
  const meta = await api("/api/meta");
  $("#provider").textContent = "provider: " + meta.provider;
  $("#policy").innerHTML = meta.policy.map((p) => `<li><b>${esc(p.id)}</b>${esc(p.text)}</li>`).join("");
  await loadTickets();
  await loadWorld();
}

async function loadTickets() {
  tickets = await api("/api/tickets");
  $("#tickets").innerHTML = tickets.map((t) => `
    <div class="card ticket ${selected === t.id ? "active" : ""}" data-id="${t.id}">
      <div class="row-between"><span class="subj">${esc(t.subject)}</span><span class="chip tag-${t.tag}">${esc(t.tag)}</span></div>
      <div class="muted">${esc(t.id)} · ${esc(t.customer_name)} · ${esc(t.status)}${t.findings.length ? ` · <span class="danger">⚠ ${t.findings.length} suspicious pattern${t.findings.length > 1 ? "s" : ""}</span>` : ""}</div>
    </div>`).join("");
  document.querySelectorAll(".ticket").forEach((el) => el.onclick = () => selectTicket(el.dataset.id));
  if (selected) selectTicket(selected);
}

function highlight(body, findings) {
  // merge spans, then wrap in <mark>
  const spans = findings.map((f) => f.span).sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const s of spans) {
    const last = merged[merged.length - 1];
    if (last && s[0] <= last[1]) last[1] = Math.max(last[1], s[1]); else merged.push([...s]);
  }
  let out = "", i = 0;
  for (const [a, b] of merged) { out += esc(body.slice(i, a)) + "<mark>" + esc(body.slice(a, b)) + "</mark>"; i = b; }
  return out + esc(body.slice(i));
}

function selectTicket(id) {
  selected = id;
  const t = tickets.find((x) => x.id === id);
  document.querySelectorAll(".ticket").forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  $("#ticket-detail").classList.remove("hidden");
  $("#td-subject").textContent = t.subject;
  $("#td-tag").textContent = t.tag; $("#td-tag").className = "chip tag-" + t.tag;
  $("#td-meta").textContent = `${t.id} · ${t.customer_name} <${t.customer_email}> · status: ${t.status}`;
  $("#td-body").innerHTML = highlight(t.body, t.findings);
  const labels = [...new Set(t.findings.map((f) => f.label))];
  $("#td-findings").innerHTML = labels.map((l) => `<span class="chip">⚠ ${esc(l)}</span>`).join("");
}

// ------------------------------------------------------------------ world
async function loadWorld() {
  const w = await api("/api/world");
  const st = (s) => `<span class="st-${s}">${esc(s)}</span>`;
  let html = `<table><tr><th>Order</th><th>Cust</th><th>Status</th><th class="num">Total</th><th class="num">Refunded</th></tr>` +
    w.orders.map((o) => `<tr><td><code>${o.id}</code></td><td>${o.customer_id}</td><td>${st(o.status)}</td><td class="num">${money(o.total)}</td><td class="num ${o.refunded > 0 ? "danger" : ""}">${money(o.refunded)}</td></tr>`).join("") + `</table>`;
  html += `<h2>Outbox <span class="muted">${w.emails.length} sent</span></h2>` + (w.emails.length ? w.emails.map((e) =>
    `<div class="mail"><div class="to">→ ${esc(e.to_addr)}</div><b>${esc(e.subject)}</b><div class="muted">${esc(e.body)}</div></div>`).join("") : `<p class="muted">nothing sent yet</p>`);
  html += `<h2>Internal notes</h2>` + (w.notes.length ? w.notes.map((n) => `<div class="mail"><div class="to">${esc(n.ticket_id)}</div>${esc(n.note)}</div>`).join("") : `<p class="muted">none</p>`);
  $("#world").innerHTML = html;
}

// ------------------------------------------------------------------ run
$("#run").onclick = async () => {
  if (!selected) return;
  $("#timeline").innerHTML = "";
  $("#approval").classList.add("hidden");
  const { run_id } = await api("/api/runs", { method: "POST", body: JSON.stringify({ ticket_id: selected }) });
  currentRun = run_id;
  $("#stop").classList.remove("hidden");
  $("#audit").classList.remove("hidden"); $("#audit").href = `/api/runs/${run_id}/audit.json`;
  $("#run-risk").classList.add("hidden");
  setStatus("running");
  if (es) es.close();
  es = new EventSource(`/api/runs/${run_id}/stream`);
  es.onmessage = (m) => onEvent(JSON.parse(m.data));
  es.onerror = () => { es.close(); };
};

$("#stop").onclick = () => currentRun && api(`/api/runs/${currentRun}/stop`, { method: "POST" });
$("#reload-policy").onclick = async () => {
  const r = await api("/api/policy/reload", { method: "POST" });
  const meta = await api("/api/meta");
  $("#policy").innerHTML = meta.policy.map((p) => `<li><b>${esc(p.id)}</b>${esc(p.text)}</li>`).join("");
  alert(`Policy reloaded. auto_limit=$${r.limits.refund_auto_limit} approval_limit=$${r.limits.refund_approval_limit}`);
};
$("#reset").onclick = async () => { await api("/api/reset", { method: "POST" }); $("#timeline").innerHTML = ""; setStatus("idle"); $("#approval").classList.add("hidden"); await loadTickets(); await loadWorld(); };

function setStatus(s) {
  const el = $("#run-status"); el.textContent = s.replace("_", " "); el.className = "chip status-" + s;
  if (["completed", "escalated", "stopped", "error", "idle"].includes(s)) $("#stop").classList.add("hidden");
}

function setRisk(r) {
  const el = $("#run-risk"); el.textContent = "risk: " + r; el.className = "chip risk-" + r; el.classList.remove("hidden");
}

const fmtArgs = (a) => Object.entries(a || {}).map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 80 ? v.slice(0, 80) + "…" : v) : JSON.stringify(v)}`).join(", ");

function onEvent(ev) {
  const t = ev.ts.slice(11, 23);
  let cls = "", icon = "•", title = "", sub = "", pre = "";
  switch (ev.type) {
    case "input_scan":
      cls = "ev-scan"; icon = "🔍";
      title = `Input scan → risk <b>${ev.risk_level}</b>`;
      sub = ev.findings.length ? ev.findings.map((f) => `${f.label}: “${f.excerpt}”`).join(" · ") : "no suspicious patterns in customer text";
      setRisk(ev.risk_level); break;
    case "llm_request":
      cls = "ev-llm"; icon = "🧠"; title = `Model step ${ev.step}`; sub = `${ev.messages} messages in context`; break;
    case "note":
      cls = "ev-status"; icon = "📝"; title = esc(ev.text); break;
    case "llm_response":
      cls = "ev-llm"; icon = "💬";
      title = ev.content ? esc(ev.content) : "<span class='muted'>(no text)</span>";
      if (ev.usage && ev.usage.model) title += ` <span class="chip">${esc(ev.usage.model)}${ev.usage.prompt_tokens ? " · " + ev.usage.prompt_tokens + " tok" : ""}</span>`;
      sub = ev.proposed.length ? "proposes: " + ev.proposed.map((p) => `<code>${p.name}(${esc(fmtArgs(p.args))})</code>`).join(", ") : "no tool call";
      break;
    case "policy_decision":
      cls = "ev-" + ev.verdict; icon = ev.verdict === "ALLOW" ? "✅" : ev.verdict === "DENY" ? "⛔" : "✋";
      title = `<span class="verdict ${ev.verdict}">${ev.verdict}</span><code>${ev.tool}</code> <span class="muted">[${ev.kind}]</span>`;
      sub = `<b>${esc(ev.rule)}</b> — ${esc(ev.reason)}`; break;
    case "approval_requested":
      cls = "ev-REQUIRE_APPROVAL"; icon = "🙋"; title = "Waiting for human approval";
      sub = `<code>${ev.tool}(${esc(fmtArgs(ev.args))})</code>`;
      showApproval(ev); break;
    case "approval_resolved":
      cls = ev.approved ? "ev-exec" : "ev-fail"; icon = ev.approved ? "👍" : "👎";
      title = ev.approved ? "Approved by reviewer" : "Rejected by reviewer"; sub = esc(ev.note || "");
      $("#approval").classList.add("hidden"); break;
    case "tool_executed":
      cls = "ev-exec"; icon = "⚙️"; title = `Executed <code>${ev.tool}</code>${ev.attempts > 1 ? ` (attempt ${ev.attempts})` : ""}`;
      pre = JSON.stringify(ev.result, null, 1);
      if (["issue_refund", "cancel_order", "send_email", "add_ticket_note", "escalate_to_human"].includes(ev.tool)) { loadWorld(); loadTickets(); }
      break;
    case "tool_failed":
      cls = "ev-fail"; icon = "💥"; title = `<code>${ev.tool}</code> failed (attempt ${ev.attempt}, ${ev.transient ? "transient" : "permanent"})`; sub = esc(ev.error); break;
    case "safe_stop":
      cls = "ev-fail"; icon = "🛑"; title = "Safe stop"; sub = esc(ev.reason); break;
    case "stopped":
      cls = "ev-fail"; icon = "🛑"; title = "Stopped"; sub = esc(ev.reason); break;
    case "error":
      cls = "ev-fail"; icon = "❗"; title = "Internal error"; sub = esc(ev.error); break;
    case "status":
      setStatus(ev.status);
      if (["completed", "escalated", "stopped", "error"].includes(ev.status)) {
        cls = "ev-status"; icon = "🏁"; title = `Run ${ev.status}`; sub = esc(ev.summary || "");
        loadWorld(); loadTickets();
      } else return;
      break;
    default: return;
  }
  const el = document.createElement("div");
  el.className = "ev " + cls;
  el.innerHTML = `<div class="icon">${icon}</div><div><span class="ts">${t}</span><div class="title">${title}</div>${sub ? `<div class="sub">${sub}</div>` : ""}${pre ? `<pre>${esc(pre)}</pre>` : ""}</div>`;
  $("#timeline").appendChild(el);
  el.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function showApproval(ev) {
  const box = $("#approval");
  box.classList.remove("hidden");
  box.innerHTML = `
    <h3>✋ Approval needed — ${esc(ev.rule)}</h3>
    <div>${esc(ev.reason)}</div>
    <pre>${esc(ev.tool)}(${esc(JSON.stringify(ev.args, null, 1))})</pre>
    <input id="apr-note" placeholder="optional note for the audit log / the agent">
    <div class="actions">
      <button class="ok" id="apr-yes">Approve &amp; execute</button>
      <button class="danger" id="apr-no">Reject</button>
    </div>`;
  const send = (approved) => api(`/api/runs/${currentRun}/approvals/${ev.approval_id}`, { method: "POST", body: JSON.stringify({ approved, note: $("#apr-note").value }) });
  $("#apr-yes").onclick = () => send(true);
  $("#apr-no").onclick = () => send(false);
  box.scrollIntoView({ behavior: "smooth" });
}

boot();
