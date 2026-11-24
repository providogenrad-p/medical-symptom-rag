"""
medical_web.py - ChatGPT-style web interface for the Medical Symptom Analysis RAG demo.

It reuses the whole pipeline from medical_rag_demo.py (same folder) and adds:
  * streaming answers (text appears word by word)
  * red-flag warning banner, clickable [n] citations, collapsible sources with MedlinePlus links
  * follow-up messages in the same chat ("I also have a rash") - earlier symptoms are kept
  * Short / Detailed answer switch

Run (same virtual environment as the demo):
    pip install fastapi uvicorn
    python medical_web.py            # opens http://127.0.0.1:8000 in your browser

    *** Educational prototype - not a medical device - not medical advice ***
"""

import json
import os
import re
import threading
import time
import webbrowser

import medical_rag_demo as rag
from langchain_core.output_parsers import StrOutputParser

VERSION = "web v1.6"
PORT = int(os.getenv("PORT", "8000"))
MAX_TURNS = 4          # how many earlier user messages are combined with the new one


FOLLOW_UP = re.compile(r"^(and|also|plus|but|still|now|then|what about|what if|i also|i've also|"
                       r"i have also|it also|it's|it is|the pain|the fever|same)\b", re.I)


def is_follow_up(message: str) -> bool:
    """Short messages or ones starting like a continuation ("I also have a rash") build on the
    earlier symptoms; a long new description is treated as a NEW problem."""
    return len(message.split()) <= 8 or bool(FOLLOW_UP.match(message.strip()))


def chat_events(messages: list[str], detailed: bool, state: dict):
    """Yields JSON events: meta -> sources -> token* -> done."""
    turns = [m.strip()[:600] for m in messages if m.strip()][-MAX_TURNS:]
    current = turns[-1] if turns else ""
    # The NEW message must be relevant on its own: "what is the time in Bulgaria" is refused
    # even if the chat contained symptoms before.
    if not rag.retrieve(state["vs"], rag.scrub_phi(current)):
        yield {"type": "meta", "red_flags": rag.detect_red_flags(current), "input": current, "follow_up": False}
        yield {"type": "sources", "sources": [], "retrieval_ms": 0, "expanded": []}
        yield {"type": "token", "text": "I can only answer questions about symptoms and health topics from the "
                                        "MedlinePlus knowledge base, and I found nothing relevant for this message."}
        yield {"type": "done", "warnings": [], "retrieval_ms": 0}
        return
    follow = len(turns) > 1 and is_follow_up(current)
    combined = ". ".join(turns) if follow else current
    patient = rag.scrub_phi(combined)
    flags = rag.detect_red_flags(combined)
    yield {"type": "meta", "red_flags": flags, "input": patient, "follow_up": follow}

    t0 = time.perf_counter()
    hits, expanded = rag.retrieve_expanded(state["vs"], patient, state["llm"])
    retrieval_ms = round((time.perf_counter() - t0) * 1000)
    sources = [{"n": i, "title": d.metadata.get("condition") or d.metadata["source"],
                "section": d.metadata.get("section") or "", "url": d.metadata.get("url", ""),
                "relevance": round(s, 2), "text": d.page_content[:600]}
               for i, (d, s) in enumerate(hits, 1)]
    yield {"type": "sources", "sources": sources, "retrieval_ms": retrieval_ms, "expanded": expanded}

    if not hits:
        yield {"type": "token", "text": "I couldn't find sufficiently relevant information in the medical "
                                        "knowledge base for this, so I won't guess. Please consult a "
                                        "healthcare professional."}
        yield {"type": "done", "warnings": [], "retrieval_ms": retrieval_ms}
        return
    if state["llm"] is None:
        yield {"type": "token", "text": "GROQ_API_KEY is not set (add it to the .env file). "
                                        "Showing the retrieved sources only."}
        yield {"type": "done", "warnings": [], "retrieval_ms": retrieval_ms}
        return

    chain = rag.PROMPT | state["llm"] | StrOutputParser()
    inputs = {"context": rag.format_context(hits), "patient": patient,
              "red_flags": "; ".join(flags) or "none detected by rules",
              "format_rules": rag.FORMAT_DETAILED if detailed else rag.FORMAT_SHORT}
    parts, t1, first = [], time.perf_counter(), None
    try:
        for piece in chain.stream(inputs):
            if piece:
                first = first or time.perf_counter()
                parts.append(piece)
                yield {"type": "token", "text": piece}
    except Exception as e:      # never crash the live demo
        yield {"type": "token", "text": f"\n\n_LLM error: {e}_"}
    answer = re.sub(r"[【〔［]\s*(\d+)\s*[】〕］]", r"[\1]", "".join(parts))
    warnings = rag.validate_citations(answer, len(hits)) if answer else []
    yield {"type": "done", "warnings": warnings, "retrieval_ms": retrieval_ms,
           "first_token_ms": round((first - t1) * 1000) if first else None,
           "total_ms": round((time.perf_counter() - t0) * 1000)}


def create_app():
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel

    state = {}

    @asynccontextmanager
    async def lifespan(_app):
        rag.ensure_seed_kb()
        state["vs"] = rag.get_vectorstore()
        if not state["vs"].get(limit=1)["ids"]:
            rag.ingest(state["vs"])
        state["llm"] = rag.get_llm()
        rag.retrieve(state["vs"], "warm up")          # load the embedding model now, not on first question
        docs = {m["source"] for m in state["vs"].get(include=["metadatas"])["metadatas"]}
        state["topics"] = len(docs)
        state["mplus"] = any(d.startswith("mplus_") for d in docs)
        print(f"[ready] {len(docs)} documents | LLM: {'on' if state['llm'] else 'OFF (no GROQ_API_KEY)'}")
        print(f"[ready] open http://127.0.0.1:{PORT}")
        yield

    app = FastAPI(title="Medical Symptom Assistant", lifespan=lifespan)

    class Chat(BaseModel):
        messages: list[str]
        detailed: bool = False

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/info")
    def info():
        return {"version": f"{rag.VERSION} / {VERSION}", "topics": state["topics"],
                "source": "MedlinePlus (U.S. National Library of Medicine)" if state["mplus"] else "demo knowledge base",
                "llm": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") if state["llm"] else None,
                "disclaimer": rag.DISCLAIMER}

    @app.post("/api/chat")
    def chat(body: Chat):
        def gen():
            for ev in chat_events(body.messages, body.detailed, state):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medical Symptom Assistant</title>
<style>
:root{--side:#f9f9f9;--bg:#fff;--ink:#0d0d0d;--muted:#6b6b6b;--line:#e5e5e5;--accent:#6a1b9a;--red:#c62828;--redbg:#fdecea;--chip:#f0e6f6}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{font:15.5px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);background:var(--bg);display:flex}
aside{width:260px;background:var(--side);border-right:1px solid var(--line);display:flex;flex-direction:column;padding:12px;gap:8px;flex-shrink:0}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;padding:6px 8px 10px}
.logo{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,var(--accent),#d0021b);display:grid;place-items:center;color:#fff;font-size:16px}
.newchat{display:flex;align-items:center;gap:8px;border:1px solid var(--line);background:#fff;border-radius:10px;padding:9px 12px;cursor:pointer;font:inherit;font-size:14px}
.newchat:hover{background:#f0f0f0}
.side-h{font-size:12px;color:var(--muted);margin:14px 8px 2px;font-weight:600}
.hist{flex:1;overflow:auto}.hist div{padding:7px 10px;border-radius:8px;font-size:13.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer}
.hist div:hover,.hist div.on{background:#ececec}
.side-foot{font-size:11.5px;color:var(--muted);padding:8px;border-top:1px solid var(--line);line-height:1.45}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{display:flex;align-items:center;justify-content:space-between;padding:10px 20px;border-bottom:1px solid var(--line)}
header b{font-size:16px}.toggle{display:flex;background:#f0f0f0;border-radius:999px;padding:3px}
.toggle button{border:0;background:none;padding:5px 12px;border-radius:999px;font:inherit;font-size:13px;cursor:pointer;color:var(--muted)}
.toggle button.on{background:#fff;color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.12)}
#scroll{flex:1;overflow:auto}
.col{max-width:780px;margin:0 auto;padding:24px 20px 10px}
.empty{text-align:center;margin-top:12vh}.empty h1{font-size:28px;font-weight:600;margin:0 0 6px}.empty p{color:var(--muted);margin:0 0 28px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:left}
.card{border:1px solid var(--line);border-radius:14px;padding:12px 14px;cursor:pointer;font-size:14px}
.card:hover{background:#f7f7f7}.card span{display:block;color:var(--muted);font-size:13px}
.msg{margin:18px 0}.user{display:flex;justify-content:flex-end}
.user .bubble{background:#f1f1f1;border-radius:18px;padding:10px 16px;max-width:75%;white-space:pre-wrap}
.bot{display:flex;gap:12px}.av{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#d0021b);flex-shrink:0;display:grid;place-items:center;color:#fff;font-size:14px;margin-top:2px}
.body{flex:1;min-width:0}
.flag{background:var(--redbg);border:1px solid #f5c2c0;color:var(--red);border-radius:12px;padding:10px 14px;margin-bottom:10px;font-weight:600}
.flag small{display:block;font-weight:400;color:#7f1d1d}
.ans p{margin:0 0 10px}.ans ol,.ans ul{margin:0 0 10px;padding-left:22px}.ans li{margin:3px 0}
.ans table{border-collapse:collapse;margin:8px 0;font-size:14px}.ans th,.ans td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
.ans h3{font-size:16px;margin:14px 0 6px}
.cite{display:inline-grid;place-items:center;min-width:18px;height:18px;padding:0 5px;margin:0 1px;border-radius:9px;background:var(--chip);color:var(--accent);font-size:11px;font-weight:700;cursor:pointer;vertical-align:2px}
.cite.bad{background:var(--redbg);color:var(--red)}
.cursor::after{content:"▍";animation:b 1s steps(1) infinite;color:var(--muted)}@keyframes b{50%{opacity:0}}
.thinking{color:var(--muted);font-size:14px}
details{margin-top:8px;border:1px solid var(--line);border-radius:12px;font-size:13.5px}
summary{cursor:pointer;padding:8px 12px;color:var(--muted);list-style:none}summary::-webkit-details-marker{display:none}
summary:before{content:"▸ ";}details[open] summary:before{content:"▾ ";}
.src{padding:8px 12px;border-top:1px solid var(--line)}.src.hl{background:#faf5fc}
.src .t{display:flex;gap:8px;align-items:center}.src .n{font-weight:700;color:var(--accent)}
.src a{color:var(--ink);font-weight:600;text-decoration:none}.src a:hover{text-decoration:underline}
.rel{margin-left:auto;font-size:12px;color:var(--muted)}.src .x{color:var(--muted);font-size:12.5px;margin-top:3px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.meta{font-size:12px;color:var(--muted);margin-top:6px}.warn{color:var(--red)}
.composer{padding:8px 20px 14px}.box{max-width:780px;margin:0 auto;border:1px solid #d9d9d9;border-radius:24px;display:flex;align-items:flex-end;padding:8px 8px 8px 18px;box-shadow:0 2px 10px rgba(0,0,0,.05)}
textarea{flex:1;border:0;outline:0;resize:none;font:inherit;max-height:180px;padding:6px 0;background:transparent}
.send{width:36px;height:36px;border-radius:50%;border:0;background:var(--ink);color:#fff;font-size:17px;cursor:pointer}
.send:disabled{background:#cfcfcf;cursor:default}
.fine{max-width:780px;margin:6px auto 0;text-align:center;font-size:11.5px;color:var(--muted)}
@media(max-width:800px){aside{display:none}.cards{grid-template-columns:1fr}}
</style></head><body>
<aside>
  <div class="brand"><div class="logo">✚</div>Symptom Assistant</div>
  <button class="newchat" id="new">✎&nbsp; New chat</button>
  <div class="side-h">Chats</div><div class="hist" id="hist"></div>
  <div class="side-foot" id="kb">Loading knowledge base…</div>
</aside>
<main>
  <header><b>Medical Symptom Assistant</b>
    <div class="toggle"><button id="short" class="on">Short</button><button id="detailed">Detailed</button></div></header>
  <div id="scroll"><div class="col" id="col"></div></div>
  <div class="composer"><div class="box"><textarea id="q" rows="1" placeholder="Describe your symptoms…"></textarea><button class="send" id="send" title="Send">↑</button></div>
    <div class="fine">Educational prototype, not a medical device. It can make mistakes. In an emergency call 112.</div></div>
</main>
<script>
const $ = s => document.querySelector(s);
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const EXAMPLES = [
  ["Fever, severe headache and a stiff neck", "light hurts my eyes, age 19"],
  ["Burning when urinating", "and I need to go very often"],
  ["Cough, fever and chest pain when breathing", "for 3 days"],
  ["Sudden diarrhea and vomiting", "after eating at a restaurant"]];
let chats = [], cur = null, detailed = false, busy = false;

fetch('/api/info').then(r => r.json()).then(i => {
  $('#kb').innerHTML = `<b>Knowledge base:</b> ${i.topics} topics from ${esc(i.source)}<br><b>Model:</b> ${esc(i.llm || 'not connected')}<br>${esc(i.version)}`; });

function newChat() { cur = {id: Date.now(), title: 'New chat', turns: []}; chats.unshift(cur); render(); renderHist(); $('#q').focus(); }
function renderHist() { $('#hist').innerHTML = chats.filter(c => c.turns.length).map(c =>
  `<div class="${c === cur ? 'on' : ''}" data-id="${c.id}">${esc(c.title)}</div>`).join(''); }
$('#hist').onclick = e => { const d = e.target.closest('[data-id]'); if (!d || busy) return; cur = chats.find(c => c.id == d.dataset.id); render(); renderHist(); };
$('#new').onclick = () => { if (!busy) newChat(); };
$('#short').onclick = () => { detailed = false; $('#short').classList.add('on'); $('#detailed').classList.remove('on'); };
$('#detailed').onclick = () => { detailed = true; $('#detailed').classList.add('on'); $('#short').classList.remove('on'); };

function md(text, n) {
  const lines = esc(text).split('\n'); let html = '', list = null, table = [];
  const inline = s => s.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/(^|[^*])\*(?!\s)(.+?)\*/g, '$1<i>$2</i>')
    .replace(/\[(\d+)\]/g, (m, k) => `<span class="cite${+k > n ? ' bad' : ''}" data-n="${k}">${k}</span>`);
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const flushTable = () => { if (!table.length) return; const rows = table.filter(l => !/^\|?\s*:?-{2,}/.test(l.replace(/^\|/, '')));
    html += '<table>' + rows.map((l, i) => '<tr>' + l.replace(/^\||\|$/g, '').split('|').map(c => i ? `<td>${inline(c.trim())}</td>` : `<th>${inline(c.trim())}</th>`).join('') + '</tr>').join('') + '</table>'; table = []; };
  for (const raw of lines) {
    const l = raw.trim();
    if (l.startsWith('|')) { closeList(); table.push(l); continue; } flushTable();
    let m;
    if ((m = l.match(/^\d+[.)]\s+(.*)/))) { if (list !== 'ol') { closeList(); html += '<ol>'; list = 'ol'; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = l.match(/^[-*•]\s+(.*)/))) { if (list !== 'ul') { closeList(); html += '<ul>'; list = 'ul'; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = l.match(/^#{1,4}\s+(.*)/))) { closeList(); html += `<h3>${inline(m[1])}</h3>`; }
    else if (l) { closeList(); html += `<p>${inline(l)}</p>`; }
  }
  closeList(); flushTable(); return html;
}
function botHTML(t, streaming) {
  const flag = t.flags && t.flags.length ? `<div class="flag">⚠ ${esc(t.flags.join(', '))}<small>Seek emergency care now - call 112 (EU) / 911 (US).</small></div>` : '';
  const ans = t.answer ? `<div class="ans${streaming ? ' cursor' : ''}">${md(t.answer, (t.sources || []).length)}</div>` : `<div class="thinking">${t.sources ? 'Writing…' : 'Searching MedlinePlus…'}</div>`;
  const src = t.sources && t.sources.length ? `<details ${t.open ? 'open' : ''}><summary>Sources (${t.sources.length}) · retrieved in ${t.retrieval_ms} ms${t.expanded && t.expanded.length ? ' · also searched: ' + esc(t.expanded.join(', ')) : ''}</summary>${t.sources.map(s => `
    <div class="src" data-n="${s.n}"><div class="t"><span class="n">${s.n}</span>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>` : `<b>${esc(s.title)}</b>`}
    ${s.section ? `<span style="color:var(--muted)">· ${esc(s.section)}</span>` : ''}<span class="rel">relevance ${s.relevance.toFixed(2)}</span></div><div class="x">${esc(s.text.split('\n').slice(1).join(' '))}</div></div>`).join('')}</details>` : '';
  const meta = t.done ? `<div class="meta">${t.warnings && t.warnings.length ? `<span class="warn">⚠ ${esc(t.warnings.join(' '))}</span>` : '✓ All citations point to retrieved sources'}${t.total_ms ? ` · ${(t.total_ms / 1000).toFixed(1)} s` : ''}</div>` : '';
  const follow = t.follow ? '<div class="meta" style="margin:0 0 6px">↺ Combined with your earlier symptoms in this chat. For a new problem, start a New chat.</div>' : '';
  return `<div class="av">✚</div><div class="body">${follow}${flag}${ans}${src}${meta}</div>`;
}
function render() {
  if (!cur.turns.length) {
    $('#col').innerHTML = `<div class="empty"><h1>How are you feeling today?</h1><p>Describe your symptoms. Answers come from MedlinePlus and cite their sources.</p>
      <div class="cards">${EXAMPLES.map(([a, b]) => `<div class="card" data-q="${esc(a + ', ' + b)}">${esc(a)}<span>${esc(b)}</span></div>`).join('')}</div></div>`;
    return; }
  $('#col').innerHTML = cur.turns.map((t, i) => t.role === 'user' ? `<div class="msg user"><div class="bubble">${esc(t.text)}</div></div>`
    : `<div class="msg bot" id="m${i}">${botHTML(t, false)}</div>`).join('');
  $('#scroll').scrollTop = 1e9;
}
$('#col').addEventListener('click', e => {
  const card = e.target.closest('.card'); if (card) { $('#q').value = card.dataset.q; send(); return; }
  const c = e.target.closest('.cite'); if (!c) return;
  const box = c.closest('.body'), d = box.querySelector('details'); if (!d) return; d.open = true;
  box.querySelectorAll('.src').forEach(s => s.classList.toggle('hl', s.dataset.n === c.dataset.n));
  const s = box.querySelector(`.src[data-n="${c.dataset.n}"]`); if (s) s.scrollIntoView({behavior: 'smooth', block: 'center'});
});
async function send() {
  const q = $('#q').value.trim(); if (!q || busy) return;
  busy = true; $('#send').disabled = true; $('#q').value = ''; autosize();
  if (!cur.turns.length) cur.title = q.slice(0, 40);
  cur.turns.push({role: 'user', text: q});
  const t = {role: 'bot', answer: '', flags: [], sources: null}; cur.turns.push(t);
  render(); renderHist();
  const el = $('#m' + (cur.turns.length - 1)), chat = cur;
  const paint = s => { if (chat === cur) { el.innerHTML = botHTML(t, s); $('#scroll').scrollTop = 1e9; } };
  try {
    const res = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: chat.turns.filter(x => x.role === 'user').map(x => x.text), detailed})});
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream: true}); let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1); if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'meta') { t.flags = ev.red_flags; t.follow = ev.follow_up; }
        else if (ev.type === 'sources') { t.sources = ev.sources; t.retrieval_ms = ev.retrieval_ms; t.expanded = ev.expanded; }
        else if (ev.type === 'token') t.answer += ev.text.replace(/[【〔［]\s*(\d+)\s*[】〕］]/g, '[$1]');
        else if (ev.type === 'done') Object.assign(t, {done: true, warnings: ev.warnings, total_ms: ev.total_ms});
        paint(ev.type !== 'done');
      }
    }
  } catch (e) { t.answer += '\n\nConnection error: ' + e; t.done = true; }
  t.done = true; paint(false); busy = false; $('#send').disabled = false; $('#q').focus();
}
function autosize() { const q = $('#q'); q.style.height = 'auto'; q.style.height = Math.min(q.scrollHeight, 180) + 'px'; }
$('#q').addEventListener('input', autosize);
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
$('#send').onclick = send;
newChat();
</script></body></html>"""


def find_browser(name: str) -> str | None:
    """Finds a browser executable on Windows (Opera is installed per user by default)."""
    if os.getenv("BROWSER_PATH") and os.path.exists(os.environ["BROWSER_PATH"]):
        return os.environ["BROWSER_PATH"]
    local = os.getenv("LOCALAPPDATA", "")
    pf, pf86 = os.getenv("ProgramFiles", r"C:\Program Files"), os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    candidates = {
        "opera": [rf"{local}\Programs\Opera\opera.exe", rf"{local}\Programs\Opera\launcher.exe",
                  rf"{pf}\Opera\launcher.exe", rf"{pf86}\Opera\launcher.exe",
                  rf"{local}\Programs\Opera GX\opera.exe", rf"{local}\Programs\Opera GX\launcher.exe"],
    }.get(name.lower(), [])
    return next((p for p in candidates if p and os.path.exists(p)), None)


def open_browser(url: str) -> None:
    import subprocess
    wanted = os.getenv("OPEN_WITH", "opera")          # set OPEN_WITH=default in .env to use the default browser
    path = find_browser(wanted) if wanted != "default" else None
    if path:
        print(f"[browser] opening {url} in {wanted}")
        subprocess.Popen([path, url])
    else:
        if wanted != "default":
            print(f"[browser] {wanted} not found - using the default browser "
                  f"(or set BROWSER_PATH=C:\\path\\to\\opera.exe in .env)")
        webbrowser.open(url)


def open_when_ready(timeout: float = 180) -> None:
    """Waits until the server really accepts connections (models loaded), then opens the browser."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                open_browser(f"http://127.0.0.1:{PORT}")
                return
        except OSError:
            time.sleep(0.5)
    print(f"[browser] server did not start within {timeout:.0f} s - check the errors above")


def main():
    import uvicorn
    print(f"Medical Symptom Assistant ({rag.VERSION} / {VERSION})")
    print("*** Educational prototype - not a medical device - not medical advice ***")
    if not os.getenv("NO_BROWSER"):
        threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
