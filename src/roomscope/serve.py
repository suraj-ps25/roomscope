"""A page on this computer for people who don't use a terminal: drop the capture in, get the plan.

  roomscope serve            # then open http://127.0.0.1:8765 (--open does it for you)

The browser sends the capture file by file (a Stray Scanner folder, a .zip of one, a video,
or one folder of photos per room) to a job folder under runs/uploads/, then the same
`roomscope run` command runs on it, and the page shows the plan, every measurement with its
interval, and plan.json to download. It listens on 127.0.0.1 only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

UPLOADS = Path("runs") / "uploads"
LOG_LINES = 40
JOBS: dict[str, dict] = {}


def safe_relative(path: str) -> Path | None:
    """The upload's path inside its job folder, or None for anything that would leave it."""
    parts = [p for p in PurePosixPath(unquote(path).replace("\\", "/")).parts if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith("/") for p in parts) or parts[0].endswith(":"):
        return None
    return Path(*parts)


def unpack(capture: Path) -> None:
    """A single uploaded .zip (how a Stray Scanner export often travels) becomes its folder."""
    files = [p for p in capture.rglob("*") if p.is_file()]
    if len(files) != 1 or files[0].suffix.lower() != ".zip":
        return
    archive = files[0]
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = safe_relative(member.filename)
            if target is None or member.is_dir() or target.parts[0] == "__MACOSX":
                continue
            (capture / target).parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, (capture / target).open("wb") as sink:
                shutil.copyfileobj(source, sink)
    archive.unlink()


def capture_root(capture: Path) -> Path:
    """The folder the browser sent wraps the capture in its own name; look inside it."""
    children = [p for p in capture.iterdir() if not p.name.startswith(".")]
    return children[0] if len(children) == 1 and children[0].is_dir() else capture


def run_job(job_id: str) -> None:
    job = JOBS[job_id]
    folder = UPLOADS / job_id
    try:
        unpack(folder / "capture")
        command = [sys.executable, "-m", "roomscope.cli", "run", str(capture_root(folder / "capture")),
                   "--out", str(folder / "result")]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            job["log"] = (job["log"] + [line.rstrip()])[-LOG_LINES:]
        process.wait()
        job["state"] = "done" if process.returncode == 0 and (folder / "result" / "plan.json").exists() else "failed"
    except Exception as error:  # surfaced on the page, not as a server crash
        job["log"].append(f"error: {error}")
        job["state"] = "failed"
    job["finished"] = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        self._send(json.dumps(data).encode(), "application/json", status)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = url.path.strip("/").split("/")
        if url.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif len(parts) == 2 and parts[0] == "status" and parts[1] in JOBS:
            job = JOBS[parts[1]]
            self._json({"state": job["state"], "log": job["log"],
                        "elapsed_s": round((job.get("finished") or time.time()) - job["started"], 1)})
        elif len(parts) == 3 and parts[0] == "result" and parts[1] in JOBS and parts[2] in ("plan.png", "plan.json"):
            path = UPLOADS / parts[1] / "result" / parts[2]
            if path.exists():
                kind = "image/png" if parts[2].endswith(".png") else "application/json"
                self._send(path.read_bytes(), kind)
            else:
                self._json({"error": "not ready"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        parts = url.path.strip("/").split("/")
        if url.path == "/job":
            job_id = uuid.uuid4().hex[:12]
            (UPLOADS / job_id / "capture").mkdir(parents=True)
            JOBS[job_id] = {"state": "uploading", "log": [], "started": time.time()}
            self._json({"job": job_id})
        elif len(parts) == 2 and parts[0] == "upload" and parts[1] in JOBS:
            relative = safe_relative(parse_qs(url.query).get("path", [""])[0])
            if relative is None:
                self._json({"error": "bad path"}, 400)
                return
            target = UPLOADS / parts[1] / "capture" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            remaining = int(self.headers.get("Content-Length", 0))
            with target.open("wb") as sink:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    sink.write(chunk)
                    remaining -= len(chunk)
            self._json({"ok": True})
        elif len(parts) == 2 and parts[0] == "run" and parts[1] in JOBS and JOBS[parts[1]]["state"] == "uploading":
            JOBS[parts[1]]["state"] = "running"
            threading.Thread(target=run_job, args=(parts[1],), daemon=True).start()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


def serve(port: int = 8765, open_browser: bool = False) -> int:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = f"http://127.0.0.1:{port}"
    print(f"roomscope: open {address} in a browser (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>roomscope</title>
<style>
:root{--ink:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--accent:#0a66c2;--bg:#fbfbfd}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:860px;margin:0 auto;padding:32px 20px}h1{font-size:26px;margin:0 0 4px}p.lead{color:var(--muted);margin:0 0 24px}
.drop{border:2px dashed var(--line);border-radius:14px;padding:36px 20px;text-align:center;background:#fff;transition:.15s}
.drop.over{border-color:var(--accent);background:#f0f6fc}.drop b{display:block;font-size:17px;margin-bottom:6px}
.pick{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:14px}
button,label.btn{font:inherit;border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 14px;cursor:pointer}
button.go{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600;padding:10px 22px}
button:disabled{opacity:.45;cursor:default}input[type=file]{display:none}
.chosen{margin:14px 0;color:var(--muted)}#status{margin-top:18px}.bar{height:8px;background:#e8e8ed;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);width:0}pre{background:#fff;border:1px solid var(--line);border-radius:9px;
padding:10px;font-size:12px;max-height:180px;overflow:auto;white-space:pre-wrap}img{max-width:100%;border:1px solid var(--line);border-radius:9px;background:#fff}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
.wrap{overflow-x:auto}.help{color:var(--muted);font-size:13px;margin-top:22px}.help li{margin:4px 0}
</style></head><body><main>
<h1>roomscope</h1>
<p class="lead">Drop in what you captured, press <b>Make the floor plan</b>, wait a few minutes.</p>
<div class="drop" id="drop"><b>Drop the capture here</b>
<span>the Stray Scanner folder (or its .zip), the video, or the folder holding one folder of photos per room</span>
<div class="pick"><label class="btn">Choose a folder…<input type="file" id="folder" webkitdirectory multiple></label>
<label class="btn">Choose a file (video or .zip)…<input type="file" id="file"></label></div></div>
<div class="chosen" id="chosen">Nothing chosen yet.</div>
<button class="go" id="go" disabled>Make the floor plan</button>
<div id="status"></div><div id="result"></div>
<ul class="help"><li>Everything stays on this computer.</li>
<li>How to capture: one page, <code>docs/capture_protocol.md</code>.</li>
<li>A number like 2.74 ± 0.26 m means the true value is inside that range 9 times out of 10.</li></ul>
</main><script>
let files=[];const $=id=>document.getElementById(id);
function choose(list){files=list;const mb=list.reduce((s,f)=>s+f.file.size,0)/1e6;
 $('chosen').textContent=list.length?`${list.length} file${list.length>1?'s':''}, ${mb.toFixed(0)} MB`:'Nothing chosen yet.';$('go').disabled=!list.length}
$('folder').onchange=e=>choose([...e.target.files].map(f=>({file:f,path:f.webkitRelativePath||f.name})));
$('file').onchange=e=>choose([...e.target.files].map(f=>({file:f,path:f.name})));
async function walk(entry,prefix,out){if(entry.isFile){await new Promise(r=>entry.file(f=>{out.push({file:f,path:prefix+entry.name});r()}));}
 else if(entry.isDirectory){const reader=entry.createReader();let batch;do{batch=await new Promise(r=>reader.readEntries(r));
 for(const e of batch)await walk(e,prefix+entry.name+'/',out)}while(batch.length)}}
const drop=$('drop');drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=async e=>{e.preventDefault();drop.classList.remove('over');const out=[];
 for(const item of [...e.dataTransfer.items]){const entry=item.webkitGetAsEntry&&item.webkitGetAsEntry();if(entry)await walk(entry,'',out)}choose(out)};
$('go').onclick=async()=>{$('go').disabled=true;$('result').innerHTML='';
 const {job}=await (await fetch('/job',{method:'POST'})).json();let sent=0;
 const show=t=>$('status').innerHTML=`<p>${t}</p><div class="bar"><i style="width:${100*sent/files.length}%"></i></div>`;
 const queue=[...files];async function worker(){while(queue.length){const f=queue.shift();
  await fetch(`/upload/${job}?path=${encodeURIComponent(f.path)}`,{method:'POST',body:f.file});sent++;if(sent%20==0||sent==files.length)show(`Copying ${sent} of ${files.length} files…`)}}
 show('Copying…');await Promise.all(Array.from({length:6},worker));await fetch(`/run/${job}`,{method:'POST'});
 const poll=setInterval(async()=>{const s=await (await fetch(`/status/${job}`)).json();
  $('status').innerHTML=`<p>${s.state=='running'?'Measuring…':s.state=='done'?'Done.':'It didn\'t work.'} (${Math.round(s.elapsed_s)} s)</p><pre>${s.log.join('\n').replace(/</g,'&lt;')}</pre>`;
  if(s.state!='running'){clearInterval(poll);$('go').disabled=false;if(s.state=='done')render(job)}},1500)};
const pm=m=>`${m.value.toFixed(2)} ± ${((m.ci_high-m.ci_low)/2).toFixed(2)}`;
async function render(job){const plan=await (await fetch(`/result/${job}/plan.json`)).json();
 let rows=plan.rooms.map(r=>`<tr><td>${r.label}</td><td>${pm(r.floor_area)} m²</td><td>${pm(r.ceiling_height)} m</td><td>${r.walls.map(w=>pm(w.length)).join('<br>')}</td><td>${r.openings.map(o=>o.type+' '+pm(o.width)).join('<br>')||'–'}</td></tr>`).join('');
 const notes=(plan.capture.notes||[]).map(n=>`<li>${n}</li>`).join('');
 $('result').innerHTML=`<h2>Floor plan</h2><img src="/result/${job}/plan.png?${Date.now()}">
 <p>${plan.rooms.length} rooms, footprint ${pm(plan.property.footprint_area)} m². <a href="/result/${job}/plan.json" download="plan.json">Download plan.json</a></p>
 <div class="wrap"><table><tr><th>Room</th><th>Floor area</th><th>Ceiling</th><th>Walls (m)</th><th>Doors and windows (m)</th></tr>${rows}</table></div>
 ${notes?`<p>Notes from the capture:</p><ul>${notes}</ul>`:''}`}
</script></body></html>
"""
