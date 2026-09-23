"""Live dashboard for the gemma4-quant pipelines (MLX GPTQ, GGUF, MTP drafter).

Polls the machine running them -- a RunPod pod over SSH, or this machine
with --local -- every --poll-interval seconds (one combined shell call per
poll) and serves an auto-refreshing page on http://localhost:8421:

  - every pipeline's steps (done / running / failed / skipped, durations),
    from $LOG_DIR/pipeline.log (written by pipeline_lib.sh);
  - per-layer calibration grids per component (text attention+MLP, text MoE
    experts, vision, audio) from the calibrate scripts' own
    gptq_progress_*.json resume files -- exact, not guessed from logs;
  - progress inside long steps: calibration batch, splice shard, imatrix
    chunk, llama-quantize tensor;
  - tail of the running step's log, GPU and disk.

Stdlib only (http.server + threading + subprocess) -- nothing to install.
Sibling of nemotron-extreme-quant/poc/pipeline_dashboard.py.

    python poc/pipeline_dashboard.py --pod-host 1.2.3.4 --pod-port 19798 \\
        --pod-ssh-key ~/.runpod/ssh/runpodctl-ssh-key
    python poc/pipeline_dashboard.py --local --work /workspace     # on the pod
    # then open http://localhost:8421
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_status import PROGRESS_FILES, calibration_state, parse_pipeline_log  # noqa: E402

STATE_LOCK = threading.Lock()
STATE: dict = {"work": None, "pipelines": {}, "calibration": None, "variant": None, "running": [],
               "tails": {}, "inner": {}, "gpu": None, "disk": None,
               "last_poll": None, "last_poll_ok": False, "error": None}

# Progress inside long-running steps, parsed from the step's own log tail.
INNER_PATTERNS = [
    ("calibration batch", re.compile(r"--- (\w+) batch (\d+)/(\d+): layers (\[[^\]]*\])")),
    ("calibration example", re.compile(r"^\s+\[(\d+)/(\d+)\]\s*$", re.M)),
    ("splice shard", re.compile(r"Shard (\d+)/(\d+):")),
    ("llama-quantize tensor", re.compile(r"\[\s*(\d+)/\s*(\d+)\]\s+\S+")),
    ("imatrix chunk", re.compile(r"\[(\d+)\][\d.]+")),
]


def inner_progress(text: str) -> list[dict]:
    out = []
    for label, rx in INNER_PATTERNS:
        matches = rx.findall(text)
        if not matches:
            continue
        m = matches[-1]
        if label == "calibration batch":
            out.append({"label": f"{m[0]} batch", "n": int(m[1]), "of": int(m[2]), "note": f"layers {m[3]}"})
        elif label == "imatrix chunk":
            out.append({"label": label, "n": int(m), "of": 200, "note": ""})
        else:
            out.append({"label": label, "n": int(m[0]), "of": int(m[1]), "note": ""})
    return out


def remote_script(work: str) -> str:
    # Plain POSIX shell -- identical over SSH or locally (--local).
    return f"""
W={work}; L=$W/logs
echo "===PIPELINE_LOG==="; tail -c 200000 $L/pipeline.log 2>/dev/null
SNAP=$(grep ' mlx DONE download ' $L/pipeline.log 2>/dev/null | tail -1 | cut -d' ' -f6)
echo "===CONFIG==="; [ -n "$SNAP" ] && cat "$SNAP/config.json" 2>/dev/null
CORR=$(grep ' mlx START _pipeline ' $L/pipeline.log 2>/dev/null | tail -1 | grep -o 'CORRECTED=[^ ]*' | cut -d= -f2)
for f in $CORR/gptq_progress_*.json; do [ -f "$f" ] && echo "===PROGRESS:$(basename $f .json | sed 's/gptq_progress_//')===" && cat "$f"; done
for f in $(ls -t $L/*_*.log 2>/dev/null | grep -v pipeline.log | head -3); do
  echo "===TAIL:$(basename $f .log)==="; tail -c 6000 "$f"; done
echo "===GPU==="; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "===DISK==="; df -h $W 2>/dev/null | tail -1
"""


def poll(args) -> None:
    script = remote_script(args.work)
    if args.local:
        cmd = ["bash", "-c", script]
    else:
        cmd = ["ssh", "-i", args.pod_ssh_key, "-p", str(args.pod_port), "-o", "ConnectTimeout=8",
               "-o", "StrictHostKeyChecking=no", f"root@{args.pod_host}", script]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout
    except Exception as e:  # keep showing the last good state
        with STATE_LOCK:
            STATE.update(last_poll=time.time(), last_poll_ok=False, error=f"poll failed: {e!r}")
        return

    parts, cur = {}, None
    for line in raw.splitlines(keepends=True):
        m = re.match(r"^===([A-Z_]+(?::[\w.-]+)?)===\n?$", line)
        if m:
            cur = m.group(1)
            parts[cur] = ""
        elif cur:
            parts[cur] += line

    pipelines = parse_pipeline_log(parts.get("PIPELINE_LOG", ""))
    running = [
        f"{p}_{step}" for p, info in pipelines.items() if info["status"] == "running"
        for step, s in info["steps"].items() if s["status"] == "START"
    ]
    variant = None
    mlx = pipelines.get("mlx")
    if mlx:
        m = re.search(r"VARIANT=(\w+)", mlx.get("detail", ""))
        variant = m.group(1) if m else None

    calibration = None
    config_text = parts.get("CONFIG", "").strip()
    if variant in PROGRESS_FILES and config_text:
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError:
            config = None
        if config:
            # calibration_state reads files; feed it the fetched JSONs via a tmp dir.
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                for key, body in parts.items():
                    if key.startswith("PROGRESS:"):
                        (Path(tmp) / f"gptq_progress_{key.split(':', 1)[1]}.json").write_text(body)
                calibration = calibration_state(config, Path(tmp), variant)

    tails = {k.split(":", 1)[1]: v[-4000:] for k, v in parts.items() if k.startswith("TAIL:")}
    inner = {name: inner_progress(tails[name]) for name in running if name in tails}

    with STATE_LOCK:
        STATE.update(
            work=args.work.rstrip("/"), pipelines=pipelines, running=running, variant=variant, calibration=calibration,
            tails=tails, inner=inner,
            gpu=parts.get("GPU", "").strip() or None, disk=parts.get("DISK", "").strip() or None,
            last_poll=time.time(), last_poll_ok=True, error=None,
        )


def poll_loop(args) -> None:
    while True:
        poll(args)
        time.sleep(args.poll_interval)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Gemma 4 Pipeline Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
         background: #0f1115; color: #e6e6e6; }
  @media (prefers-color-scheme: light) { body { background: #f7f7f8; color: #1a1a1a; } }
  h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
  h2 { font-size: 1rem; margin: 1.8rem 0 0.6rem; padding-top: 0.8rem; border-top: 1px solid rgba(128,128,128,0.25); }
  h3 { font-size: 0.85rem; opacity: 0.75; margin: 1rem 0 0.4rem; font-weight: 600; }
  .sub { opacity: 0.6; font-size: 0.85rem; }
  .badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;
           margin-left: 0.4rem; vertical-align: middle; background: #2a3140; }
  .badge.running { background: #1f3b66; } .badge.done { background: #16351f; } .badge.failed { background: #5a1c20; }
  .badge.stale { background: #4a3a10; }
  @media (prefers-color-scheme: light) {
    .badge { background: #e3e6ec; } .badge.running { background: #cfe0fb; } .badge.done { background: #d4f0da; }
    .badge.failed { background: #f6d2d5; } .badge.stale { background: #f5e6c0; } }
  table.steps { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
  table.steps td { padding: 0.28rem 0.5rem; border-bottom: 1px solid rgba(128,128,128,0.18); vertical-align: top; }
  td.step { min-width: 16rem; }
  td.icon { width: 1.4rem; text-align: center; } td.dur { width: 5rem; text-align: right; opacity: 0.7;
    font-variant-numeric: tabular-nums; } td.detail { opacity: 0.6; font-size: 0.8rem; }
  .START { color: #6ea8ff; } .DONE { color: #4ad691; } .FAIL { color: #ff6b6b; } .SKIP { opacity: 0.45; }
  .grid { display: flex; flex-wrap: wrap; gap: 3px; margin: 0.3rem 0 0.8rem; }
  .cell { width: 15px; height: 15px; border-radius: 3px; background: rgba(128,128,128,0.25); }
  .cell.done.text { background: #d64ac2; } .cell.done.text_moe { background: #4ad691; }
  .cell.done.vision { background: #4a7fd6; } .cell.done.audio { background: #e0a030; }
  .cell.current { outline: 2px solid currentColor; animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: 0.4; } }
  .bar { height: 6px; border-radius: 3px; background: rgba(128,128,128,0.25); overflow: hidden; margin: 0.2rem 0 0.6rem; }
  .bar > div { height: 100%; background: #6ea8ff; }
  .inner { font-size: 0.82rem; }
  .meta { font-size: 0.85rem; opacity: 0.65; margin: 1rem 0; }
  pre { background: #1a1d24; color: #b8c4d0; padding: 0.8rem; border-radius: 8px; font-size: 0.72rem;
        overflow-x: auto; max-height: 260px; overflow-y: auto; white-space: pre-wrap; }
  .empty { opacity: 0.5; font-size: 0.85rem; }
</style></head>
<body>
  <h1>Gemma 4 Pipeline Dashboard</h1>
  <div class="sub" id="sub">connecting...</div>
  <div id="pipes"></div>
  <div class="meta" id="meta"></div>
<script>
const TITLES = {mlx: 'MLX: GPTQ calibrate → splice → smoke → publish', gguf: 'GGUF: imatrix + JANG → ollama',
                mtp: 'MTP drafter: parity → 8-bit → bench'};
const ICON = {START: '●', DONE: '✓', FAIL: '✗', SKIP: '–'};
const PASS_LABEL = {text: 'text: attention + dense MLP', text_moe: 'text: MoE experts', vision: 'vision', audio: 'audio'};
const STALE_S = 900;
function fmtDur(s) { if (s == null) return ''; s = Math.round(s);
  return s < 90 ? s + 's' : s < 5400 ? Math.round(s/60) + 'm' : (s/3600).toFixed(1) + 'h'; }
let WORK = null;
function esc(t) {
  t = String(t);
  if (WORK) t = t.split(WORK).join('$WORK');  // pod paths are long; show them relative
  return t.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function calibrationHtml(s, running) {
  if (!s.calibration) return '';
  let html = `<h3>Calibration (${s.variant}) — per layer, from gptq_progress_*.json</h3>`;
  const cur = new Set();
  for (const r of running) for (const ip of (s.inner[r] || []))
    if (ip.label.endsWith(' batch')) { const m = ip.note.match(/\\[(.*)\\]/);
      if (m) m[1].split(',').forEach(x => cur.add(ip.label.split(' ')[0] + ':' + x.trim())); }
  for (const [comp, st] of Object.entries(s.calibration)) {
    for (const [pass, done] of Object.entries(st.passes)) {
      const doneSet = new Set(done);
      let cells = '';
      for (let i = 0; i < st.total; i++) {
        const isCur = !doneSet.has(i) && cur.has(pass + ':' + i);
        cells += `<div class="cell ${doneSet.has(i) ? 'done ' + pass : ''}${isCur ? ' current' : ''}" title="${pass} layer ${i}"></div>`;
      }
      html += `<div class="inner">${PASS_LABEL[pass] || pass}: ${doneSet.size}/${st.total}</div><div class="grid">${cells}</div>`;
    }
  }
  return html;
}

function render(s) {
  WORK = s.work;
  const now = Date.now() / 1000;
  document.getElementById('sub').textContent = s.last_poll
    ? `last poll ${Math.round(now - s.last_poll)}s ago` + (s.last_poll_ok ? '' : ' — poll failed, showing last known state')
    : 'connecting...';
  const order = ['mlx', 'gguf', 'mtp'].filter(p => s.pipelines[p]).concat(
    Object.keys(s.pipelines).filter(p => !['mlx', 'gguf', 'mtp'].includes(p)));
  let html = '';
  if (!order.length) html = '<p class="empty">No pipeline has run yet (no logs/pipeline.log).</p>';
  for (const p of order) {
    const info = s.pipelines[p];
    const lastTs = Math.max(...Object.values(info.steps).map(x => x.ts), info.started || 0);
    const stale = info.status === 'running' && now - lastTs > STALE_S && !(s.running || []).some(r => r.startsWith(p + '_'));
    const badge = stale ? 'stale' : info.status;
    html += `<h2>${TITLES[p] || p}<span class="badge ${badge}">${stale ? 'no activity ' + fmtDur(now - lastTs) : info.status}</span></h2>`;
    html += `<div class="sub">${esc((info.detail || '').replace(/^\\w+ /, ''))}</div>`;
    html += '<table class="steps">';
    for (const step of info.order) {
      const st = info.steps[step];
      const dur = st.status === 'START' ? now - st.started : (st.started ? st.ts - st.started : null);
      const inner = (s.inner[p + '_' + step] || []).map(ip =>
        `<div class="inner">${esc(ip.label)} ${ip.n}/${ip.of} ${esc(ip.note)}</div>` +
        `<div class="bar"><div style="width:${Math.min(100, 100 * ip.n / ip.of)}%"></div></div>`).join('');
      html += `<tr><td class="icon ${st.status}">${ICON[st.status] || '?'}</td><td class="step">${esc(step)}${inner}</td>` +
              `<td class="detail">${esc(st.detail || '')}</td><td class="dur">${fmtDur(dur)}</td></tr>`;
    }
    html += '</table>';
    if (p === 'mlx') html += calibrationHtml(s, s.running || []);
    const tailKey = (s.running || []).find(r => r.startsWith(p + '_')) ||
      Object.keys(s.tails || {}).find(k => k.startsWith(p + '_') && info.steps[k.slice(p.length + 1)]?.status === 'FAIL');
    if (tailKey && s.tails[tailKey]) html += `<h3>${esc(tailKey)}.log</h3><pre>${esc(s.tails[tailKey])}</pre>`;
  }
  document.getElementById('pipes').innerHTML = html;
  const meta = [];
  if (s.gpu) meta.push('GPU: ' + s.gpu);
  if (s.disk) meta.push('disk: ' + s.disk.split(/\\s+/).slice(1, 5).join(' '));
  if (s.error) meta.push(s.error);
  document.getElementById('meta').textContent = meta.join('  |  ');
}
async function tick() {
  try { render(await (await fetch('/status.json')).json()); } catch (e) {}
}
tick(); setInterval(tick, 3000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/status.json":
            with STATE_LOCK:
                body = json.dumps(STATE).encode()
            ctype = "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="read --work on this machine instead of over SSH")
    ap.add_argument("--work", default="/workspace", help="the pipelines' $WORK (logs live in $WORK/logs)")
    ap.add_argument("--pod-host")
    ap.add_argument("--pod-port", default="22")
    ap.add_argument("--pod-ssh-key", default=str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"))
    ap.add_argument("--poll-interval", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--once", action="store_true", help="poll once, print the state as JSON, exit (for testing)")
    args = ap.parse_args()
    if not args.local and not args.pod_host:
        ap.error("--pod-host is required unless --local")
    if args.once:
        poll(args)
        print(json.dumps(STATE, indent=2, default=str))
        return
    threading.Thread(target=poll_loop, args=(args,), daemon=True).start()
    print(f"dashboard: http://localhost:{args.port}  (polling {'local ' + args.work if args.local else args.pod_host})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
