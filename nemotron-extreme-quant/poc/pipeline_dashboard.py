"""Local live dashboard for the GPTQ pipeline running on the RunPod pod --
polls the pod over SSH every --poll-interval seconds (one combined SSH call
per poll: log tail, sensitivity dry-run log, ppl_summary.json, GPU/disk
stats) and serves an auto-refreshing HTML page showing block-by-block
progress, current stage, and perplexity results against baselines, so you
don't have to ask "how's it going" every few minutes.

Stdlib only (http.server + threading + subprocess) -- nothing to pip
install.

Usage:
    python poc/pipeline_dashboard.py \
        --pod-host 154.54.102.33 --pod-port 19798 \
        --pod-ssh-key ~/.runpod/ssh/runpodctl-ssh-key
    # then open http://localhost:8420
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_LOCK = threading.Lock()
STATE = {
    "run_name": None,
    "stage": "unknown",
    "stage_label": "waiting for first poll...",
    "block_idx": None,
    "block_total": None,
    "blocks": [],  # list of {idx, kind, upgraded, elapsed}
    "calib_progress": None,
    "gpu": None,
    "disk": None,
    "ppl_summary": None,
    "sensitivity": None,
    "axolotl": None,
    "log_age_s": None,
    "raw_tail": "",
    "error": None,
    "last_poll": None,
    "last_poll_ok": False,
}

BLOCK_RE = re.compile(r"\[block (\d+)/(\d+)\] (\w+):(.*?), .*?total_elapsed=(\d+)s")
CALIB_RE = re.compile(r"calibration pass (\d+)/(\d+) done")
AXOLOTL_STEP_RE = re.compile(r"(\d+)/(\d+) \[")
AXOLOTL_METRICS_RE = re.compile(r"\{'(?:loss|eval_loss)':.*?\}")


def parse_axolotl_log(text: str) -> dict | None:
    """axolotl/transformers' Trainer prints tqdm step bars (`N/M [...]`)
    interleaved with Python-dict-repr metrics lines (`{'loss': ..., 'ppl':
    ...}` for train steps, `{'eval_loss': ..., 'eval_ppl': ...}` after each
    eval pass) -- a completely different log shape from the GPTQ pipeline's
    own [block N/M] lines above, so this is a separate small parser rather
    than trying to force it through parse_log.
    """
    if not text:
        return None
    out = {"step": None, "total_steps": None, "train": None, "eval": None, "stage": "unknown"}
    steps = AXOLOTL_STEP_RE.findall(text)
    if steps:
        step, total = steps[-1]
        out["step"], out["total_steps"] = int(step), int(total)
    for m in AXOLOTL_METRICS_RE.finditer(text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "eval_loss" in d:
            out["eval"] = d
        else:
            out["train"] = d
    if "Traceback" in text:
        out["stage"] = "error"
    elif out["step"] is not None and out["total_steps"] and out["step"] >= out["total_steps"]:
        out["stage"] = "done"
    elif out["train"] or out["eval"]:
        out["stage"] = "training"
    return out


def parse_log(text: str) -> dict:
    out = {
        "stage": "unknown", "stage_label": "no log yet", "block_idx": None,
        "block_total": None, "blocks": [], "calib_progress": None,
    }
    if not text:
        return out

    if "Loading" in text:
        out["stage"], out["stage_label"] = "loading", "Loading model..."
    calib_matches = CALIB_RE.findall(text)
    if calib_matches:
        n, m = calib_matches[-1]
        out["calib_progress"] = [int(n), int(m)]
        out["stage"], out["stage_label"] = "calibrating", f"Calibration forward pass {n}/{m}"
    if "Calibration capture done" in text or "Sequential mode: capturing" in text:
        out["stage"], out["stage_label"] = "quantizing", "Quantizing blocks..."

    blocks = []
    for m in BLOCK_RE.finditer(text):
        idx, total, kind, detail, elapsed = m.groups()
        upgraded = "6b" in detail or "down_bits=6" in detail or "up_bits=6" in detail
        blocks.append({
            "idx": int(idx), "total": int(total), "kind": kind,
            "upgraded": upgraded, "elapsed": int(elapsed), "detail": detail.strip(),
        })
    if blocks:
        blocks.sort(key=lambda b: b["idx"])
        out["blocks"] = blocks
        out["block_idx"] = blocks[-1]["idx"]
        out["block_total"] = blocks[-1]["total"]
        out["stage"], out["stage_label"] = "quantizing", f"Quantizing block {blocks[-1]['idx']}/{blocks[-1]['total']}"

    if "GPTQ_STAGE_DONE" in text:
        out["stage"], out["stage_label"] = "gptq_done", "GPTQ stage done, running sanity check..."
    if "SANITY_CHECK_PASSED" in text:
        out["stage"], out["stage_label"] = "sanity_passed", "Sanity check passed, converting to MLX..."
    if "SANITY_CHECK_FAILED" in text:
        out["stage"], out["stage_label"] = "sanity_failed", "SANITY CHECK FAILED -- inspect log"
    if "[INFO] Quantizing" in text and "MLX_CONVERT_DONE" not in text:
        out["stage"], out["stage_label"] = "converting", "mlx_lm.convert quantizing..."
    if "MLX_CONVERT_DONE" in text:
        out["stage"], out["stage_label"] = "done", "MLX conversion done!"
    if "Traceback" in text or re.search(r"^Error", text, re.MULTILINE):
        out["stage"], out["stage_label"] = "error", "Error detected -- check raw log"
    if "--sensitivity-dry-run: stopping here" in text:
        out["stage"], out["stage_label"] = "dry_run_done", "Sensitivity dry-run complete (no quantization performed)"

    return out


def poll_pod(pod_host: str, pod_port: str, pod_ssh_key: str, log_glob: str) -> None:
    remote_cmd = f"""
LOG=$(ls -t {log_glob} 2>/dev/null | head -1)
echo "===RUN_NAME==="; basename "$LOG" 2>/dev/null
echo "===LOG_MTIME==="; [ -n "$LOG" ] && stat -c %Y "$LOG" 2>/dev/null
echo "===LOG_TAIL==="; [ -n "$LOG" ] && tail -c 24000 "$LOG"
echo "===DRYRUN==="; [ -f /root/sensitivity_dryrun.log ] && tail -c 20000 /root/sensitivity_dryrun.log
echo "===PPL_SUMMARY==="; [ -f /root/ppl_summary.json ] && cat /root/ppl_summary.json
echo "===AXOLOTL==="; [ -f /root/axolotl_train.log ] && tail -c 12000 /root/axolotl_train.log
echo "===GPU==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "===DISK==="; df -h / | tail -1
"""
    ssh_cmd = [
        "ssh", "-i", pod_ssh_key, "-p", str(pod_port),
        "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
        f"root@{pod_host}", remote_cmd,
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=20)
        raw = result.stdout
    except Exception as e:
        with STATE_LOCK:
            STATE["last_poll"] = time.time()
            STATE["last_poll_ok"] = False
            STATE["error"] = f"SSH poll failed: {e!r}"
        return

    sections = re.split(r"===(\w+)===\n?", raw)
    parts = {}
    for i in range(1, len(sections), 2):
        parts[sections[i]] = sections[i + 1] if i + 1 < len(sections) else ""

    log_tail = parts.get("LOG_TAIL", "")
    parsed = parse_log(log_tail)

    log_age_s = None
    mtime_str = parts.get("LOG_MTIME", "").strip()
    if mtime_str.isdigit():
        log_age_s = time.time() - int(mtime_str)

    ppl_summary = None
    if parts.get("PPL_SUMMARY", "").strip():
        try:
            ppl_summary = json.loads(parts["PPL_SUMMARY"])
        except json.JSONDecodeError:
            pass

    sensitivity = None
    dryrun_text = parts.get("DRYRUN", "")
    if dryrun_text.strip():
        rows = []
        for line in dryrun_text.splitlines():
            m = re.match(r"\s*\[(UPGRADE| *)\] block\s*(\d+) (\S+)\s+score=([\d.eE+-]+)", line)
            if m:
                marker, idx, proj, score = m.groups()
                rows.append({"idx": int(idx), "proj": proj, "score": float(score), "upgraded": marker.strip() == "UPGRADE"})
        if rows:
            sensitivity = rows

    with STATE_LOCK:
        STATE["run_name"] = parts.get("RUN_NAME", "").strip() or None
        STATE.update(parsed)
        STATE["gpu"] = parts.get("GPU", "").strip() or None
        STATE["disk"] = parts.get("DISK", "").strip() or None
        STATE["ppl_summary"] = ppl_summary if ppl_summary is not None else STATE["ppl_summary"]
        STATE["sensitivity"] = sensitivity if sensitivity is not None else STATE["sensitivity"]
        STATE["axolotl"] = parse_axolotl_log(parts.get("AXOLOTL", ""))
        STATE["log_age_s"] = log_age_s
        STATE["raw_tail"] = log_tail[-4000:]
        STATE["last_poll"] = time.time()
        STATE["last_poll_ok"] = True
        if parsed["stage"] != "error":
            STATE["error"] = None


def poll_loop(pod_host, pod_port, pod_ssh_key, log_glob, interval):
    while True:
        poll_pod(pod_host, pod_port, pod_ssh_key, log_glob)
        time.sleep(interval)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>GPTQ Pipeline Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;
         background: #0f1115; color: #e6e6e6; }
  @media (prefers-color-scheme: light) { body { background: #f7f7f8; color: #1a1a1a; } }
  h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
  .sub { opacity: 0.6; font-size: 0.85rem; margin-bottom: 1.5rem; }
  .stage { font-size: 1.1rem; font-weight: 600; padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 1rem;
           background: #1e2530; }
  @media (prefers-color-scheme: light) { .stage { background: #e8eaf0; } }
  .stage.done { background: #16351f; } @media (prefers-color-scheme: light) { .stage.done { background: #d9f2df; } }
  .stage.error { background: #3a1618; } @media (prefers-color-scheme: light) { .stage.error { background: #f8d7da; } }
  .grid { display: flex; flex-wrap: wrap; gap: 3px; margin: 1rem 0; }
  .cell { width: 16px; height: 16px; border-radius: 3px; opacity: 0.35; }
  .cell.done { opacity: 1; }
  .cell.current { outline: 2px solid #fff; opacity: 1; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.5); } 50% { box-shadow: 0 0 0 4px rgba(255,255,255,0); } }
  .mamba { background: #4a7fd6; } .moe { background: #4ad691; } .attention { background: #d64ac2; }
  .cell.upgraded { border: 2px solid gold; }
  .legend { font-size: 0.8rem; opacity: 0.7; margin-bottom: 1.5rem; }
  .legend span { margin-right: 1rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
  td, th { padding: 0.35rem 0.6rem; text-align: left; border-bottom: 1px solid rgba(128,128,128,0.25); }
  .ours { font-weight: 700; }
  .meta { font-size: 0.85rem; opacity: 0.65; margin: 1rem 0; }
  pre { background: #1a1d24; color: #b8c4d0; padding: 0.8rem; border-radius: 8px; font-size: 0.72rem;
        overflow-x: auto; max-height: 260px; overflow-y: auto; }
  @media (prefers-color-scheme: light) { pre { background: #1a1d24; color: #d3dbe4; } }
  .stale { color: #e0a030; }
</style></head>
<body>
  <h1>GPTQ Pipeline Dashboard</h1>
  <div class="sub" id="run-name">connecting...</div>
  <div class="stage" id="stage">--</div>
  <div class="grid" id="grid"></div>
  <div class="legend">
    <span><b style="color:#4a7fd6">■</b> mamba</span>
    <span><b style="color:#d64ac2">■</b> attention</span>
    <span><b style="color:#4ad691">■</b> moe</span>
    <span><b style="border:2px solid gold; padding:0 3px">■</b> upgraded (high-bits)</span>
  </div>
  <div id="ppl-section"></div>
  <div id="sensitivity-section"></div>
  <div id="axolotl-section"></div>
  <div class="meta" id="meta"></div>
  <pre id="raw-tail"></pre>

<script>
function render(s) {
  const STALE_THRESHOLD_S = 300;  // GPTQ pipeline logs append at least every few min while actually running
  const logStale = s.log_age_s !== null && s.log_age_s > STALE_THRESHOLD_S;
  document.getElementById('run-name').textContent = (s.run_name || 'no run detected') +
    (s.last_poll ? ' -- last poll ' + Math.round((Date.now()/1000 - s.last_poll)) + 's ago' : '') +
    (logStale ? ` -- log last written ${Math.round(s.log_age_s/60)}min ago, no active pipeline run` : '');
  const stageEl = document.getElementById('stage');
  stageEl.textContent = (logStale ? '[stale] ' : '') + (s.stage_label || '--');
  // A stale error/sanity_failed is old history from a run that already
  // ended, not a live problem -- don't paint it alarming red, that's what
  // confused a fresh "is the pipeline currently broken?" glance.
  const isLiveError = (s.stage === 'error' || s.stage === 'sanity_failed') && !logStale;
  stageEl.className = 'stage' + (s.stage === 'done' && !logStale ? ' done' : (isLiveError ? ' error' : ''));

  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  if (s.blocks && s.blocks.length) {
    const total = s.block_total || s.blocks.length;
    const byIdx = {};
    s.blocks.forEach(b => byIdx[b.idx] = b);
    for (let i = 0; i < total; i++) {
      const b = byIdx[i];
      const div = document.createElement('div');
      div.className = 'cell' + (b ? ' done ' + b.kind + (b.upgraded ? ' upgraded' : '') : '');
      if (i === s.block_idx) div.className += ' current';
      div.title = b ? `block ${b.idx}: ${b.kind} (${b.detail})` : `block ${i}: pending`;
      grid.appendChild(div);
    }
  }

  const pplDiv = document.getElementById('ppl-section');
  if (s.ppl_summary && s.ppl_summary.models) {
    let rows = s.ppl_summary.models.slice().sort((a,b) => a.ppl - b.ppl)
      .map(m => `<tr class="${m.name && m.name.startsWith('Our') ? 'ours' : ''}"><td>${m.name}</td><td>${m.ppl.toFixed(2)}</td><td>${m.note||''}</td></tr>`).join('');
    pplDiv.innerHTML = `<h3>Perplexity (wikitext-2, lower is better)</h3><table><tr><th>model</th><th>PPL</th><th>note</th></tr>${rows}</table>`;
  } else { pplDiv.innerHTML = ''; }

  const sensDiv = document.getElementById('sensitivity-section');
  if (s.sensitivity && s.sensitivity.length) {
    const rows = s.sensitivity.slice(0, 25).map(r =>
      `<tr class="${r.upgraded ? 'ours' : ''}"><td>${r.upgraded ? 'UPGRADE' : ''}</td><td>block ${r.idx}</td><td>${r.proj}</td><td>${r.score.toExponential(3)}</td></tr>`
    ).join('');
    sensDiv.innerHTML = `<h3>Sensitivity scores (top 25 of ${s.sensitivity.length})</h3><table><tr><th></th><th>block</th><th>proj</th><th>score</th></tr>${rows}</table>`;
  } else { sensDiv.innerHTML = ''; }

  const axoDiv = document.getElementById('axolotl-section');
  if (s.axolotl && (s.axolotl.step !== null || s.axolotl.train || s.axolotl.eval)) {
    const a = s.axolotl;
    const pct = (a.step !== null && a.total_steps) ? Math.round(100 * a.step / a.total_steps) : null;
    let rows = '';
    if (a.train) rows += `<tr><td>train</td><td>loss=${a.train.loss}</td><td>ppl=${a.train.ppl||''}</td><td>epoch=${a.train.epoch||''}</td></tr>`;
    if (a.eval) rows += `<tr><td>eval</td><td>loss=${a.eval.eval_loss}</td><td>ppl=${a.eval.eval_ppl||''}</td><td>epoch=${a.eval.epoch||''}</td></tr>`;
    axoDiv.innerHTML = `<h3>ipsupport-code LoRA training${pct !== null ? ` -- step ${a.step}/${a.total_steps} (${pct}%)` : ''}${a.stage === 'done' ? ' -- DONE' : ''}${a.stage === 'error' ? ' -- ERROR' : ''}</h3><table>${rows}</table>`;
  } else { axoDiv.innerHTML = ''; }

  let meta = [];
  if (s.calib_progress) meta.push(`calibration: ${s.calib_progress[0]}/${s.calib_progress[1]}`);
  if (s.gpu) meta.push('GPU: ' + s.gpu);
  if (s.disk) meta.push('disk: ' + s.disk.split(/\\s+/).slice(1,5).join(' '));
  document.getElementById('meta').innerHTML = meta.join(' &nbsp;|&nbsp; ') +
    (s.last_poll_ok ? '' : ' <span class="stale">(poll failed, showing last known state)</span>');

  document.getElementById('raw-tail').textContent = s.raw_tail || '';
}

async function tick() {
  try {
    const r = await fetch('/status.json');
    render(await r.json());
  } catch (e) { /* keep last render */ }
}
tick();
setInterval(tick, 3000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- the dashboard's own poll log is more useful than per-request access logs

    def do_GET(self):
        if self.path == "/status.json":
            with STATE_LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-host", required=True)
    parser.add_argument("--pod-port", required=True)
    parser.add_argument("--pod-ssh-key", required=True)
    parser.add_argument("--log-glob", default="/root/pipeline-*.log")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    t = threading.Thread(
        target=poll_loop,
        args=(args.pod_host, args.pod_port, args.pod_ssh_key, args.log_glob, args.poll_interval),
        daemon=True,
    )
    t.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
