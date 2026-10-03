"""Live dashboard for the ternary distillation pipeline.

Polls the training host over SSH (one combined shell call every
--poll-interval seconds) and serves http://localhost:8422:

  - pipeline steps (master, data, teacher, train) from logs/pipeline.log;
  - data and teacher shard progress, teacher tokens/s;
  - training: step, tokens, tokens/s, ETA, phase, last checkpoint and its age;
  - curves: eval KL and top-1 against the teacher vs. training tokens, with
    the reference evals (QAT q4_0 grid, MLX 4-bit, 2-bit RTN) as dashed
    lines; training loss (EMA);
  - GPU utilization / power / temperature, memory, disk, the running step's
    log tail.

Stdlib only. The page keeps working on stale data if a poll fails.

    python ternary_dashboard.py --host dgx --work '~/ternary'
    # open http://localhost:8422
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE: dict = {"ok": False, "error": "no poll yet", "polled": 0}
LOCK = threading.Lock()
SEP = "\n@@@@ "

REMOTE = r"""
W={work}
s(){{ printf '\n@@@@ %s\n' "$1"; }}
s pipeline; grep ' ternary ' $W/logs/pipeline.log 2>/dev/null | tail -n 300
s data; cat $W/data/progress.json 2>/dev/null
s teacher; cat $W/teacher/progress.json 2>/dev/null
s status; cat $W/run/status.json 2>/dev/null
s metrics; cat $W/run/metrics.jsonl 2>/dev/null
s ckpt; for d in $W/run/ckpt/step_*; do [ -f "$d/DONE" ] && echo "$(basename $d) $(stat -c %Y $d)"; done 2>/dev/null
s gpu; nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null
s mem; free -b | awk '/Mem:/{{print $2, $3, $7}}'
s disk; df -B1 $W | awk 'NR==2{{print $2, $3, $4}}'
s now; date +%s
s log; f=$(ls -t $W/logs/ternary_*.log 2>/dev/null | head -1); echo "$f"; tail -c 20000 "$f" 2>/dev/null | tr '\r' '\n' | grep -v -i -E 'warn|^\s*$|Loading weights' | tail -n 25
"""


def poll(host: str, work: str) -> dict:
    cmd = REMOTE.format(work=work)
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, "bash -s"],
                       input=cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 and not r.stdout:
        raise RuntimeError(r.stderr.strip()[-300:] or f"ssh exit {r.returncode}")
    sec = {}
    for chunk in r.stdout.split(SEP)[1:]:
        name, _, body = chunk.partition("\n")
        sec[name.strip()] = body.rstrip("\n")

    def js(name):
        try:
            return json.loads(sec.get(name) or "null")
        except json.JSONDecodeError:
            return None

    steps: dict[str, dict] = {}
    for line in sec.get("pipeline", "").splitlines():
        p = line.split(None, 5)
        if len(p) < 5 or p[0] != "PIPE":
            continue
        ts, status, step = int(p[1]), p[3], p[4]
        if step == "_pipeline":
            continue
        st = steps.setdefault(step, {})
        if status == "START":
            st.clear(); st.update(state="running", start=ts)
        else:
            st.update(state={"DONE": "done", "FAIL": "failed", "SKIP": "skipped"}.get(status, status), end=ts)

    train, evals, refs = [], [], {}
    for line in sec.get("metrics", "").splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "ref" in m:
            refs[m["ref"]] = m
        elif m.get("eval"):
            evals.append(m)
        else:
            train.append(m)
    # downsample the per-step loss to <= 600 points, EMA-smoothed
    ema, pts = None, []
    for m in train:
        ema = m["loss"] if ema is None else 0.95 * ema + 0.05 * m["loss"]
        pts.append([m["tokens"], ema])
    stride = max(1, len(pts) // 600)
    loss = pts[::stride] + (pts[-1:] if pts and len(pts) % stride else [])

    gpu = (sec.get("gpu") or "").split(",")
    mem = (sec.get("mem") or "0 0 0").split()
    disk = (sec.get("disk") or "0 0 0").split()
    ck = [l.split() for l in sec.get("ckpt", "").splitlines() if l.strip()]
    log_lines = sec.get("log", "").splitlines()
    return {
        "ok": True, "polled": time.time(), "now": int(sec.get("now") or time.time()),
        "steps": steps, "data": js("data"), "teacher": js("teacher"), "status": js("status"),
        "evals": evals, "refs": refs, "loss": loss, "last_train": train[-1] if train else None,
        "ckpt": [{"name": c[0], "time": int(c[1])} for c in ck if len(c) == 2],
        "gpu": {"util": gpu[0].strip(), "power": gpu[1].strip(), "temp": gpu[2].strip()} if len(gpu) >= 3 else None,
        "mem": {"total": int(mem[0]), "used": int(mem[1]), "avail": int(mem[2])},
        "disk": {"total": int(disk[0]), "used": int(disk[1]), "free": int(disk[2])},
        "log_file": log_lines[0] if log_lines else "", "log": log_lines[1:],
    }


def poller(host, work, interval):
    while True:
        try:
            st = poll(host, work)
            with LOCK:
                STATE.clear(); STATE.update(st)
        except Exception as e:  # keep serving the last good state
            with LOCK:
                STATE["ok"] = False
                STATE["error"] = str(e)
        time.sleep(interval)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ternary Pilot</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a;--yellow:#eda100;--red:#d23c3c;--gray:#9a9993}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121212;--card:#1c1c1c;--ink:#f1f0ec;--ink2:#a9a8a3;--rule:#2e2e2c;--blue:#5a9cf0;--orange:#f08a5c;--aqua:#3cc995;--yellow:#f2b632;--gray:#6f6e69}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}
.k{color:var(--ink2);font-size:12px}.v{font-size:22px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}.s{color:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}
.bar{height:6px;background:var(--rule);border-radius:3px;margin-top:8px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--blue)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:10px;margin-bottom:14px}
.ct{font-weight:600;margin-bottom:4px}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--ink2);margin-top:4px}
.legend b{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}
.steps{display:flex;flex-wrap:wrap;gap:8px}.step{padding:6px 10px;border-radius:8px;border:1px solid var(--rule);font-size:13px}
.done{color:var(--aqua)}.running{color:var(--blue)}.failed{color:var(--red)}.skipped{color:var(--gray)}
pre{margin:0;white-space:pre-wrap;word-break:break-all;font:12px/1.4 ui-monospace,Menlo,monospace;color:var(--ink2);max-height:340px;overflow:auto}
.err{color:var(--red);font-size:13px;margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule)}th{color:var(--ink2);font-weight:500}
</style></head><body><div class="wrap">
<h1>Gemma 4 12B → ternary · pilot</h1><div class="sub" id="sub">connecting…</div>
<div class="err" id="err"></div>
<div class="grid" id="tiles"></div>
<div class="card" style="margin-bottom:14px"><div class="ct">Pipeline</div><div class="steps" id="steps"></div></div>
<div class="charts">
 <div class="card"><div class="ct">Eval KL to the teacher (log scale, lower is better)</div><div id="c_kl"></div><div class="legend" id="l_kl"></div></div>
 <div class="card"><div class="ct">Eval top-1 agreement with the teacher</div><div id="c_top"></div><div class="legend" id="l_top"></div></div>
 <div class="card"><div class="ct">Training loss (KL over top-k, EMA)</div><div id="c_loss"></div></div>
 <div class="card"><div class="ct">Evaluations</div><div id="tbl"></div></div>
</div>
<div class="card"><div class="ct" id="logt">Log</div><pre id="log"></pre></div>
</div>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const REFC={"q4_0":"--aqua","rtn4g64":"--yellow","rtn2g128":"--orange"};
const REFN={"q4_0":"q4_0 (QAT grid, 4.5 b)","rtn4g64":"MLX 4-bit RTN","rtn2g128":"2-bit RTN, same size"};
const fmtT=s=>{if(s==null||!isFinite(s))return"—";s=Math.max(0,s);const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?`${d}d ${h}h`:h?`${h}h ${m}m`:`${m}m`};
const fmtN=n=>n>=1e9?(n/1e9).toFixed(2)+"B":n>=1e6?(n/1e6).toFixed(1)+"M":n>=1e3?(n/1e3).toFixed(0)+"k":String(Math.round(n));
const GB=b=>(b/1e9).toFixed(0)+" GB";
function tile(k,v,s,p){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s||""}</div>${p!=null?`<div class="bar"><i style="width:${Math.min(100,p*100).toFixed(1)}%"></i></div>`:""}</div>`}
function chart(el,series,refs,opt){
 const W=560,H=230,L=52,R=12,T=10,B=30;
 let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1])}));refs.forEach(r=>ys.push(r.y));
 if(!ys.length){el.innerHTML=`<div class="s" style="padding:40px 0;text-align:center">no data yet</div>`;return}
 let x0=0,x1=Math.max(1,...xs);let y0=Math.min(...ys),y1=Math.max(...ys);
 const lg=opt.log;const f=v=>lg?Math.log10(Math.max(v,1e-6)):v;
 let a=f(y0),b=f(y1);if(a===b){a-=lg?0.5:0.05;b+=lg?0.5:0.05}const pad=(b-a)*0.08;a-=pad;b+=pad;
 if(opt.max!=null&&!lg)b=Math.min(b,opt.max+pad);
 const X=v=>L+(v-x0)/(x1-x0||1)*(W-L-R),Y=v=>T+(b-f(v))/(b-a)*(H-T-B);
 let g=`<svg viewBox="0 0 ${W} ${H}">`;
 const ticks=[];if(lg){for(let e=Math.floor(a);e<=Math.ceil(b);e++)[1,2,5].forEach(m=>{const v=m*10**e;if(f(v)>=a&&f(v)<=b)ticks.push(v)})}else{for(let i=0;i<=4;i++)ticks.push(10**(a+(b-a)*i/4)>0&&false?0:a+(b-a)*i/4)}
 ticks.forEach(v=>{const y=Y(v);g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${css('--rule')}"/><text x="${L-6}" y="${y+4}" text-anchor="end" font-size="11" fill="${css('--ink2')}">${opt.fmt(v)}</text>`});
 for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;g+=`<text x="${X(v)}" y="${H-10}" text-anchor="middle" font-size="11" fill="${css('--ink2')}">${fmtN(v)}</text>`}
 refs.forEach(r=>{const y=Y(r.y);g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${css(r.c)}" stroke-width="2" stroke-dasharray="6 4"><title>${r.n}: ${opt.fmt(r.y)}</title></line>`});
 series.forEach(s=>{if(!s.pts.length)return;const d=s.pts.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join("");
  g+=`<path d="${d}" fill="none" stroke="${css(s.c)}" stroke-width="2"/>`;
  if(s.dots)s.pts.forEach(p=>g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${css(s.c)}" stroke="${css('--card')}" stroke-width="2"><title>${fmtN(p[0])} tokens: ${opt.fmt(p[1])}</title></circle>`)});
 el.innerHTML=g+`<text x="${(L+W-R)/2}" y="${H}" text-anchor="middle" font-size="11" fill="${css('--ink2')}"></text></svg>`;
}
function legend(el,items){el.innerHTML=items.map(i=>`<span><b style="background:${css(i.c)}"></b>${i.n}</span>`).join("")}
async function tick(){
 let s;try{s=await (await fetch("/state")).json()}catch(e){document.getElementById("err").textContent="dashboard server unreachable";return}
 document.getElementById("err").textContent=s.ok?"":"poll failed: "+(s.error||"")+" (showing last data)";
 if(!s.now)return;
 const st=s.status||{},lt=s.last_train;const now=s.now;
 document.getElementById("sub").textContent=`host time ${new Date(now*1000).toLocaleString()} · polled ${Math.round(Date.now()/1000-s.polled)}s ago`;
 const tokens=lt?lt.tokens:0,target=st.tokens_target||0;
 const lastck=s.ckpt.length?s.ckpt[s.ckpt.length-1]:null;
 const lastEval=s.evals.length?s.evals[s.evals.length-1]:null;
 const ph=st.phase||(s.steps.train?s.steps.train.state:"—");
 let t=tile("Phase",ph,st.pid?`pid ${st.pid}`:"");
 t+=tile("Training tokens",fmtN(tokens),`of ${fmtN(target)} · step ${st.step||0}/${st.total_steps||"—"}`,target?tokens/target:null);
 t+=tile("Speed",st.tok_s?Math.round(st.tok_s)+" tok/s":"—",st.eta_s!=null?"ETA "+fmtT(st.eta_s):"");
 t+=tile("Last checkpoint",lastck?lastck.name.replace("step_","step "):"none",lastck?fmtT(now-lastck.time)+" ago":"");
 t+=tile("Eval KL",lastEval?lastEval.kl.toFixed(4):"—",lastEval?`top-1 ${(lastEval.top1*100).toFixed(1)}% · ppl ${lastEval.ppl.toFixed(2)}`:"");
 if(s.data)t+=tile("Data shards",`${s.data.done}/${s.data.total}`,"",s.data.done/s.data.total);
 if(s.teacher)t+=tile("Teacher shards",`${s.teacher.done}/${s.teacher.total}`,s.teacher.tok_s?Math.round(s.teacher.tok_s)+" tok/s":"",s.teacher.done/s.teacher.total);
 if(s.gpu)t+=tile("GPU",s.gpu.util+"%",`${s.gpu.power} W · ${s.gpu.temp}°C`,s.gpu.util/100);
 t+=tile("Memory",GB(s.mem.used),`of ${GB(s.mem.total)} · ${GB(s.mem.avail)} available`,s.mem.used/s.mem.total);
 t+=tile("Disk free",GB(s.disk.free),`of ${GB(s.disk.total)}`,s.disk.used/s.disk.total);
 document.getElementById("tiles").innerHTML=t;
 const order=["master","data","teacher","train"];
 document.getElementById("steps").innerHTML=order.map(k=>{const x=s.steps[k]||{};const st_=x.state||"pending";const d=x.start?fmtT((x.end||now)-x.start):"";return `<div class="step ${st_}">${k} · ${st_}${d?" · "+d:""}</div>`}).join("");
 const refs=Object.entries(s.refs);
 const ser=[{n:"ternary (ours)",c:"--blue",dots:true}];
 chart(document.getElementById("c_kl"),[{...ser[0],pts:s.evals.map(e=>[e.tokens,e.kl])}],refs.map(([k,r])=>({n:REFN[k]||k,c:REFC[k]||"--gray",y:r.kl})),{log:true,fmt:v=>v>=1?v.toFixed(1):v>=0.1?v.toFixed(2):v.toFixed(3)});
 chart(document.getElementById("c_top"),[{...ser[0],pts:s.evals.map(e=>[e.tokens,e.top1])}],refs.map(([k,r])=>({n:REFN[k]||k,c:REFC[k]||"--gray",y:r.top1})),{fmt:v=>(v*100).toFixed(0)+"%"});
 const leg=[{n:"ternary (ours)",c:"--blue"}].concat(refs.map(([k])=>({n:REFN[k]||k,c:REFC[k]||"--gray"})));
 legend(document.getElementById("l_kl"),leg);legend(document.getElementById("l_top"),leg);
 chart(document.getElementById("c_loss"),[{n:"loss",c:"--blue",pts:s.loss}],[],{log:true,fmt:v=>v>=1?v.toFixed(1):v.toFixed(2)});
 let rows=refs.map(([k,r])=>`<tr><td>${REFN[k]||k}</td><td>ref</td><td>${r.kl.toFixed(4)}</td><td>${(r.top1*100).toFixed(1)}%</td><td>${r.ppl.toFixed(2)}</td></tr>`).join("");
 rows+=s.evals.slice(-8).map(e=>`<tr><td>ternary</td><td>${fmtN(e.tokens)}</td><td>${e.kl.toFixed(4)}</td><td>${(e.top1*100).toFixed(1)}%</td><td>${e.ppl.toFixed(2)}</td></tr>`).join("");
 document.getElementById("tbl").innerHTML=`<table><tr><th>Model</th><th>Tokens</th><th>KL</th><th>Top-1</th><th>PPL</th></tr>${rows}</table>`;
 document.getElementById("logt").textContent="Log · "+(s.log_file||"");
 document.getElementById("log").textContent=s.log.join("\n");
}
tick();setInterval(tick,10000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            ctype = "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="dgx", help="ssh host running the pipeline")
    ap.add_argument("--work", default="~/ternary", help="WORK dir on that host")
    ap.add_argument("--port", type=int, default=8422)
    ap.add_argument("--poll-interval", type=float, default=15)
    args = ap.parse_args()
    threading.Thread(target=poller, args=(args.host, args.work, args.poll_interval), daemon=True).start()
    print(f"http://localhost:{args.port}  (polling {args.host}:{args.work} every {args.poll_interval:.0f}s)")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
