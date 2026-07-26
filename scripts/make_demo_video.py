#!/usr/bin/env python3
"""
CogniMem Devpost Demo Video — Full Product Demo
10 slides: UI walkthrough + features + philosophy + architecture
"""

import base64, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "screenshots" / "ui"
ALL = ROOT / "screenshots"
W, H = 1280, 720
FPS = 30
DUR = 6  # seconds per slide

def img(p):
    p = Path(p)
    return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" if p.exists() else ""

S = type("S", (), {})()  # namespace
for f in ["CM-Chat-Welcome","CM-Chat-Response","CM-Chat-Search","CM-Create-Project",
          "CM-Create-Project-Filled","CM-Agent-Created","CM-Dashboard-Data",
          "CM-Dashboard-Data2","CM-Graph-Page","CM-Dashboard-Home","CM-Dashboard-Memory"]:
    p = UI / f"{f}.png"
    if p.exists(): setattr(S, f.replace("-","_"), str(p))
for f in ["CogniMem-Arch"]:
    p = ALL / f"{f}.png"
    if p.exists(): setattr(S, f.replace("-","_"), str(p))

def slide(n, t, title, subtitle, body, accent="#8b5cf6"):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden;
  font-family:-apple-system,'Segoe UI',system-ui,sans-serif;
  background:linear-gradient(135deg,#0b1120,#151b30,#0b1120);
  color:#e2e8f0; display:flex; flex-direction:column; }}
.sl {{ flex:1; display:flex; flex-direction:column; padding:32px 44px 20px; }}
.hd {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
.brand {{ font-size:12px; font-weight:700; background:linear-gradient(135deg,#60a5fa,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:1px; }}
.page {{ font-size:11px; color:#334155; }}
.tt {{ font-size:26px; font-weight:700; margin-bottom:4px; }}
.tt .hl {{ background:linear-gradient(135deg,{accent},#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.sb {{ font-size:13px; color:#64748b; margin-bottom:14px; line-height:1.4; }}
.bd {{ flex:1; overflow:hidden; }}
.ft {{ text-align:center; font-size:10px; color:#1e293b; padding-top:10px; border-top:1px solid #1e293b; }}
img {{ border-radius:8px; border:1px solid #1e293b; width:100%; height:100%; object-fit:contain; }}
.iw {{ flex:1; display:flex; align-items:center; justify-content:center; overflow:hidden; }}
.g2 {{ display:flex; gap:12px; height:100%; }}
.g2 > * {{ flex:1; min-width:0; display:flex; flex-direction:column; }}
.card {{ background:#131c33; border:1px solid #1e293b; border-radius:10px; padding:12px 16px; }}
.ch {{ background:linear-gradient(135deg,#1e3a5f,#1f1550); border-color:{accent}; }}
.tags {{ display:flex; gap:5px; flex-wrap:wrap; }}
.tg {{ background:#1e293b; padding:3px 9px; border-radius:5px; font-size:11px; color:#94a3b8; }}
</style></head><body>
<div class="sl">
  <div class="hd"><span class="brand">◆ CogniMem</span><span class="page">{n}/{t}</span></div>
  <div class="tt"><span class="hl">{title}</span></div>
  <div class="sb">{subtitle}</div>
  <div class="bd">{body}</div>
  <div class="ft">Global AI Hackathon · Qwen Cloud · MemoryAgent Track</div>
</div></body></html>"""

# ═══ 10 SLIDES ═══

def s01_title():
    i = getattr(S, "CM_Chat_Welcome", "")
    return slide(1, 10, "CogniMem",
        "A cognitive memory agent with persistent memory, autonomous planning, and 12 built-in tools.",
        f"""<div class="g2">
  <div class="iw">{f'<img src="{img(i)}">' if i else ''}</div>
  <div style="width:240px;display:flex;flex-direction:column;gap:6px;justify-content:center">
    <div class="card ch"><div style="font-size:20px;font-weight:700;color:#60a5fa">12</div><div style="font-size:11px;color:#94a3b8">Built-in Tools</div></div>
    <div class="card"><div style="font-size:20px;font-weight:700;color:#34d399">5</div><div style="font-size:11px;color:#94a3b8">Recall Levels</div></div>
    <div class="card"><div style="font-size:20px;font-weight:700;color:#f59e0b">6</div><div style="font-size:11px;color:#94a3b8">Governance Signals</div></div>
    <div class="card ch"><div style="font-size:20px;font-weight:700;color:#f87171">0</div><div style="font-size:11px;color:#94a3b8">External Dependencies</div></div>
  </div>
</div>""", "#60a5fa")

def s02_chat():
    i = getattr(S, "CM_Chat_Response", "")
    return slide(2, 10, "💬 Chat with Memory",
        "Streaming SSE chat — agent responds with memory awareness, tool calls, and real-time results.",
        f"""<div class="iw">{f'<img src="{img(i)}">' if i else ''}</div>""", "#a78bfa")

def s03_search():
    i = getattr(S, "CM_Chat_Search", "")
    return slide(3, 10, "🔍 Autonomous Web Search",
        "Agent searches the web, processes results, and responds — all in a single autonomous loop.",
        f"""<div class="iw">{f'<img src="{img(i)}">' if i else ''}</div>""", "#60a5fa")

def s04_create():
    i1, i2, i3 = getattr(S, "CM_Create_Project",""), getattr(S, "CM_Create_Project_Filled",""), getattr(S, "CM_Agent_Created","")
    return slide(4, 10, "➕ Create Project / Agent",
        "Each project has its own isolated memory space. Create once, chat immediately.",
        f"""<div class="g2">
  <div style="display:flex;flex-direction:column;gap:6px">
    <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:0.5px">① Click + to create</div>
    {f'<div class="iw" style="flex:1"><img src="{img(i1)}"></div>' if i1 else ''}
  </div>
  <div style="display:flex;flex-direction:column;gap:6px">
    <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:0.5px">② Name it</div>
    {f'<div class="iw" style="flex:1"><img src="{img(i2)}"></div>' if i2 else ''}
  </div>
  <div style="display:flex;flex-direction:column;gap:6px">
    <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:0.5px">③ Ready!</div>
    {f'<div class="iw" style="flex:1"><img src="{img(i3)}"></div>' if i3 else ''}
  </div>
</div>""", "#34d399")

def s05_dashboard():
    i = getattr(S, "CM_Dashboard_Data", "")
    return slide(5, 10, "📊 Real-Time Dashboard",
        "Health monitoring, memory stats, system status — all at a glance with live updates.",
        f"""<div class="iw">{f'<img src="{img(i)}">' if i else ''}</div>""", "#f59e0b")

def s06_memory():
    i1, i2 = getattr(S, "CM_Dashboard_Data2",""), getattr(S, "CM_Dashboard_Memory","")
    return slide(6, 10, "🧠 Memory Browser",
        "Browse structured triples, track contradictions (SPO), inspect confidence & evidence chains.",
        f"""<div class="g2">
  {f'<div class="iw"><img src="{img(i1)}"></div>' if i1 else ''}
  {f'<div class="iw"><img src="{img(i2)}"></div>' if i2 else ''}
</div>""", "#f87171")

def s07_graph():
    i = getattr(S, "CM_Graph_Page", "")
    return slide(7, 10, "🔗 Knowledge Graph",
        "Visualize relationships between stored facts. See connections, clusters, and patterns.",
        f"""<div class="iw">{f'<img src="{img(i)}">' if i else ''}</div>""", "#c084fc")

def s08_arch():
    i = getattr(S, "CogniMem_Arch", "")
    return slide(8, 10, "🏗️ Architecture",
        "5-layer design: Chat UI → Agent Engine → CogniMem Brain → LLM → PostgreSQL Infrastructure.",
        f"""<div class="iw">{f'<img src="{img(i)}">' if i else ''}</div>""", "#60a5fa")

def s09_four():
    return slide(9, 10, "🎯 The Four \"Betters\"",
        "Design philosophy driving every decision in CogniMem.",
        """<div style="display:flex;flex-direction:column;gap:10px;height:100%;justify-content:center">
  <div style="display:flex;gap:12px">
    <div style="flex:1;background:linear-gradient(135deg,#1e3a5f,#1a1f35);border:1px solid #60a5fa;border-radius:12px;padding:18px 22px">
      <div style="font-size:20px;margin-bottom:6px">🦾</div>
      <div style="font-size:16px;font-weight:700;color:#60a5fa;margin-bottom:4px">Smarter</div>
      <div style="font-size:12px;color:#94a3b8;line-height:1.5">Structured SPO triples, contradiction detection, pattern abstraction — not raw text chunks.</div>
    </div>
    <div style="flex:1;background:linear-gradient(135deg,#064e3b,#1a1f35);border:1px solid #34d399;border-radius:12px;padding:18px 22px">
      <div style="font-size:20px;margin-bottom:6px">💰</div>
      <div style="font-size:16px;font-weight:700;color:#34d399;margin-bottom:4px">Token Efficient</div>
      <div style="font-size:12px;color:#94a3b8;line-height:1.5">0-token rule extraction, 0-token cache hits, 0-token BM25 — LLM only as last resort.</div>
    </div>
  </div>
  <div style="display:flex;gap:12px">
    <div style="flex:1;background:linear-gradient(135deg,#451a03,#1a1f35);border:1px solid #f59e0b;border-radius:12px;padding:18px 22px">
      <div style="font-size:20px;margin-bottom:6px">⚡</div>
      <div style="font-size:16px;font-weight:700;color:#f59e0b;margin-bottom:4px">Resource Efficient</div>
      <div style="font-size:12px;color:#94a3b8;line-height:1.5">Ebbinghaus forgetting curve, pure-Python vector search, auto-consolidation — zero extra infra.</div>
    </div>
    <div style="flex:1;background:linear-gradient(135deg,#7c1d1d,#1a1f35);border:1px solid #f87171;border-radius:12px;padding:18px 22px">
      <div style="font-size:20px;margin-bottom:6px">💡</div>
      <div style="font-size:16px;font-weight:700;color:#f87171;margin-bottom:4px">Innovative</div>
      <div style="font-size:12px;color:#94a3b8;line-height:1.5">Contradiction-driven learning, scientific forgetting, autonomous agent loop with self-correction.</div>
    </div>
  </div>
</div>""", "#60a5fa")

def s10_closing():
    return slide(10, 10, "Try It Now",
        "Fully open source. Live demo on Alibaba Cloud ECS.",
        """<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;height:100%">
  <div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center">
    <div style="background:linear-gradient(135deg,#1e3a5f,#2d1b69);border:1px solid #60a5fa;border-radius:12px;padding:20px 30px;text-align:center;min-width:240px">
      <div style="font-size:13px;color:#94a3b8;margin-bottom:4px">🌐 Live Demo</div>
      <div style="font-size:17px;font-weight:600;color:#60a5fa">47.99.151.253:8000</div>
    </div>
    <div style="background:#131c33;border:1px solid #1e293b;border-radius:12px;padding:20px 30px;text-align:center;min-width:240px">
      <div style="font-size:13px;color:#94a3b8;margin-bottom:4px">📦 GitHub</div>
      <div style="font-size:17px;font-weight:600;color:#f1f5f9">github.com/1421906110/qwen-memoryagent</div>
    </div>
  </div>
  <div class="tags" style="justify-content:center;gap:8px">
    <span class="tg" style="background:#1e3a5f;color:#60a5fa">Qwen Cloud</span>
    <span class="tg" style="background:#1e3a5f;color:#34d399">DeepSeek</span>
    <span class="tg" style="background:#1e3a5f;color:#f59e0b">Python</span>
    <span class="tg" style="background:#1e3a5f;color:#a78bfa">FastAPI</span>
    <span class="tg" style="background:#1e3a5f;color:#f87171">PostgreSQL</span>
    <span class="tg" style="background:#1e3a5f;color:#fb923c">Alibaba Cloud</span>
  </div>
</div>""", "#60a5fa")

SLIDES = [s01_title, s02_chat, s03_search, s04_create, s05_dashboard, s06_memory, s07_graph, s08_arch, s09_four, s10_closing]

def main():
    out = ROOT / "output" / "cognimem_demo.mp4"
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cm_") as td:
        from playwright.sync_api import sync_playwright
        pngs = []
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=True)
            pg = br.new_page(viewport={"width": W, "height": H})
            for i, fn in enumerate(SLIDES):
                html = fn()
                hp = os.path.join(td, f"s{i+1:02d}.html")
                with open(hp, "w") as f: f.write(html)
                pg.goto(f"file://{hp}", wait_until="networkidle")
                pp = os.path.join(td, f"s{i+1:02d}.png")
                pg.screenshot(path=pp)
                pngs.append(pp)
                print(f"  [{i+1}/{len(SLIDES)}] rendering...")
            br.close()
        inputs = []
        for p in pngs:
            inputs.extend(["-loop", "1", "-t", str(DUR), "-i", p])
        filters = []
        for i in range(len(pngs)):
            filters.append(f"[{i}:v]format=yuv420p,fade=t=in:st=0:d=0.5,fade=t=out:st={DUR-0.5}:d=0.5,setpts=PTS-STARTPTS[s{i}]")
        filters.append("".join(f"[s{i}]" for i in range(len(pngs))) + f"concat=n={len(pngs)}:v=1:a=0[out]")
        r = subprocess.run(["ffmpeg","-y"]+inputs+["-filter_complex",";".join(filters),"-map","[out]","-c:v","libx264","-pix_fmt","yuv420p","-r","30",str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            print("FFMPEG ERROR:", r.stderr[-600:])
            return 1
        mb = os.path.getsize(out) / 1048576
        dur = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(out)], capture_output=True, text=True).stdout.strip()
        print(f"\n✅ Video: {out} ({mb:.1f} MB, {dur}s)")
        import shutil
        desk = Path("/Users/baikai/Desktop/CogniMem-Demo.mp4")
        shutil.copy(out, desk)
        print(f"✅ Desktop: {desk}")

if __name__ == "__main__":
    sys.exit(main())
