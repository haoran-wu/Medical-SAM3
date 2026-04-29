#!/usr/bin/env python3
"""
hpc_dashboard.py — 在浏览器里看 HPC job 状态

用法: python hpc_dashboard.py
然后浏览器会自动打开 http://localhost:8765
"""

import json
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "bouchet"
LOG_DIR = "/home/hw646/Medical-SAM3/output/visium_hd_exp1"
PORT = 8765

# ─── 数据获取 ─────────────────────────────────────────────────────────────────

_cache = {"jobs": [], "last_update": None, "error": None}
_lock = threading.Lock()


def ssh(cmd: str) -> str:
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", HOST, cmd],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def fetch_jobs() -> list[dict]:
    out = ssh("squeue -u hw646 --format='%.10i|%.18j|%.8T|%.12M|%R' --noheader 2>/dev/null")
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        jid = parts[0].strip()
        job = {
            "id":     jid,
            "name":   parts[1].strip(),
            "state":  parts[2].strip(),
            "time":   parts[3].strip(),
            "reason": parts[4].strip(),
            "log":    "",
        }
        if job["state"] == "RUNNING":
            job["log"] = ssh(
                f"find {LOG_DIR} -name '*{jid}*' 2>/dev/null | head -1 "
                f"| xargs -I{{}} tail -8 {{}} 2>/dev/null"
            )
        jobs.append(job)
    return jobs


def refresh_loop():
    while True:
        try:
            jobs = fetch_jobs()
            with _lock:
                _cache["jobs"] = jobs
                _cache["last_update"] = datetime.now().strftime("%H:%M:%S")
                _cache["error"] = None
        except Exception as e:
            with _lock:
                _cache["error"] = str(e)
                _cache["last_update"] = datetime.now().strftime("%H:%M:%S")
        time.sleep(30)


# ─── HTML ─────────────────────────────────────────────────────────────────────

def render_html(jobs, last_update, error) -> str:
    def state_badge(state):
        colors = {
            "RUNNING": ("#22c55e", "#dcfce7"),
            "PENDING": ("#f59e0b", "#fef3c7"),
            "FAILED":  ("#ef4444", "#fee2e2"),
            "CANCELLED": ("#6b7280", "#f3f4f6"),
        }
        fg, bg = colors.get(state, ("#6b7280", "#f3f4f6"))
        icons = {"RUNNING": "▶", "PENDING": "◌", "FAILED": "✗", "CANCELLED": "✗"}
        icon = icons.get(state, "?")
        return (f'<span style="background:{bg};color:{fg};padding:3px 10px;'
                f'border-radius:12px;font-weight:600;font-size:13px">'
                f'{icon} {state}</span>')

    rows = ""
    if error:
        rows = f'<tr><td colspan="5" style="color:#ef4444;padding:20px">SSH 连接失败: {error}</td></tr>'
    elif not jobs:
        rows = '<tr><td colspan="5" style="color:#9ca3af;padding:20px;text-align:center">队列里没有 job</td></tr>'
    else:
        for j in jobs:
            reason = f'<span style="color:#9ca3af;font-size:12px">({j["reason"]})</span>' if j["reason"] not in ("None", "") else ""
            log_html = ""
            if j["log"]:
                lines = "\n".join(j["log"].splitlines()[-8:])
                log_html = (f'<tr><td colspan="5" style="padding:0 16px 12px 16px">'
                            f'<pre style="background:#1e1e2e;color:#cdd6f4;padding:10px 14px;'
                            f'border-radius:8px;font-size:12px;margin:0;overflow-x:auto;'
                            f'white-space:pre-wrap">{lines}</pre></td></tr>')
            rows += f"""
            <tr style="border-bottom:1px solid #f3f4f6">
              <td style="padding:12px 16px;font-family:monospace;font-weight:600;color:#6366f1">{j['id']}</td>
              <td style="padding:12px 16px;font-weight:500">{j['name']}</td>
              <td style="padding:12px 16px">{state_badge(j['state'])}</td>
              <td style="padding:12px 16px;color:#6b7280">{j['time']}</td>
              <td style="padding:12px 16px">{reason}</td>
            </tr>
            {log_html}"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="30">
  <title>HPC Jobs</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0 }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f8fafc; color: #1e293b; padding: 32px }}
    h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px }}
    .subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 24px }}
    .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); overflow: hidden }}
    table {{ width: 100%; border-collapse: collapse }}
    th {{ background: #f8fafc; padding: 10px 16px; text-align: left;
          font-size: 12px; font-weight: 600; color: #64748b;
          text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid #e2e8f0 }}
    tr:hover {{ background: #fafafa }}
    .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
             background:#22c55e; margin-right:6px;
             animation: pulse 2s infinite }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.4}} }}
    .footer {{ margin-top: 16px; color: #94a3b8; font-size: 13px; display:flex; justify-content:space-between }}
  </style>
</head>
<body>
  <h1>HPC Job Monitor</h1>
  <div class="subtitle">bouchet · hw646 · 每 30 秒自动刷新</div>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Job ID</th><th>名称</th><th>状态</th><th>运行时间</th><th>备注</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div class="footer">
    <span><span class="dot"></span>实时监控中</span>
    <span>最后更新: {last_update or "—"}</span>
  </div>
</body>
</html>"""


# ─── HTTP server ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _lock:
            html = render_html(_cache["jobs"], _cache["last_update"], _cache["error"])
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, *_):
        pass  # 关掉 HTTP 请求日志


def main():
    # 先抓一次数据
    print("连接 HPC，获取初始数据...")
    try:
        jobs = fetch_jobs()
        with _lock:
            _cache["jobs"] = jobs
            _cache["last_update"] = datetime.now().strftime("%H:%M:%S")
    except Exception as e:
        with _lock:
            _cache["error"] = str(e)

    # 后台定时刷新
    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()

    # 启动 HTTP server
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Dashboard 已启动 → {url}")
    print("Ctrl+C 停止\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")


if __name__ == "__main__":
    main()
