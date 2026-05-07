#!/usr/bin/env python3
"""
Local web dashboard for Bouchet SLURM jobs.

Usage:
  python3 hpc_dashboard.py
  python3 hpc_dashboard.py --open

Then visit:
  http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shlex
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath


HOST = "bouchet"
HPC_USER = "hw646"
PROJECT_ROOT = "/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3"
PROJECT_RESULTS = f"{PROJECT_ROOT}/results/visium_hd_exp1"
PROJECT_LOG_DIR = f"{PROJECT_RESULTS}/logs"
PROJECT_TRAINING_DIR = f"{PROJECT_RESULTS}/training_runs"
LEGACY_SCRATCH_LOG_DIR = "/nfs/roberts/scratch/pi_xy48/hw646/Medical-SAM3/visium_hd_exp1/logs"
DEFAULT_PORT = 8765
DEFAULT_INTERVAL = 20
DEFAULT_HISTORY_HOURS = 24


@dataclass
class Job:
    id: str
    name: str
    partition: str
    state: str
    elapsed: str
    time_limit: str
    node_or_reason: str
    submit: str = ""
    priority: str = ""
    gres: str = ""
    start: str = ""
    end: str = ""
    exit_code: str = ""
    state_detail: str = ""
    finish_reason: str = ""
    log_path: str = ""
    log_tail: str = ""
    output_dir: str = ""
    script_path_hint: str = ""
    source: str = "squeue"


_cache = {
    "jobs": [],
    "last_update": None,
    "error": None,
    "project_log_dir": PROJECT_LOG_DIR,
    "project_training_dir": PROJECT_TRAINING_DIR,
    "history_hours": DEFAULT_HISTORY_HOURS,
    "interval": DEFAULT_INTERVAL,
}
_config = {"history_hours": DEFAULT_HISTORY_HOURS, "interval": DEFAULT_INTERVAL}
_lock = threading.Lock()


def ssh(cmd: str, timeout: int = 20) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(stderr or f"ssh command failed with exit code {result.returncode}")
    return result.stdout.rstrip("\n")


def parse_pipe_table(text: str, expected_min: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= expected_min:
            rows.append(parts)
    return rows


def find_output_dir(script: str) -> str:
    if not script:
        return ""
    match = re.search(r"--output-dir(?:=|\s+)([^\s\\]+)", script)
    if match:
        return match.group(1).strip("'\"")
    match = re.search(r"#SBATCH\s+--output=([^\s]+)", script)
    if match:
        return str(PurePosixPath(match.group(1).strip("'\"")).parent)
    return ""


def get_batch_script(job_id: str) -> str:
    cmd = f"scontrol write batch_script {shlex.quote(job_id)} - 2>/dev/null || true"
    return ssh(cmd, timeout=12)


def get_log_tail(job_id: str, lines: int = 36) -> tuple[str, str]:
    quoted_project = shlex.quote(PROJECT_LOG_DIR)
    quoted_legacy = shlex.quote(LEGACY_SCRATCH_LOG_DIR)
    quoted_job = shlex.quote(job_id)
    cmd = (
        f"for d in {quoted_project} {quoted_legacy}; do "
        f"  [ -d \"$d\" ] || continue; "
        f"  f=$(find \"$d\" -maxdepth 1 -type f -name '*'{quoted_job}'*' 2>/dev/null | sort | tail -1); "
        f"  if [ -n \"$f\" ]; then echo __LOG_PATH__\"$f\"; tail -n {int(lines)} \"$f\"; exit 0; fi; "
        f"done"
    )
    out = ssh(cmd, timeout=15)
    if out.startswith("__LOG_PATH__"):
        first, *rest = out.splitlines()
        return first.replace("__LOG_PATH__", "", 1), "\n".join(rest)
    return "", ""


def infer_finish_reason(job: Job) -> str:
    state_detail = job.state_detail or job.state
    if job.state in {"RUNNING", "PENDING"}:
        return job.node_or_reason or state_detail
    if job.state == "COMPLETED" and job.exit_code in {"0:0", "0"}:
        return "正常完成"
    if job.state == "CANCELLED":
        return state_detail if state_detail != "CANCELLED" else "任务被取消"

    patterns = [
        r"(ModuleNotFoundError: .+)",
        r"(FileNotFoundError: .+)",
        r"(AssertionError: .+)",
        r"(RuntimeError: .+)",
        r"(CUDA out of memory.+)",
        r"(OutOfMemoryError: .+)",
        r"(No space left on device.+)",
        r"(ImportError: .+)",
        r"(ValueError: .+)",
        r"(TypeError: .+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, job.log_tail or "", flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    if job.exit_code:
        return f"{state_detail}, exit={job.exit_code}"
    return state_detail


def fetch_squeue_jobs() -> list[Job]:
    fmt = "%.18i|%.40j|%.16P|%.12T|%.16M|%.14l|%R|%.10Q|%b"
    out = ssh(
        f"squeue -u {shlex.quote(HPC_USER)} --format={shlex.quote(fmt)} --noheader 2>/dev/null",
        timeout=15,
    )
    jobs: list[Job] = []
    for parts in parse_pipe_table(out, 7):
        job = Job(
            id=parts[0],
            name=parts[1],
            partition=parts[2],
            state=parts[3],
            elapsed=parts[4],
            time_limit=parts[5],
            node_or_reason=parts[6],
            priority=parts[7] if len(parts) > 7 else "",
            gres=parts[8] if len(parts) > 8 else "",
            state_detail=parts[3],
        )
        try:
            script = get_batch_script(job.id)
            job.output_dir = find_output_dir(script)
            if PROJECT_ROOT in script:
                job.script_path_hint = "project"
            elif "/scratch/" in script:
                job.script_path_hint = "scratch"
            elif "/home/" in script:
                job.script_path_hint = "home"
        except Exception as exc:
            job.script_path_hint = f"script unavailable: {exc}"
        try:
            job.log_path, job.log_tail = get_log_tail(job.id)
        except Exception as exc:
            job.log_tail = f"log unavailable: {exc}"
        job.finish_reason = infer_finish_reason(job)
        jobs.append(job)
    return jobs


def fetch_sacct_jobs(hours: float) -> list[Job]:
    start = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    fmt = "JobIDRaw,JobName%50,Partition,State,Submit,Start,End,Elapsed,ExitCode,NodeList"
    cmd = (
        f"sacct -u {shlex.quote(HPC_USER)} --starttime {shlex.quote(start)} "
        f"--format={shlex.quote(fmt)} --parsable2 --noheader 2>/dev/null || true"
    )
    out = ssh(cmd, timeout=18)
    jobs: list[Job] = []
    for parts in parse_pipe_table(out, 10):
        job_id = parts[0]
        if not job_id or "." in job_id:
            continue
        state_detail = parts[3]
        state = state_detail.split()[0]
        job = Job(
            id=job_id,
            name=parts[1],
            partition=parts[2],
            state=state,
            elapsed=parts[7],
            time_limit="",
            node_or_reason=parts[9],
            submit=parts[4],
            start=parts[5],
            end=parts[6],
            exit_code=parts[8],
            state_detail=state_detail,
            source="sacct",
        )
        job.finish_reason = infer_finish_reason(job)
        jobs.append(job)
    return jobs


def fetch_jobs() -> list[Job]:
    active = fetch_squeue_jobs()
    active_by_id = {job.id: job for job in active}
    history = fetch_sacct_jobs(float(_config["history_hours"]))
    merged: dict[str, Job] = {}
    for job in history:
        if job.id in active_by_id:
            live = active_by_id[job.id]
            job.state = live.state
            job.state_detail = live.state_detail or live.state
            job.elapsed = live.elapsed or job.elapsed
            job.time_limit = live.time_limit
            job.node_or_reason = live.node_or_reason
            job.priority = live.priority
            job.gres = live.gres
            job.output_dir = live.output_dir
            job.script_path_hint = live.script_path_hint
            job.source = "squeue+sacct"
            if live.log_path:
                job.log_path = live.log_path
                job.log_tail = live.log_tail
            job.finish_reason = infer_finish_reason(job)
        merged[job.id] = job
    for job in active:
        merged.setdefault(job.id, job)

    def sort_key(job: Job) -> str:
        return job.submit or job.start or job.end or ""

    jobs = sorted(merged.values(), key=sort_key, reverse=True)[:60]
    log_budget = 24
    for job in jobs:
        if log_budget <= 0:
            break
        if job.log_tail:
            continue
        if job.state in {"RUNNING", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"}:
            try:
                job.log_path, job.log_tail = get_log_tail(job.id, lines=24)
                job.finish_reason = infer_finish_reason(job)
            except Exception:
                pass
            log_budget -= 1
    return jobs


def refresh_once() -> None:
    jobs = fetch_jobs()
    with _lock:
        _cache["jobs"] = [asdict(job) for job in jobs]
        _cache["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _cache["error"] = None
        _cache["history_hours"] = _config["history_hours"]
        _cache["interval"] = _config["interval"]


def refresh_loop(interval: int) -> None:
    while True:
        try:
            refresh_once()
        except Exception as exc:
            with _lock:
                _cache["error"] = str(exc)
                _cache["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(interval)


def badge(state: str) -> str:
    palette = {
        "RUNNING": ("running", "运行中"),
        "PENDING": ("pending", "排队中"),
        "COMPLETED": ("done", "完成"),
        "FAILED": ("bad", "失败"),
        "TIMEOUT": ("bad", "超时"),
        "OUT_OF_MEMORY": ("bad", "显存/内存爆了"),
        "NODE_FAIL": ("bad", "节点故障"),
        "CANCELLED": ("muted", "已取消"),
    }
    klass, label = palette.get(state, ("muted", state))
    return f'<span class="badge {klass}">{html.escape(label)}</span>'


def path_tag(path: str) -> str:
    if not path:
        return '<span class="dim">未找到</span>'
    klass = "path"
    if path.startswith(PROJECT_ROOT):
        klass += " project"
    elif "/scratch/" in path:
        klass += " scratch"
    elif "/home/" in path:
        klass += " home"
    return f'<code class="{klass}">{html.escape(path)}</code>'


def render_job_card(job: dict) -> str:
    output = path_tag(job.get("output_dir", ""))
    log_path = path_tag(job.get("log_path", ""))
    note = html.escape(job.get("node_or_reason") or "")
    tail = html.escape(job.get("log_tail") or "")
    source = "当前队列 + 历史" if job.get("source") == "squeue+sacct" else ("当前队列" if job.get("source") == "squeue" else "历史台账")
    finish_reason = html.escape(job.get("finish_reason") or "")
    warning = ""
    hint = job.get("script_path_hint") or ""
    if hint == "home":
        warning = '<div class="warn">注意：这个 job script 里仍出现 home 路径，需要迁到 project。</div>'
    elif hint == "scratch":
        warning = '<div class="warn soft">这个 job script 使用 scratch，适合临时缓存，但最终结果应同步回 project。</div>'

    state = job.get("state") or ""
    elapsed = html.escape(job.get("elapsed") or "-")
    time_limit = html.escape(job.get("time_limit") or "-")
    raw_exit_code = job.get("exit_code") or "-"
    exit_code = "-" if state in {"RUNNING", "PENDING"} else html.escape(raw_exit_code)
    gres = html.escape(job.get("gres") or "-")
    partition = html.escape(job.get("partition") or "-")
    job_id = html.escape(job.get("id") or "")
    job_name = html.escape(job.get("name") or "")
    detail_hint = {
        "RUNNING": ("运行时长", f"{elapsed} / {time_limit}"),
        "PENDING": ("排队原因", note or "-"),
        "COMPLETED": ("已完成", elapsed),
    }.get(state, ("退出原因", finish_reason or state or "-"))
    action_hint = {
        "RUNNING": "还在运行；重点看日志是否持续更新、有没有报错。",
        "PENDING": f"等待原因：{note or '-'}",
        "COMPLETED": "正常完成；需要时再点开更多信息确认输出路径。",
    }.get(state, finish_reason or "异常退出；先看日志里的最后一个错误。")
    log_block = (
        f"""<details class="logbox">
          <summary>查看最近日志</summary>
          <pre>{tail or '还没有找到日志，可能 job 还没开始写文件。'}</pre>
        </details>"""
    )

    return f"""
    <article class="job-card {html.escape(state.lower())}">
      <div class="job-top">
        <div>
          <div class="job-name">{job_name}</div>
          <div class="job-id">#{job_id} · {partition} · {source}</div>
        </div>
        {badge(job.get('state') or '')}
      </div>
      <div class="job-summary">
        <div><span>{html.escape(detail_hint[0])}</span><strong>{html.escape(detail_hint[1])}</strong></div>
        <div><span>节点 / 等待</span><strong>{note or '-'}</strong></div>
        <div><span>GPU</span><strong>{gres}</strong></div>
        <div><span>Exit</span><strong>{exit_code}</strong></div>
      </div>
      <div class="reason"><span>我应该看什么</span><strong>{html.escape(action_hint)}</strong></div>
      {warning}
      <details class="morebox">
        <summary>更多信息</summary>
        <div class="details-grid">
          <div><span>提交</span><strong>{html.escape(job.get('submit') or '-')}</strong></div>
          <div><span>开始</span><strong>{html.escape(job.get('start') or '-')}</strong></div>
          <div><span>结束</span><strong>{html.escape(job.get('end') or '-')}</strong></div>
          <div><span>Time limit</span><strong>{time_limit}</strong></div>
        </div>
        <div class="paths">
          <div><span>输出</span>{output}</div>
          <div><span>日志</span>{log_path}</div>
        </div>
      </details>
      {log_block}
    </article>
    """


def section_block(title: str, subtitle: str, cards: str, klass: str = "") -> str:
    section_class = f"section-card {klass}".strip()
    return f"""
    <section class="{section_class}">
      <div class="section-head">
        <div>
          <h2>{html.escape(title)}</h2>
          <div class="section-subtitle">{html.escape(subtitle)}</div>
        </div>
      </div>
      <div class="grid">{cards}</div>
    </section>
    """


def render_html() -> str:
    with _lock:
        jobs = list(_cache["jobs"])
        last_update = _cache["last_update"]
        error = _cache["error"]
        history_hours = _cache.get("history_hours", DEFAULT_HISTORY_HOURS)
        interval = _cache.get("interval", DEFAULT_INTERVAL)

    visible_jobs = [job for job in jobs if job.get("state") != "CANCELLED"]
    running_jobs = [job for job in visible_jobs if job.get("state") == "RUNNING"]
    pending_jobs = [job for job in visible_jobs if job.get("state") == "PENDING"]
    alert_jobs = [job for job in visible_jobs if job.get("state") in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}]
    completed_jobs = [job for job in visible_jobs if job.get("state") == "COMPLETED"]

    counts = {
        "RUNNING": sum(1 for job in visible_jobs if job.get("state") == "RUNNING"),
        "PENDING": sum(1 for job in visible_jobs if job.get("state") == "PENDING"),
        "BAD": len(alert_jobs),
        "DONE": sum(1 for job in visible_jobs if job.get("state") == "COMPLETED"),
    }

    cards = []
    if running_jobs:
        cards.append(
            section_block(
                "正在运行",
                "先看这里：任务已经拿到资源，重点看运行时长、节点和日志有没有异常。",
                "\n".join(render_job_card(job) for job in running_jobs),
                "section-live",
            )
        )
    if pending_jobs:
        cards.append(
            section_block(
                "正在排队",
                "只保留排队原因和资源请求，Priority / Resources 一眼能看到。",
                "\n".join(render_job_card(job) for job in pending_jobs),
                "section-pending",
            )
        )
    if alert_jobs:
        cards.append(
            section_block(
                "需要注意",
                "异常退出、超时、爆显存/内存、节点故障。这里主要看退出原因和日志。",
                "\n".join(render_job_card(job) for job in alert_jobs),
                "section-alert",
            )
        )
    if completed_jobs:
        cards.append(
            section_block(
                "刚正常结束",
                "正常完成的任务放最后，只做快速确认。",
                "\n".join(render_job_card(job) for job in completed_jobs),
            )
        )
    if not cards:
        cards.append('<div class="empty">当前没有排队、运行、异常退出或刚完成的 job。</div>')

    cards_html = "\n".join(cards)
    error_html = f'<div class="error">SSH/SLURM 读取失败：{html.escape(error)}</div>' if error else ""

    headline = "没有新的异常退出，先看正在运行和排队。" if not alert_jobs else f"{len(alert_jobs)} 个异常退出需要处理；正在运行仍放在最前面。"

    payload = json.dumps(
        {
            "last_update": last_update,
            "jobs": visible_jobs,
            "project_log_dir": PROJECT_LOG_DIR,
            "project_training_dir": PROJECT_TRAINING_DIR,
            "history_hours": history_hours,
            "interval": interval,
        },
        ensure_ascii=False,
    )

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{interval}">
  <title>Medical-SAM3 HPC Radar</title>
  <style>
    :root {{
      --bg: #eef1f4;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #647181;
      --line: #d9e0e7;
      --line-strong: #c4cdd7;
      --green: #16804f;
      --amber: #b77905;
      --red: #c0342b;
      --blue: #2f6da5;
      --shadow: 0 14px 38px rgba(21, 35, 49, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      background:
        linear-gradient(180deg, #f8fafc 0, var(--bg) 280px),
        var(--bg);
      min-height: 100vh;
    }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px 22px 48px; }}
    header {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(420px, .9fr); gap: 18px; align-items: stretch; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 34px; line-height: 1; font-weight: 800; }}
    .subtitle {{ color: var(--muted); font-size: 13px; margin-top: 9px; max-width: 720px; line-height: 1.45; }}
    .status-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
    .stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 13px; box-shadow: var(--shadow); position: relative; overflow: hidden; }}
    .stat::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--line-strong); }}
    .stat:nth-child(1)::before {{ background: var(--green); }}
    .stat:nth-child(2)::before {{ background: var(--amber); }}
    .stat:nth-child(3)::before {{ background: var(--blue); }}
    .stat:nth-child(4)::before {{ background: var(--red); }}
    .stat span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .stat strong {{ display: block; font-size: 30px; line-height: 1.05; margin-top: 6px; }}
    .toolbar {{ display: flex; justify-content: space-between; align-items: center; gap: 14px; background: #17202a; color: #edf5ef; border-radius: 8px; padding: 10px 13px; margin: 12px 0; font-size: 13px; }}
    .toolbar > div {{ min-width: 0; }}
    .toolbar code {{ color: #bfdfcb; overflow-wrap: anywhere; word-break: break-word; }}
    .headline {{ margin: 0 0 18px; padding: 11px 13px; border-radius: 8px; background: var(--panel); border: 1px solid var(--line); font-weight: 700; box-shadow: var(--shadow); }}
    .headline.alert {{ background: #fff2f0; border-color: rgba(192, 52, 43, .34); color: #8f251f; }}
    .dot {{ width: 8px; height: 8px; display: inline-block; border-radius: 99px; background: #57d68b; margin-right: 8px; box-shadow: 0 0 0 5px rgba(87, 214, 139, .14); }}
    .grid {{ display: grid; gap: 10px; }}
    .sections {{ display: grid; gap: 22px; }}
    .section-card {{ display: grid; gap: 9px; }}
    .section-head {{ display: flex; justify-content: space-between; align-items: end; gap: 12px; padding: 0 2px; }}
    .section-head h2 {{ margin: 0; font-size: 18px; font-weight: 800; }}
    .section-subtitle {{ color: var(--muted); font-size: 13px; margin-top: 3px; }}
    .job-card {{ background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--line-strong); border-radius: 8px; padding: 14px 15px; box-shadow: var(--shadow); }}
    .job-card.running {{ border-left-color: var(--green); }}
    .job-card.pending {{ border-left-color: var(--amber); }}
    .job-card.failed, .job-card.timeout, .job-card.out_of_memory, .job-card.node_fail {{ border-left-color: var(--red); }}
    .section-alert .job-card {{ background: #fffaf9; }}
    .job-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }}
    .job-name {{ font-size: 17px; font-weight: 800; }}
    .job-id {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .badge {{ border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 800; white-space: nowrap; }}
    .badge.running {{ color: #085a34; background: #dff7e9; }}
    .badge.pending {{ color: #805000; background: #ffefc4; }}
    .badge.done {{ color: #17456f; background: #dfeeff; }}
    .badge.bad {{ color: #961f14; background: #ffe1dc; }}
    .badge.muted {{ color: #4f5d55; background: #e8ece7; }}
    .job-summary {{ display: grid; grid-template-columns: 1.25fr 1.25fr .75fr .6fr; gap: 8px; margin: 12px 0; }}
    .job-summary div, .details-grid div {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; background: #f8fafc; min-width: 0; }}
    .job-summary span, .details-grid span, .paths span, .reason span {{ display: block; color: var(--muted); font-size: 11px; margin-bottom: 4px; }}
    .job-summary strong, .details-grid strong {{ font-size: 13px; overflow-wrap: anywhere; }}
    .reason {{ border-radius: 6px; padding: 9px 10px; background: #f5f8fb; border: 1px solid var(--line); }}
    .reason strong {{ overflow-wrap: anywhere; }}
    .morebox {{ margin-top: 10px; }}
    .morebox summary {{ cursor: pointer; color: var(--blue); font-weight: 800; }}
    .details-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    .paths {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }}
    code.path {{ display: block; padding: 8px 9px; border-radius: 6px; background: #f4f6f8; overflow-wrap: anywhere; font-size: 12px; }}
    code.project {{ outline: 1px solid rgba(22, 115, 72, .26); }}
    code.scratch, code.home {{ outline: 1px solid rgba(184, 120, 8, .34); }}
    .warn {{ margin-top: 10px; border-radius: 6px; padding: 9px 10px; background: #ffe1dc; color: #7f1d1d; font-weight: 700; }}
    .warn.soft {{ background: #fff0cd; color: #704700; }}
    details.logbox {{ margin-top: 10px; }}
    details.logbox summary {{ cursor: pointer; color: var(--blue); font-weight: 800; }}
    pre {{ white-space: pre-wrap; overflow-x: auto; background: #111820; color: #eaf7ea; padding: 12px; border-radius: 6px; font-size: 12px; line-height: 1.45; }}
    .empty, .error {{ border: 1px dashed var(--line); border-radius: 8px; padding: 24px; background: var(--panel); color: var(--muted); text-align: center; }}
    .error {{ color: var(--red); border-color: rgba(180, 35, 24, .35); }}
    .dim {{ color: var(--muted); }}
    @media (max-width: 980px) {{
      header {{ grid-template-columns: 1fr; }}
      .paths {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 860px) {{
      .status-strip, .job-summary, .details-grid {{ grid-template-columns: 1fr 1fr; }}
      .toolbar {{ align-items: flex-start; flex-direction: column; }}
    }}
    @media (max-width: 560px) {{
      main {{ padding: 18px 14px 36px; }}
      .status-strip, .job-summary, .details-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>HPC Monitor</h1>
        <div class="subtitle">Bouchet · hw646 · 最近 {history_hours} 小时。异常退出置顶，取消的任务不显示，日志默认盯 project。</div>
      </div>
      <section class="status-strip">
        <div class="stat"><span>Running</span><strong>{counts['RUNNING']}</strong></div>
        <div class="stat"><span>Pending</span><strong>{counts['PENDING']}</strong></div>
        <div class="stat"><span>Done</span><strong>{counts['DONE']}</strong></div>
        <div class="stat"><span>Need Action</span><strong>{counts['BAD']}</strong></div>
      </section>
    </header>
    <div class="toolbar">
      <div><span class="dot"></span>每 {interval}s 自动刷新 · 最后更新：{html.escape(last_update or '还没成功读取')}</div>
      <div>project logs: <code>{html.escape(PROJECT_LOG_DIR)}</code></div>
    </div>
    <div class="headline {'alert' if alert_jobs else ''}">{html.escape(headline)}</div>
    {error_html}
    <section class="sections">{cards_html}</section>
  </main>
  <script id="payload" type="application/json">{html.escape(payload)}</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/jobs"):
            with _lock:
                payload = json.dumps(_cache, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        payload = render_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Bouchet SLURM dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--history-hours", type=float, default=DEFAULT_HISTORY_HOURS,
                        help="How many hours of submitted/finished jobs to keep visible")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _config["history_hours"] = args.history_hours
    _config["interval"] = args.interval
    with _lock:
        _cache["history_hours"] = args.history_hours
        _cache["interval"] = args.interval

    thread = threading.Thread(target=refresh_loop, args=(args.interval,), daemon=True)
    thread.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard running: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
