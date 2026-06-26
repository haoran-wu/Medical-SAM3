#!/usr/bin/env python3
"""Report HPC job progress and optionally push state-change summaries.

Examples:
  python hpc_progress_reporter.py --once
  python hpc_progress_reporter.py --once --job-ids 10811364,10811365,10811366
  python hpc_progress_reporter.py --watch --interval 60 --notify-local-on change
  python hpc_progress_reporter.py --watch --interval 300 --email-to you@example.com
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import smtplib
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable


DEFAULT_HOST = "bouchet"
DEFAULT_USER = "hw646"
DEFAULT_LOG_DIR = "/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs"
DEFAULT_EXTRA_LOG_DIRS = "/nfs/roberts/scratch/pi_xy48/hw646/Medical-SAM3/visium_hd_exp1/logs"
DEFAULT_STATE_FILE = Path.home() / ".hpc_progress_reporter_state.json"
DEFAULT_LOCAL_LOG = Path.home() / ".hpc_progress_reporter.log"
DEFAULT_ANALYSIS_DIR = Path("output/hpc_monitoring/analysis")


@dataclass
class Job:
    id: str
    name: str
    state: str
    elapsed: str
    reason: str = ""
    exit_code: str = ""
    source: str = "squeue"


def run_ssh(host: str, cmd: str, timeout: int = 20, retries: int = 2) -> str:
    last_error = ""
    for attempt in range(retries + 1):
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        last_error = result.stderr.strip() or f"ssh exited with {result.returncode}"
        if attempt < retries:
            time.sleep(2)
    raise RuntimeError(last_error)


def parse_job_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_squeue_jobs(host: str, user: str) -> dict[str, Job]:
    cmd = f"squeue -u {user} --format='%.18i|%.32j|%.12T|%.16M|%R' --noheader 2>/dev/null"
    out = run_ssh(host, cmd)
    jobs: dict[str, Job] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, reason = [p.strip() for p in parts[:5]]
        jobs[jid] = Job(jid, name, state, elapsed, reason=reason, source="squeue")
    return jobs


def get_sacct_jobs(host: str, job_ids: Iterable[str]) -> dict[str, Job]:
    ids = [jid for jid in job_ids if jid]
    if not ids:
        return {}
    id_arg = ",".join(ids)
    cmd = (
        f"sacct -j {id_arg} "
        "--format=JobID,JobName,State,Elapsed,ExitCode "
        "--parsable2 --noheader 2>/dev/null"
    )
    out = run_ssh(host, cmd)
    jobs: dict[str, Job] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, exit_code = [p.strip() for p in parts[:5]]
        if "." in jid:
            continue
        jobs[jid] = Job(jid, name, state, elapsed, exit_code=exit_code, source="sacct")
    return jobs


def get_recent_sacct_jobs(host: str, user: str, lookback_hours: int) -> dict[str, Job]:
    if lookback_hours <= 0:
        return {}
    start = (datetime.now() - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M")
    cmd = (
        f"sacct -u {user} --starttime {start} "
        "--format=JobID,JobName,State,Elapsed,ExitCode "
        "--parsable2 --noheader 2>/dev/null"
    )
    out = run_ssh(host, cmd)
    jobs: dict[str, Job] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, exit_code = [p.strip() for p in parts[:5]]
        if "." in jid:
            continue
        jobs[jid] = Job(jid, name, state, elapsed, exit_code=exit_code, source="sacct")
    return jobs


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def log_dirs(primary: str, extra: Iterable[str] | None = None) -> list[str]:
    dirs = [primary]
    dirs.extend([item for item in (extra or []) if item])
    deduped: list[str] = []
    for item in dirs:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def get_log_tail(host: str, log_dir: str, job_id: str, lines: int, extra_log_dirs: Iterable[str] | None = None) -> str:
    dirs = " ".join(shell_quote(item) for item in log_dirs(log_dir, extra_log_dirs))
    quoted_job = shell_quote(job_id)
    cmd = (
        f"for d in {dirs}; do "
        f"  [ -d \"$d\" ] || continue; "
        f"  f=$(find \"$d\" -maxdepth 1 -type f -name '*'{quoted_job}'*.log' 2>/dev/null | sort | tail -1); "
        f"  [ -n \"$f\" ] && tail -n {int(lines)} \"$f\" && exit 0; "
        f"done; true"
    )
    return run_ssh(host, cmd)


def get_job_log_path(host: str, log_dir: str, job_id: str, extra_log_dirs: Iterable[str] | None = None) -> str:
    dirs = " ".join(shell_quote(item) for item in log_dirs(log_dir, extra_log_dirs))
    quoted_job = shell_quote(job_id)
    cmd = (
        f"for d in {dirs}; do "
        f"  [ -d \"$d\" ] || continue; "
        f"  f=$(find \"$d\" -maxdepth 1 -type f -name '*'{quoted_job}'*.log' 2>/dev/null | sort | tail -1); "
        f"  [ -n \"$f\" ] && echo \"$f\" && exit 0; "
        f"done; true"
    )
    return run_ssh(host, cmd)


def get_scheduler_context(host: str, user: str, job_ids: Iterable[str]) -> str:
    ids = ",".join(sorted({jid for jid in job_ids if jid}))
    pieces = []
    if ids:
        pieces.append("Tracked estimated starts:")
        pieces.append(
            run_ssh(
                host,
                f"squeue -j {ids} --start -o '%.12i %.18j %.12P %.20S %.18R' 2>/dev/null || true",
            )
        )
        pieces.append("")
    pieces.append("User queue:")
    pieces.append(
        run_ssh(
            host,
            f"squeue -u {user} -o '%.12i %.18j %.12P %.8T %.10M %.12l %.18R %.10Q %b' 2>/dev/null",
        )
    )
    pieces.append("")
    pieces.append("GPU queue summary:")
    pieces.append(
        run_ssh(
            host,
            "squeue -h -o '%P %T' 2>/dev/null | awk '{c[$1\" \"$2]++} END{for (k in c) print k,c[k]}' | sort",
        )
    )
    return "\n".join(part.strip() for part in pieces if part is not None).strip()


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"jobs": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"jobs": {}}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def save_reporter_error(path: Path, message: str) -> bool:
    state = load_state(path)
    previous = state.get("reporter_error")
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state["reporter_error"] = message
    save_state(path, state)
    return previous != message


def job_to_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "name": job.name,
        "state": job.state,
        "elapsed": job.elapsed,
        "reason": job.reason,
        "exit_code": job.exit_code,
        "source": job.source,
    }


def terminal_state(state: str) -> bool:
    state_upper = state.upper()
    return any(
        key in state_upper
        for key in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
    )


def cancelled_state(state: str) -> bool:
    return "CANCELLED" in state.upper()


def bad_state(state: str) -> bool:
    state_upper = state.upper()
    return any(key in state_upper for key in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"))


def alert_failure_state(state: str) -> bool:
    state_upper = state.upper()
    return any(key in state_upper for key in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"))


def extract_metric_lines(text: str, max_lines: int = 12) -> list[str]:
    patterns = re.compile(
        r"(loss|f1|accuracy|acc|dice|iou|auc|epoch|val|train|macro|micro|best)",
        re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if patterns.search(line)]
    return lines[-max_lines:]


def diagnose_job(job: Job, log_tail: str) -> list[str]:
    state = job.state.upper()
    combined = f"{job.reason}\n{log_tail}".lower()
    advice: list[str] = []

    if "CANCELLED" in state:
        if "by 30116" in state:
            advice.append("这是我们主动取消的任务，不按 failure 处理；如果是替换路径/换队列导致，可以忽略。")
        else:
            advice.append("任务被取消；需要确认是用户取消、依赖取消，还是 scheduler/preemption 导致。")
        return advice

    if state == "PENDING":
        reason = job.reason.lower()
        if "dependency" in reason:
            advice.append("还在等依赖任务结束；先看上游训练任务，不需要单独处理这个 summary job。")
        elif "priority" in reason:
            advice.append("主要卡在 Slurm priority，不是代码问题；可以保留正式队列，同时投 scavenge/gpu_devel 短任务赛马。")
        elif "resources" in reason:
            advice.append("优先级够了但暂时没有匹配资源；降低 CPU/内存/时间或换 scavenge_gpu 可能更快。")
        elif reason:
            advice.append(f"Pending reason 是 {job.reason}；需要结合队列和分区限制判断。")
        else:
            advice.append("Pending 但没有明确原因；建议查 squeue --start 和分区资源。")

    if "disk quota exceeded" in combined or "no space left" in combined or "quota" in combined:
        advice.append("检测到磁盘/quota 类问题；输出必须放 scratch/project，避免写 /home，并检查 checkpoint 是否 0 byte。")
    if "cuda out of memory" in combined or "outofmemory" in combined or "oom" in combined:
        advice.append("检测到 OOM 风险；优先降 batch size / num_workers / crop size，或改投 H200/B200。")
    if "modulenotfounderror" in combined or "importerror" in combined:
        advice.append("检测到 Python 依赖问题；需要在 Bouchet 环境里修 conda/pip，而不是改训练逻辑。")
    if "conda: command not found" in combined:
        advice.append("检测到 sbatch 环境找不到 conda；用 conda env 的 python 绝对路径，或在脚本里 source conda.sh。")
    if "filenotfounderror" in combined or "no such file or directory" in combined:
        advice.append("检测到输入/输出路径不存在；先确认 sbatch 使用的是整理后的 project/scratch 绝对路径。")
    if "not json serializable" in combined or "json/encoder.py" in combined:
        advice.append("检测到 JSON 序列化问题；通常是 Path/NumPy 类型直接写入 json，需要先转成 str/int/float/list。")
    if "traceback (most recent call last)" in combined and not any("Traceback" in item for item in advice):
        advice.append("日志含 Python traceback；应优先修最后一行异常，再重投短 smoke 验证。")
    if "time limit" in combined or "timeout" in state:
        advice.append("任务可能撞 time limit；要么缩小 smoke 验证，要么增加 walltime。")
    if "nan" in combined:
        advice.append("日志里出现 NaN；需要降低 learning rate、检查 class weights/label 和 loss 输入。")
    if state == "RUNNING":
        metric_lines = extract_metric_lines(log_tail, max_lines=4)
        if metric_lines:
            advice.append("训练正在产生日志指标；下一步重点看 val/macro-F1 是否持续上升。")
        else:
            advice.append("任务在 RUNNING 但 tail 里还没看到指标；如果长时间无输出，要检查 data loading 是否卡住。")
    if state == "COMPLETED":
        advice.append("任务完成；下一步应汇总 metrics/checkpoints，比较 best metric 并决定是否扩大训练。")
    if bad_state(state) and not advice:
        advice.append("任务失败但 tail 没有明显模式；需要看完整 slurm log 和 sacct ExitCode。")
    return advice


def build_analysis_report(
    jobs: dict[str, Job],
    host: str,
    user: str,
    log_dir: str,
    log_lines: int,
    extra_log_dirs: Iterable[str] | None = None,
    focus_job_ids: Iterable[str] | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    focus = set(focus_job_ids or [])
    display_jobs = {jid: job for jid, job in jobs.items() if not focus or jid in focus}
    lines = [
        "# HPC Change Analysis",
        "",
        f"- Time: {now}",
        f"- Host: {host}",
        f"- Log dir: `{log_dir}`",
        f"- Extra log dirs: `{', '.join(extra_log_dirs or []) or 'none'}`",
        "",
        "## Diagnosis",
    ]

    if not display_jobs:
        lines.append("- No tracked jobs found.")
    for job in sorted(display_jobs.values(), key=lambda j: j.id):
        try:
            log_path = get_job_log_path(host, log_dir, job.id, extra_log_dirs)
        except Exception as exc:
            log_path = f"<log path lookup failed: {exc}>"
        try:
            tail = get_log_tail(host, log_dir, job.id, log_lines, extra_log_dirs)
        except Exception as exc:
            tail = f"<log fetch failed: {exc}>"
        advice = diagnose_job(job, tail)
        reason = f"; reason={job.reason}" if job.reason else ""
        exit_code = f"; exit={job.exit_code}" if job.exit_code else ""
        lines.append("")
        lines.append(f"### {job.id} `{job.name}`")
        lines.append(f"- State: `{job.state}`; elapsed={job.elapsed}{reason}{exit_code}")
        lines.append(f"- Log: `{log_path or 'not found yet'}`")
        for item in advice:
            lines.append(f"- Recommendation: {item}")
        metric_lines = extract_metric_lines(tail)
        if metric_lines:
            lines.append("- Recent metric lines:")
            for metric_line in metric_lines:
                lines.append(f"  - `{metric_line[:220]}`")

    lines.extend(["", "## Scheduler Context", ""])
    try:
        context = get_scheduler_context(host, user, display_jobs.keys())
    except Exception as exc:
        context = f"<scheduler context failed: {exc}>"
    lines.append("```text")
    lines.append(context)
    lines.append("```")
    lines.extend(["", "## Policy", "- This reporter writes analysis only. It does not cancel, resubmit, or edit experiments automatically."])
    return "\n".join(lines).strip() + "\n"


def write_analysis_report(path: Path, report: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = path / f"hpc_change_analysis_{stamp}.md"
    report_path.write_text(report)
    latest_path = path / "latest.md"
    latest_path.write_text(report)
    return report_path


def collect_jobs(
    host: str,
    user: str,
    job_ids: list[str],
    previous_ids: Iterable[str],
    sacct_lookback_hours: int,
    include_cancelled: bool,
) -> dict[str, Job]:
    current = get_squeue_jobs(host, user)
    recent = get_recent_sacct_jobs(host, user, sacct_lookback_hours)
    tracked_ids = sorted(set(job_ids) | set(current) | set(previous_ids) | set(recent))
    historical = recent
    historical.update(get_sacct_jobs(host, tracked_ids))
    merged = historical
    merged.update(current)
    jobs = {jid: merged[jid] for jid in tracked_ids if jid in merged}
    if include_cancelled:
        return jobs
    return {jid: job for jid, job in jobs.items() if not cancelled_state(job.state)}


def build_report(
    jobs: dict[str, Job],
    prev_state: dict,
    host: str,
    log_dir: str,
    log_lines: int,
    include_logs: bool,
    extra_log_dirs: Iterable[str] | None = None,
    include_cancelled: bool = False,
) -> tuple[str, bool, bool, set[str], set[str]]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_jobs = prev_state.get("jobs", {})
    changed = False
    failed = False
    changed_ids: set[str] = set()
    failed_ids: set[str] = set()

    lines = [
        f"HPC progress report",
        f"Time: {now}",
        f"Host: {host}",
        "",
    ]

    if not jobs:
        lines.append("No tracked jobs found in squeue/sacct.")
    else:
        lines.append("Jobs:")
        for job in sorted(jobs.values(), key=lambda j: j.id):
            prev = previous_jobs.get(job.id, {})
            prev_state_name = prev.get("state")
            marker = ""
            if prev_state_name and prev_state_name != job.state:
                marker = f"  ({prev_state_name} -> {job.state})"
                changed = True
                changed_ids.add(job.id)
            elif job.id not in previous_jobs:
                marker = "  (new)"
                changed = True
                changed_ids.add(job.id)

            if job.id in changed_ids and alert_failure_state(job.state):
                failed = True
                failed_ids.add(job.id)

            reason = f", reason={job.reason}" if job.reason else ""
            exit_code = f", exit={job.exit_code}" if job.exit_code else ""
            lines.append(
                f"- {job.id} {job.name}: {job.state}, elapsed={job.elapsed}"
                f"{reason}{exit_code}{marker}"
            )

        gone_ids = sorted(set(previous_jobs) - set(jobs))
        for jid in gone_ids:
            prev = previous_jobs[jid]
            if not include_cancelled and cancelled_state(prev.get("state", "")):
                continue
            lines.append(f"- {jid} {prev.get('name', '')}: no longer visible (was {prev.get('state')})")
            changed = True

    if include_logs and jobs:
        lines.append("")
        lines.append(f"Recent logs (last {log_lines} lines when available):")
        log_jobs = [
            job for job in jobs.values()
            if job.id in changed_ids or alert_failure_state(job.state) or job.state.upper() == "RUNNING"
        ][:12]
        for job in sorted(log_jobs, key=lambda j: j.id):
            if job.state.upper() == "PENDING" and not alert_failure_state(job.state):
                continue
            try:
                tail = get_log_tail(host, log_dir, job.id, log_lines, extra_log_dirs)
            except Exception as exc:
                tail = f"<log fetch failed: {exc}>"
            if tail:
                lines.append("")
                lines.append(f"--- {job.id} {job.name} ---")
                lines.append(tail)

    all_terminal = jobs and all(terminal_state(job.state) for job in jobs.values())
    if all_terminal and not prev_state.get("all_terminal", False):
        changed = True

    return "\n".join(lines).strip() + "\n", changed, failed, changed_ids, failed_ids


def send_via_mail(to_addr: str, subject: str, body: str) -> None:
    """Hand off to the local mail system.

    On macOS this can succeed even when external SMTP delivery is not configured,
    so prefer --email-method smtp for reliable off-machine email.
    """
    subprocess.run(
        ["mail", "-s", subject, to_addr],
        input=body,
        text=True,
        check=True,
    )


def send_via_smtp(to_addr: str, subject: str, body: str) -> None:
    missing = [name for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"SMTP email requires environment variables: {', '.join(missing)}")

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("SMTP_FROM", username or to_addr)

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(msg)


def send_email(to_addr: str, subject: str, body: str, method: str) -> None:
    if method == "smtp":
        send_via_smtp(to_addr, subject, body)
    else:
        send_via_mail(to_addr, subject, body)


def post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        response.read()


def post_form(url: str, payload: dict[str, str]) -> None:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        response.read()


def send_pushdeer(subject: str, body: str) -> None:
    pushkey = os.environ.get("PUSHDEER_PUSHKEY")
    if not pushkey:
        raise RuntimeError("PushDeer requires PUSHDEER_PUSHKEY")
    post_form(
        "https://api2.pushdeer.com/message/push",
        {
            "pushkey": pushkey,
            "text": subject,
            "desp": body,
            "type": "markdown",
        },
    )


def send_serverchan(subject: str, body: str) -> None:
    sendkey = os.environ.get("SERVERCHAN_SENDKEY")
    if not sendkey:
        raise RuntimeError("ServerChan requires SERVERCHAN_SENDKEY")
    post_form(
        f"https://sctapi.ftqq.com/{sendkey}.send",
        {
            "title": subject,
            "desp": body,
        },
    )


def send_wecom(subject: str, body: str) -> None:
    webhook = os.environ.get("WECOM_WEBHOOK")
    if not webhook:
        raise RuntimeError("Enterprise WeChat webhook requires WECOM_WEBHOOK")
    post_json(
        webhook,
        {
            "msgtype": "markdown",
            "markdown": {
                "content": f"**{subject}**\n\n{body}",
            },
        },
    )


def send_bark(subject: str, body: str) -> None:
    server = os.environ.get("BARK_SERVER", "https://api.day.app").rstrip("/")
    device_key = os.environ.get("BARK_DEVICE_KEY")
    if not device_key:
        raise RuntimeError("Bark requires BARK_DEVICE_KEY")
    post_json(
        f"{server}/push",
        {
            "device_key": device_key,
            "title": subject,
            "body": body,
            "group": "hpc",
            "level": "active",
        },
    )


def send_push(channel: str, subject: str, body: str) -> None:
    if channel == "pushdeer":
        send_pushdeer(subject, body)
        return
    if channel == "serverchan":
        send_serverchan(subject, body)
        return
    if channel == "wecom":
        send_wecom(subject, body)
        return
    if channel == "bark":
        send_bark(subject, body)
        return
    raise RuntimeError(f"Unsupported push channel: {channel}")


def send_local_notification(subject: str, body: str, sound: str = "Glass") -> None:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    script = f'display notification "{esc(body)}" with title "{esc(subject)}" sound name "{esc(sound)}"'
    subprocess.run(["osascript", "-e", script], capture_output=True, check=True)


def should_email(mode: str, changed: bool, failed: bool) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode == "failure":
        return failed
    return changed


def append_local_log(path: Path, message: str) -> None:
    with path.open("a") as f:
        f.write(message)
        if not message.endswith("\n"):
            f.write("\n")


def should_push(mode: str, changed: bool, failed: bool) -> bool:
    return should_email(mode, changed, failed)


def should_local_notify(mode: str, changed: bool, failed: bool) -> bool:
    return should_email(mode, changed, failed)


def compact_notification_body(report: str, max_lines: int = 8) -> str:
    picked: list[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("Analysis report:"):
            picked.append(stripped)
        if " -> " in stripped or "  (new)" in stripped:
            picked.append(stripped)
    if not picked:
        for line in report.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                picked.append(stripped)
    return "\n".join(picked[:max_lines]) or "HPC job status changed."


def run_once(args: argparse.Namespace) -> tuple[str, bool, bool]:
    job_ids = parse_job_ids(args.job_ids)
    extra_log_dirs = parse_job_ids(args.extra_log_dirs)
    prev_state = load_state(args.state_file)
    previous_ids = prev_state.get("jobs", {}).keys()
    jobs = collect_jobs(
        args.host,
        args.user,
        job_ids,
        previous_ids,
        args.sacct_lookback_hours,
        args.include_cancelled,
    )
    report, changed, failed, changed_ids, failed_ids = build_report(
        jobs,
        prev_state,
        args.host,
        args.log_dir,
        args.log_lines,
        args.include_logs,
        extra_log_dirs,
        args.include_cancelled,
    )
    analysis_path = None
    analyzed_failures = dict(prev_state.get("analyzed_failures", {}))
    current_failure_keys = {
        jid: f"{job.state}|{job.exit_code}|{job.elapsed}"
        for jid, job in jobs.items()
        if alert_failure_state(job.state)
    }
    unanalyzed_failed_ids = {
        jid for jid, key in current_failure_keys.items()
        if analyzed_failures.get(jid) != key
    }
    focus_ids = failed_ids or unanalyzed_failed_ids or changed_ids
    if len(focus_ids) > 8:
        focus_ids = set(sorted(focus_ids, key=lambda item: int(item) if item.isdigit() else 0, reverse=True)[:8])
    should_analyze = (args.analyze_on_change and changed) or (args.analyze_on_failure and bool(unanalyzed_failed_ids))
    if should_analyze:
        analysis = build_analysis_report(
            jobs,
            args.host,
            args.user,
            args.log_dir,
            args.analysis_log_lines,
            extra_log_dirs,
            focus_job_ids=focus_ids,
        )
        analysis_path = write_analysis_report(args.analysis_dir, analysis)
        report = f"{report}\nAnalysis report: {analysis_path}\n"
        for jid in current_failure_keys:
            analyzed_failures[jid] = current_failure_keys[jid]
    save_state(args.state_file, {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "all_terminal": bool(jobs) and all(terminal_state(job.state) for job in jobs.values()),
        "reporter_error": None,
        "last_analysis_path": str(analysis_path) if analysis_path else prev_state.get("last_analysis_path"),
        "analyzed_failures": analyzed_failures,
        "jobs": {jid: job_to_dict(job) for jid, job in jobs.items()},
    })
    return report, changed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Report HPC progress and optionally send notifications.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--extra-log-dirs", default=DEFAULT_EXTRA_LOG_DIRS,
                        help="Comma-separated fallback log directories, used for old scratch jobs")
    parser.add_argument("--job-ids", default=None, help="Comma-separated job IDs to track")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--local-log", type=Path, default=DEFAULT_LOCAL_LOG)
    parser.add_argument("--log-lines", type=int, default=12)
    parser.add_argument("--sacct-lookback-hours", type=int, default=24,
                        help="Also inspect recent sacct jobs so short jobs are not missed")
    parser.add_argument("--include-cancelled", action="store_true",
                        help="Keep CANCELLED jobs in reports and notifications")
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--analyze-on-change", action="store_true",
                        help="Write a diagnosis report whenever tracked job state changes")
    parser.add_argument("--analyze-on-failure", action="store_true",
                        help="Write a diagnosis report whenever a failed job has not been analyzed yet")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--analysis-log-lines", type=int, default=80)
    parser.add_argument("--once", action="store_true", help="Run one report and exit")
    parser.add_argument("--watch", action="store_true", help="Poll forever")
    parser.add_argument("--interval", type=int, default=300, help="Watch interval in seconds")
    parser.add_argument("--email-to", default=None)
    parser.add_argument("--email-on", choices=["always", "change", "failure", "never"], default="change")
    parser.add_argument("--email-method", choices=["mail", "smtp"], default="mail")
    parser.add_argument("--notify-local-on", choices=["always", "change", "failure", "never"], default="never",
                        help="Send macOS desktop notifications")
    parser.add_argument(
        "--push-channel",
        choices=["pushdeer", "serverchan", "wecom", "bark"],
        default=None,
        help="Phone push channel. For WeChat, prefer pushdeer/serverchan or enterprise WeChat webhook.",
    )
    parser.add_argument(
        "--push-on",
        choices=["always", "change", "failure", "never"],
        default="change",
    )
    parser.add_argument("--subject-prefix", default="[Medical-SAM3 HPC]")
    args = parser.parse_args()

    if not args.once and not args.watch:
        args.once = True

    def tick() -> None:
        report, changed, failed = run_once(args)
        print(report)
        append_local_log(args.local_log, report)
        if args.email_to and should_email(args.email_on, changed, failed):
            subject_state = "failure" if failed else "update" if changed else "status"
            subject = f"{args.subject_prefix} {subject_state} {datetime.now():%Y-%m-%d %H:%M}"
            send_email(args.email_to, subject, report, args.email_method)
            if args.email_method == "mail":
                print(f"Email handed to local mail system for {args.email_to}")
            else:
                print(f"SMTP email sent to {args.email_to}")
        if args.push_channel and should_push(args.push_on, changed, failed):
            subject_state = "failure" if failed else "update" if changed else "status"
            subject = f"{args.subject_prefix} {subject_state} {datetime.now():%m-%d %H:%M}"
            send_push(args.push_channel, subject, report)
            print(f"Push sent via {args.push_channel}")
        if should_local_notify(args.notify_local_on, changed, failed):
            subject_state = "failure" if failed else "update" if changed else "status"
            subject = f"{args.subject_prefix} {subject_state}"
            send_local_notification(
                subject,
                compact_notification_body(report),
                sound="Basso" if failed else "Glass",
            )
            print("Local notification sent")

    if args.once:
        tick()
        return

    while True:
        try:
            tick()
        except Exception as exc:
            msg = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] reporter error: {exc}"
            error_changed = save_reporter_error(args.state_file, str(exc))
            print(msg, file=sys.stderr)
            append_local_log(args.local_log, msg)
            if args.email_to and args.email_on in ("always", "failure"):
                send_email(args.email_to, f"{args.subject_prefix} reporter error", msg, args.email_method)
            if args.push_channel and args.push_on in ("always", "failure"):
                send_push(args.push_channel, f"{args.subject_prefix} reporter error", msg)
            should_notify_error = args.notify_local_on in ("always", "failure") or (
                args.notify_local_on == "change" and error_changed
            )
            if should_notify_error:
                send_local_notification(f"{args.subject_prefix} reporter error", msg, sound="Basso")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
