#!/usr/bin/env python3
"""
hpc_notify.py — HPC job 状态变化时发送 macOS 系统通知

用法:
  python hpc_notify.py &        # 后台运行，关掉终端也继续
  python hpc_notify.py          # 前台运行，Ctrl+C 停止

状态变化时会弹出系统通知:
  PENDING → RUNNING  (job 开始跑了)
  RUNNING → 消失     (job 完成/失败)
  新 job 出现        (有新任务进队列)
"""

import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path

HOST = "bouchet"
LOG_DIR = "/home/hw646/Medical-SAM3/output/visium_hd_exp1"
POLL_INTERVAL = 30  # 秒

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(Path.home() / ".hpc_notify.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def notify(title: str, message: str, subtitle: str = "") -> None:
    """发送 macOS 系统通知。"""
    subtitle_part = f'subtitle "{subtitle}"' if subtitle else ""
    script = f'display notification "{message}" with title "{title}" {subtitle_part} sound name "Glass"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def ssh(cmd: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", HOST, cmd],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def get_jobs() -> dict[str, dict]:
    out = ssh("squeue -u hw646 --format='%.10i|%.18j|%.8T|%.12M|%R' --noheader 2>/dev/null")
    jobs = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        jid = parts[0].strip()
        jobs[jid] = {
            "id":     jid,
            "name":   parts[1].strip(),
            "state":  parts[2].strip(),
            "time":   parts[3].strip(),
            "reason": parts[4].strip(),
        }
    return jobs


def get_log_tail(job_id: str, n: int = 3) -> str:
    out = ssh(
        f"find {LOG_DIR} -name '*{job_id}*' 2>/dev/null | head -1 "
        f"| xargs -I{{}} tail -{n} {{}} 2>/dev/null"
    )
    return out.strip()


def main() -> None:
    log.info("HPC 通知启动，每 %ds 轮询一次 → %s", POLL_INTERVAL, HOST)
    notify("HPC 监控已启动", f"每 {POLL_INTERVAL}s 轮询 {HOST}", "hpc_notify")

    prev: dict[str, dict] = {}
    ssh_error_count = 0

    while True:
        try:
            current = get_jobs()
            ssh_error_count = 0
        except Exception as e:
            ssh_error_count += 1
            if ssh_error_count == 3:
                notify("HPC 监控 ⚠️", f"SSH 连接失败: {e}", "连续失败 3 次")
            log.warning("SSH 失败: %s", e)
            time.sleep(POLL_INTERVAL)
            continue

        # 新 job 出现
        for jid, job in current.items():
            if jid not in prev:
                log.info("新 job: %s (%s) → %s", job["name"], jid, job["state"])
                notify(
                    "新 Job 进队列",
                    f"{job['name']}  状态: {job['state']}",
                    f"Job ID: {jid}",
                )

        # 状态变化
        for jid, job in current.items():
            if jid in prev and prev[jid]["state"] != job["state"]:
                old, new = prev[jid]["state"], job["state"]
                log.info("状态变化: %s (%s)  %s → %s", job["name"], jid, old, new)
                if new == "RUNNING":
                    notify(
                        f"▶ Job 开始运行",
                        f"{job['name']}",
                        f"Job ID: {jid}",
                    )
                elif new in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY"):
                    tail = get_log_tail(jid)
                    msg = tail.splitlines()[-1] if tail else "无日志"
                    notify(
                        f"✗ Job 失败 ({new})",
                        f"{job['name']}\n{msg}",
                        f"Job ID: {jid}",
                    )
                    log.error("Job 失败 %s (%s): %s\n%s", job["name"], jid, new, tail)

        # job 消失（完成或取消）
        for jid, job in prev.items():
            if jid not in current and job["state"] in ("RUNNING", "PENDING"):
                tail = get_log_tail(jid)
                last_line = tail.splitlines()[-1] if tail else "无日志"
                log.info("Job 结束: %s (%s)\n%s", job["name"], jid, tail)
                notify(
                    f"✓ Job 完成",
                    f"{job['name']}\n{last_line}",
                    f"Job ID: {jid}",
                )

        prev = current
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
