#!/usr/bin/env python3
"""
watch_jobs.py — 监控 HPC 上的 SLURM job 状态

用法:
  python watch_jobs.py                  # 每 30 秒刷新一次
  python watch_jobs.py --interval 60   # 每 60 秒刷新一次
  python watch_jobs.py --once          # 只看一次，不循环
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

HOST = "bouchet"
LOG_DIR = "/home/hw646/Medical-SAM3/output/visium_hd_exp1"
LOG_TAIL_LINES = 6
SSH_READY = False

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"


def ssh(cmd: str) -> str:
    global SSH_READY
    if not SSH_READY:
        check_script = Path(__file__).resolve().parent / "scripts" / "hpc_ssh_check.sh"
        if check_script.exists():
            subprocess.run(["bash", str(check_script)], check=True)
        SSH_READY = True
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", HOST, cmd],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def get_jobs() -> list[dict]:
    out = ssh("squeue -u hw646 --format='%.10i|%.18j|%.8T|%.12M|%R' --noheader 2>/dev/null")
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        jobs.append({
            "id":     parts[0].strip(),
            "name":   parts[1].strip(),
            "state":  parts[2].strip(),
            "time":   parts[3].strip(),
            "reason": parts[4].strip(),
        })
    return jobs


def get_log_tail(job_id: str, job_name: str) -> str:
    # Try common log name patterns
    patterns = [
        f"{LOG_DIR}/{job_name}_{job_id}.log",
        f"{LOG_DIR}/cls_{job_id}.log",
        f"{LOG_DIR}/train_{job_id}.log",
        f"{LOG_DIR}/stage3_{job_id}.log",
        f"{LOG_DIR}/exp1_mp_dual_{job_id}.log",
    ]
    cmd = " || ".join(f"tail -{LOG_TAIL_LINES} {p} 2>/dev/null" for p in patterns)
    return ssh(cmd)


def state_color(state: str) -> str:
    if state == "RUNNING":
        return GREEN + state + RESET
    if state == "PENDING":
        return YELLOW + state + RESET
    if state in ("FAILED", "CANCELLED", "TIMEOUT"):
        return RED + state + RESET
    return state


def print_status(jobs: list[dict], prev_ids: set[str]) -> set[str]:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}HPC Job Monitor  {CYAN}{now}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")

    current_ids = {j["id"] for j in jobs}

    if not jobs:
        print(f"{GRAY}  队列里没有你的 job（所有任务可能已完成）{RESET}")
    else:
        for j in jobs:
            state_str = state_color(j["state"])
            reason = f"  {GRAY}({j['reason']}){RESET}" if j["reason"] not in ("None", "") else ""
            print(f"  {BOLD}{j['id']}{RESET}  {j['name']:<20}  {state_str}  运行时间: {j['time']}{reason}")

            if j["state"] == "RUNNING":
                tail = get_log_tail(j["id"], j["name"])
                if tail:
                    print(f"{GRAY}    ┌─ 最新日志 ──────────────────────────────{RESET}")
                    for line in tail.splitlines():
                        print(f"{GRAY}    │ {line}{RESET}")
                    print(f"{GRAY}    └────────────────────────────────────────{RESET}")

    # 检测刚刚消失的 job（完成或失败）
    finished = prev_ids - current_ids
    for fid in finished:
        print(f"\n{GREEN}✓ Job {fid} 已离开队列（完成/取消/失败）{RESET}")
        # 尝试找日志
        tail = ssh(f"find {LOG_DIR} -name '*{fid}*' 2>/dev/null | head -1 | xargs tail -{LOG_TAIL_LINES} 2>/dev/null")
        if tail:
            print(f"{GRAY}  最后几行日志:{RESET}")
            for line in tail.splitlines():
                print(f"{GRAY}    {line}{RESET}")

    return current_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="刷新间隔（秒），默认 30")
    parser.add_argument("--once", action="store_true", help="只看一次")
    args = parser.parse_args()

    prev_ids: set[str] = set()

    while True:
        try:
            jobs = get_jobs()
            prev_ids = print_status(jobs, prev_ids)
        except Exception as e:
            print(f"{RED}SSH 连接失败: {e}{RESET}")

        if args.once:
            break

        print(f"\n{GRAY}下次刷新：{args.interval} 秒后（Ctrl+C 退出）{RESET}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
