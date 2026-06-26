#!/usr/bin/env python3
"""
hpc.py — HPC 任务管理入口

子命令:
  python hpc.py status          查看当前 job 状态（一次性）
  python hpc.py start           后台启动通知守护进程
  python hpc.py stop            停止守护进程
  python hpc.py logs            实时跟踪守护进程日志
  python hpc.py watch           持续刷新 job 状态（每 30s）
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HOST        = "bouchet"
LOG_DIR     = "/home/hw646/Medical-SAM3/output/visium_hd_exp1"
LOG_FILE    = Path.home() / ".hpc_notify.log"
PID_FILE    = Path.home() / ".hpc_notify.pid"
THIS_FILE   = Path(__file__).resolve()
SSH_READY   = False

RESET  = "\033[0m";  BOLD   = "\033[1m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"
RED    = "\033[91m"; CYAN   = "\033[96m"
GRAY   = "\033[90m"; BLUE   = "\033[94m"


# ─── SSH helpers ──────────────────────────────────────────────────────────────

def ensure_ssh_ready() -> None:
    global SSH_READY
    if SSH_READY:
        return
    check_script = THIS_FILE.parents[1] / "hpc_ssh_check.sh"
    if check_script.exists():
        subprocess.run(["bash", str(check_script)], check=True)
    SSH_READY = True


def ssh(cmd: str, timeout: int = 10) -> str:
    ensure_ssh_ready()
    r = subprocess.run(
        ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes", HOST, cmd],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


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


def get_log_tail(job_id: str, n: int = 5) -> str:
    return ssh(
        f"find {LOG_DIR} -name '*{job_id}*' 2>/dev/null | head -1 "
        f"| xargs -I{{}} tail -{n} {{}} 2>/dev/null"
    )


# ─── Display ──────────────────────────────────────────────────────────────────

def state_color(state: str) -> str:
    if state == "RUNNING":  return GREEN  + "● RUNNING"  + RESET
    if state == "PENDING":  return YELLOW + "◌ PENDING"  + RESET
    return RED + f"✗ {state}" + RESET


def print_jobs(jobs: list[dict], show_logs: bool = True) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{BOLD}{'─'*62}{RESET}")
    print(f"  {BOLD}HPC Jobs{RESET}  {CYAN}{now}{RESET}")
    print(f"{BOLD}{'─'*62}{RESET}")

    if not jobs:
        print(f"  {GRAY}队列里没有 job（全部完成或没有提交任务）{RESET}\n")
        return

    for j in jobs:
        reason = f"  {GRAY}({j['reason']}){RESET}" if j["reason"] not in ("None", "") else ""
        print(f"  {BOLD}{j['id']}{RESET}  {j['name']:<22} {state_color(j['state'])}  {j['time']}{reason}")
        if show_logs and j["state"] == "RUNNING":
            tail = get_log_tail(j["id"])
            if tail:
                for line in tail.splitlines()[-4:]:
                    print(f"  {GRAY}    {line}{RESET}")
    print()


# ─── Daemon (通知守护进程) ──────────────────────────────────────────────────────

def notify(title: str, message: str, subtitle: str = "") -> None:
    sub = f'subtitle "{subtitle}"' if subtitle else ""
    script = f'display notification "{message}" with title "{title}" {sub} sound name "Glass"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def daemon_loop(poll: int = 30) -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info("守护进程启动（PID %d），每 %ds 轮询", os.getpid(), poll)
    notify("HPC 监控已启动", f"每 {poll}s 轮询 {HOST}", "hpc.py")

    prev: dict[str, dict] = {}
    errors = 0

    while True:
        try:
            current = {j["id"]: j for j in get_jobs()}
            errors = 0
        except Exception as e:
            errors += 1
            if errors == 3:
                notify("HPC 监控 ⚠️", f"SSH 连续失败: {e}")
            log.warning("SSH 失败: %s", e)
            time.sleep(poll)
            continue

        for jid, j in current.items():
            if jid not in prev:
                log.info("新 job: %s (%s) %s", j["name"], jid, j["state"])
                notify("新 Job 进队列", f"{j['name']}  {j['state']}", f"ID {jid}")

        for jid, j in current.items():
            if jid in prev and prev[jid]["state"] != j["state"]:
                old, new = prev[jid]["state"], j["state"]
                log.info("状态变化: %s (%s)  %s → %s", j["name"], jid, old, new)
                if new == "RUNNING":
                    notify("▶ Job 开始运行", j["name"], f"ID {jid}")
                elif new in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY"):
                    tail = get_log_tail(jid, 2)
                    last = tail.splitlines()[-1] if tail else "无日志"
                    notify(f"✗ Job 失败 ({new})", f"{j['name']}\n{last}", f"ID {jid}")
                    log.error("失败: %s (%s) %s | %s", j["name"], jid, new, last)

        for jid, j in prev.items():
            if jid not in current and j["state"] in ("RUNNING", "PENDING"):
                tail = get_log_tail(jid, 3)
                last = tail.splitlines()[-1] if tail else "无日志"
                log.info("Job 结束: %s (%s) | %s", j["name"], jid, last)
                notify("✓ Job 完成", f"{j['name']}\n{last}", f"ID {jid}")

        prev = current
        time.sleep(poll)


# ─── 子命令 ───────────────────────────────────────────────────────────────────

def cmd_status(_args) -> None:
    print(f"  {GRAY}连接 {HOST}...{RESET}", end="", flush=True)
    try:
        jobs = get_jobs()
        print("\r" + " " * 30 + "\r", end="")
        print_jobs(jobs, show_logs=True)
    except Exception as e:
        print(f"\n  {RED}SSH 失败: {e}{RESET}\n")

    # 守护进程状态
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"  {GREEN}通知守护进程正在运行{RESET}  PID {pid}  日志→ {LOG_FILE}\n")
        except ProcessLookupError:
            print(f"  {YELLOW}守护进程 PID {pid} 已退出（PID 文件残留）{RESET}\n")
            PID_FILE.unlink(missing_ok=True)
    else:
        print(f"  {GRAY}通知守护进程未运行。用 python hpc.py start 启动。{RESET}\n")


def cmd_start(args) -> None:
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"{YELLOW}守护进程已在运行（PID {pid}）{RESET}")
            return
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)

    proc = subprocess.Popen(
        [sys.executable, str(THIS_FILE), "_daemon", "--interval", str(args.interval)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"{GREEN}守护进程已启动{RESET}  PID {proc.pid}")
    print(f"  日志: {LOG_FILE}")
    print(f"  查看状态: python hpc.py status")
    print(f"  实时日志: python hpc.py logs")


def cmd_stop(_args) -> None:
    if not PID_FILE.exists():
        print(f"{GRAY}守护进程未运行{RESET}")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        print(f"{GREEN}守护进程已停止（PID {pid}）{RESET}")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print(f"{GRAY}进程 {pid} 已不存在（PID 文件已清理）{RESET}")


def cmd_logs(_args) -> None:
    if not LOG_FILE.exists():
        print(f"{GRAY}日志文件不存在: {LOG_FILE}{RESET}")
        return
    print(f"{GRAY}实时日志（Ctrl+C 退出）→ {LOG_FILE}{RESET}\n")
    try:
        subprocess.run(["tail", "-f", "-n", "40", str(LOG_FILE)])
    except KeyboardInterrupt:
        pass


def cmd_watch(args) -> None:
    prev: set[str] = set()
    while True:
        try:
            jobs = get_jobs()
            os.system("clear")
            print_jobs(jobs, show_logs=True)
            print(f"  {GRAY}每 {args.interval}s 刷新（Ctrl+C 退出）{RESET}")
            cur_ids = {j["id"] for j in jobs}
            gone = prev - cur_ids
            for jid in gone:
                print(f"  {GREEN}✓ Job {jid} 已完成{RESET}")
            prev = cur_ids
        except Exception as e:
            print(f"{RED}SSH 失败: {e}{RESET}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n退出。")
            break


# ─── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="hpc.py", description="HPC 任务管理")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status",  help="查看当前 job 状态和守护进程状态")

    p_start = sub.add_parser("start", help="后台启动通知守护进程")
    p_start.add_argument("--interval", type=int, default=30, help="轮询间隔（秒）")

    sub.add_parser("stop",  help="停止守护进程")
    sub.add_parser("logs",  help="实时查看守护进程日志")

    p_watch = sub.add_parser("watch", help="持续刷新 job 状态")
    p_watch.add_argument("--interval", type=int, default=30)

    # 内部子命令（守护进程自身调用）
    p_daemon = sub.add_parser("_daemon")
    p_daemon.add_argument("--interval", type=int, default=30)

    args = parser.parse_args()

    if args.cmd == "status":   cmd_status(args)
    elif args.cmd == "start":  cmd_start(args)
    elif args.cmd == "stop":   cmd_stop(args)
    elif args.cmd == "logs":   cmd_logs(args)
    elif args.cmd == "watch":  cmd_watch(args)
    elif args.cmd == "_daemon":
        daemon_loop(args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
