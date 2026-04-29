#!/bin/bash
# hpc_ssh_check.sh
# 检查 bouchet ControlMaster 是否存活。
# 如果死了：弹出通知 + 自动打开 Terminal 让用户做一次 DUO，然后等待连接建立。
# 用法: source scripts/hpc_ssh_check.sh   (或直接被 Claude 在 Bash 工具里调用)

HOST="bouchet"
MAX_WAIT=120   # 最多等 120 秒让用户完成 DUO
POLL=2

check_alive() {
    ssh -O check "$HOST" 2>/dev/null
}

if check_alive; then
    exit 0
fi

# ControlMaster 不存在或已过期
echo "ControlMaster 已断开，正在请求重新认证..."

# macOS 通知
osascript -e 'display notification "请在弹出的终端里完成 DUO 认证" with title "HPC 需要重新连接" subtitle "ssh bouchet" sound name "Ping"' 2>/dev/null

# 自动打开 Terminal 并运行 ssh bouchet
osascript <<'EOF'
tell application "Terminal"
    activate
    do script "echo '👋 请完成 DUO 认证后关闭此窗口' && ssh bouchet"
end tell
EOF

# 等待 ControlMaster 建立
elapsed=0
while ! check_alive; do
    sleep $POLL
    elapsed=$((elapsed + POLL))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "ERROR: 等待 DUO 认证超时（${MAX_WAIT}s）"
        exit 1
    fi
done

echo "ControlMaster 已建立，继续执行。"
exit 0
