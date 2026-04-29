# Claude Code Instructions

## HPC SSH 规则

**每次执行涉及 `bouchet` 的 SSH 命令之前，必须先运行：**

```bash
bash scripts/hpc_ssh_check.sh
```

这个脚本会：
1. 检查 ControlMaster 是否活着
2. 如果断了，自动弹出 Terminal 请用户做一次 DUO 认证
3. 等连接建立后再继续

不要跳过这一步，否则 SSH 命令会因为 DUO 认证失败而报错。

HPC 主机名: `bouchet`  
用户名: `hw646`  
项目路径: `~/Medical-SAM3/`
