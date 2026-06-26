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

Codex/agent 规则：

- 不要因为 BatchMode SSH 失败就停下来等用户；先运行 `bash scripts/hpc_ssh_check.sh` 自动检查/建立 SSH ControlMaster。
- 如果脚本需要弹出 Duo/Terminal 或访问 `~/.ssh/controlmasters`，在 Codex sandbox 中必须用非 sandbox 权限执行；否则脚本内部的 `ssh` 可能会被本地 sandbox 拦住，即使单独的 `ssh bouchet` 已经可用。
- 如果 `scripts/hpc_ssh_check.sh` 已经建立 ControlMaster，再继续提交/同步/监控脚本。
- 只有当 Duo 账号状态异常、认证超时多次、或管理员权限问题时，才向 Haoran 报告 blocker。

HPC 主机名: `bouchet`  
用户名: `hw646`  
项目路径: `~/Medical-SAM3/`

## Medical-SAM3 实验设计规则

这个项目里做训练、backbone 选择、loss 设计、评价指标、队列/资源取舍时，不要凭直觉直接拍方案。

默认流程：
1. 先查相关文献和已有 benchmark，优先看和当前任务最接近的病理 foundation model、spatial transcriptomics、cross-modal retrieval、multi-task learning 文献。
2. 把文献依据和当前数据/任务约束对应起来，再给出推荐方案。
3. 如果存在几个合理选择，不要只选一个；优先设计 ablation 或轻量 smoke/eval，把选择转成可验证实验。
4. 每次新实验要保存关键参数、loss 曲线、metrics、日志和结果路径，最终结果写到 project 对应目录，不写到 home 作为最终输出。
5. 对失败任务要先读日志、判断原因，再决定是否安全重跑或改脚本；不要只报告失败。

当前 Medical-SAM3/Visium HD Exp1 的主结果目录：

```bash
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1
```
