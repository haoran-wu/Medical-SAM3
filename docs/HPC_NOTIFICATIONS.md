# HPC Notifications

`hpc_progress_reporter.py` can watch SLURM jobs on `bouchet` and send updates by email or phone push.

## Fastest way

If you want the current recommended setup for this repo, use:

```bash
bash scripts/start_hpc_watch.sh
```

This starts:

- the local dashboard on `http://127.0.0.1:8765`
- the reporter in watch mode
- local macOS notifications on state changes by default
- failure analysis reports in `output/hpc_monitoring/analysis`

Default behavior:

- Dashboard refreshes every 20 seconds
- Reporter polls every 120 seconds unless `HPC_NOTIFY_INTERVAL` is set
- Email and phone push are optional and only enabled if you export the matching env vars

Recommended minimal env file:

```bash
mkdir -p ~/.config
cat > ~/.config/hpc-notify.env <<'EOF'
export HPC_NOTIFY_INTERVAL="120"
export HPC_NOTIFY_LOCAL_ON="change"
export HPC_NOTIFY_EMAIL_ON="failure"
export HPC_NOTIFY_PUSH_ON="failure"
EOF
```

## Supported channels

- Email via local `mail`
- Email via SMTP
- `PushDeer` for WeChat-friendly phone push
- `Server酱`
- `Enterprise WeChat` robot webhook
- `Bark`

## Recommended setup

For personal iPhone + WeChat, try `PushDeer` first.

1. Install the PushDeer app and bind your WeChat delivery path.
2. Export your key:

```bash
export PUSHDEER_PUSHKEY="PDUxxxxxxxxxxxxxxxx"
```

3. Run:

```bash
python hpc_progress_reporter.py \
  --watch \
  --interval 120 \
  --job-ids 12345678,12345679 \
  --push-channel pushdeer \
  --push-on change
```

## Email via Gmail SMTP

```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your_email@gmail.com"
export SMTP_FROM="your_email@gmail.com"
export SMTP_PASS="your_app_password"
```

Then:

```bash
python hpc_progress_reporter.py \
  --watch \
  --interval 120 \
  --job-ids 12345678 \
  --email-to your_email@gmail.com \
  --email-method smtp \
  --email-on change
```

## Server酱

```bash
export SERVERCHAN_SENDKEY="SCTxxxxxxxxxxxxxxxx"
python hpc_progress_reporter.py --watch --job-ids 12345678 --push-channel serverchan
```

## Enterprise WeChat webhook

```bash
export WECOM_WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"
python hpc_progress_reporter.py --watch --job-ids 12345678 --push-channel wecom
```

## Notes

- `--push-on change` means notify on new job or state change.
- `--push-on failure` means only notify on failed terminal states.
- If you omit `--job-ids`, the script tracks current `squeue` jobs for user `hw646`.
- Reports are cached in `~/.hpc_progress_reporter_state.json`.
- Failure analysis reports are written to `output/hpc_monitoring/analysis`.
- The helper launcher `scripts/start_hpc_watch.sh` is the easiest way to run the full stack.

## All jobs, always on

Use `scripts/hpc_progress_reporter_launch.sh` with macOS `launchd` if you want background monitoring for all jobs under user `hw646`.

1. Create `~/.config/hpc-notify.env`
2. Put these lines in it:

```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="vettel.hwu@gmail.com"
export SMTP_FROM="vettel.hwu@gmail.com"
export SMTP_PASS="your_16_char_app_password"
export HPC_NOTIFY_EMAIL="vettel.hwu@gmail.com"
export HPC_NOTIFY_INTERVAL="120"
export HPC_NOTIFY_EMAIL_ON="change"
```

3. Install `scripts/com.haoranwu.hpc-notify.plist.template` as `~/Library/LaunchAgents/com.haoranwu.hpc-notify.plist`
4. Load it:

```bash
launchctl unload ~/Library/LaunchAgents/com.haoranwu.hpc-notify.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.haoranwu.hpc-notify.plist
launchctl start com.haoranwu.hpc-notify
```

This mode does not set `--job-ids`, so it monitors all visible jobs for your HPC account.
