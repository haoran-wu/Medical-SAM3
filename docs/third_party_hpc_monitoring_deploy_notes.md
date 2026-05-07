# Third-Party HPC Monitoring Deployment Notes

Date: 2026-05-06

## Summary

Two third-party projects were evaluated for direct deployment:

1. `rackslab/Slurm-web`
2. `neilmunday/slurm-mail`

Both repositories were cloned locally for inspection:

- `tmp/Slurm-web`
- `tmp/slurm-mail`

Neither can be fully deployed as a useful production integration from this local
workspace alone, because both require cluster-side Slurm services or admin-level
configuration on Bouchet.

## Slurm-web

Repository:

- <https://github.com/rackslab/Slurm-web>

What it needs:

- A working Slurm REST backend (`slurmrestd`)
- Slurm-web agent/gateway services
- Python dependencies
- Frontend dependencies (`npm ci`)
- Cluster-side configuration so the agent can authenticate to `slurmrestd`

Important finding:

- Slurm-web is not a simple wrapper around `ssh + squeue`.
- Its docs and config are centered on `slurmrestd`.
- Without `slurmrestd` access on Bouchet, a local frontend deployment would not
  have live data to show.

Relevant local sources:

- `tmp/Slurm-web/README.md`
- `tmp/Slurm-web/docs/modules/install/pages/install/sources.adoc`
- `tmp/Slurm-web/docs/modules/overview/pages/architecture.adoc`

Developer-mode local startup from source:

```bash
cd tmp/Slurm-web
pip install -e .
cd frontend && npm ci
slurm-web-agent
slurm-web-gateway
cd frontend && npm run dev
```

Why this is not enough here:

- This still expects a real or emulated Slurm REST backend.
- Bouchet-side `slurmrestd` availability and auth are not confirmed from the
  current user-level access.
- If Bouchet does not expose `slurmrestd`, the app cannot operate as intended.

## Slurm-mail

Repository:

- <https://github.com/neilmunday/slurm-mail>

What it needs:

- Cluster-side installation on a Slurm host
- `cron`
- `logrotate`
- Python 3.8+
- Slurm 22/23/24/25
- A working SMTP server
- Update of `MailProg` in `slurm.conf`
- Restart of `slurmctld`

Important finding:

- Slurm-mail is a drop-in replacement for Slurm mail handling.
- It is not a local per-user notification helper.
- It must be installed where the Slurm controller can invoke it.

Relevant local source:

- `tmp/slurm-mail/README.md`

Critical cluster-side steps from the project docs:

```bash
python setup.py install
cp etc/logrotate.d/slurm-mail /etc/logrotate.d/
cp etc/cron.d/slurm-mail /etc/cron.d/
install -d -m 700 -o slurm -g slurm /var/log/slurm-mail
```

Then:

```bash
MailProg=/usr/bin/slurm-spool-mail
systemctl restart slurmctld
```

Why this is not enough here:

- These steps require root/admin privileges on the Slurm controller host.
- A normal user on Bouchet cannot safely or correctly perform them.
- It also requires SMTP configuration and controller-level operational approval.

## Recommended Path

For this project, the most practical path remains:

1. Keep the local `hpc_dashboard.py` workflow.
2. Continue improving the dashboard UX using Slurm-web as inspiration.
3. Add user-level notifications directly in this repo, rather than trying to
   replace Slurm controller mail handling.

## Most Useful Next Step

Implement one of these directly in the current local dashboard stack:

1. New-failure alert banner with deduping
2. Email alert on first `FAILED/TIMEOUT/OUT_OF_MEMORY/NODE_FAIL`
3. Slack/webhook alert on first abnormal exit
4. “Needs attention since last refresh” section

This avoids cluster-admin dependencies while still giving the practical benefit
the two third-party tools are aiming for.
