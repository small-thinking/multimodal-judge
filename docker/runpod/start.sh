#!/bin/sh
set -eu
mkdir -p /run/sshd /root/.ssh /workspace/cache/huggingface /workspace/artifacts
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi
if [ -f /root/.ssh/authorized_keys ]; then
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
printf '%s\n' 'RunPod ready. Training must be started explicitly; outputs belong under /workspace.'
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password
