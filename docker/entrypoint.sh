#!/usr/bin/env bash
set -e

export PYTHONPATH="/workspace/reasonsql:/opt/verl:${PYTHONPATH:-}"
exec "$@"

