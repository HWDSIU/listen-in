#!/bin/bash
# Cluster-scoped init script for classic/dedicated Databricks compute.
# Installs ffmpeg, which listen-in needs to record streams.
# Store it in a Unity Catalog Volume or workspace file and reference it under
# Compute -> Advanced options -> Init scripts.
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends ffmpeg
