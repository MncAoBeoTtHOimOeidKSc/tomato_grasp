#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH PYTHONNOUSERSITE
tomato_grasp_args=("$@")
set --
source /opt/ros/melodic/setup.bash
source ~/piper_ws/devel/setup.bash --extend
export PYTHONPATH="$HOME/librealsense-2.50.0/build/wrappers/python${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 "$root/main.py" "${tomato_grasp_args[@]}"
