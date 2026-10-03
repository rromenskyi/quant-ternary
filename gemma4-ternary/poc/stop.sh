#!/usr/bin/env bash
# Graceful stop: training saves a checkpoint after its current optimizer
# step and exits; the teacher/data steps are safe to kill at any point
# (they resume per shard). Then rerun ternary_pipeline.sh to continue.
if pkill -TERM -f '[t]rain_ternary.py'; then
  echo "train: SIGTERM sent, it saves and exits (watch run/status.json)"
elif pkill -TERM -f '[t]eacher_topk.py|[p]repare_data.py'; then
  echo "teacher/data stopped; completed shards are kept"
else
  echo "nothing running"
fi
