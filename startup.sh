#!/usr/bin/env bash
set -e
python - <<'PY'
import os
# 用 openmim 下载 config+权重
os.system("python -m pip install -U openmim")
os.system("mim download mmsegmentation --config segformer_mit-b0_8xb2-160k_ade20k-512x512 --dest .")
PY
exec python app.py
