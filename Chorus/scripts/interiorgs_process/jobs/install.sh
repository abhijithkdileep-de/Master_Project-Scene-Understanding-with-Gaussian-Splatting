#!/bin/bash
set -euo pipefail

python -m pip install \
  gsplat==1.4.0 \
  opencv-python \
  scikit-image \
  scikit-learn \
  matplotlib \
  plyfile \
  open-clip-torch \
  miniball \
  shapely
