# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r jaxnerf/requirements.txt
pip uninstall jax
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m jaxnerf.train \
  --data_dir=/mnt/data/NeRF_Data/nerf_synthetic/lego \
  --train_dir=test_output \
  --max_steps=5 \
  --factor=2 \
  --batch_size=512 \
  --config=configs/orig_nerf_tpu_vm_test \
  --precompute_pkl_path /mnt/data/NeRF_Data/nerf_synthetic/lego/clip_cache_train_factor4_float32.pkl
