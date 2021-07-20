# Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis Implementation

<img width="586" alt="스크린샷 2021-07-04 오후 4 11 51" src="https://user-images.githubusercontent.com/77657524/124376591-b312b780-dce2-11eb-80ad-9129d6f5eedb.png">

the Pytorch, JAX/Flax based code implementation of this paper : https://arxiv.org/abs/2104.00677
The model gives the 3D neural scene representation (NeRF: Neural Radiances Field) estimated from a few images. 
Which is based on extracting the semantic information using a pre-trained visual encoder such as CLIP, a Vision Transformer

Our Project is started in the HuggingFace X GoogleAI (JAX) Community Week Event.
https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104

## DietNeRF v.s. NeRF 
DietNeRF has a strong capacity to generalise on novel and challenging views with EXTREMELY SMALL TRAINING SAMPLES!  
The animations below shows the performance difference between DietNeRF (left) v.s. NeRF (right) with only 4 training images: 

#### SHIP
![Text](./assets/ship-dietnerf.gif) ![Alt Text](./assets/ship-nerf.gif)

#### LEGO
![Text](./assets/ship-dietnerf.gif) ![Alt Text](./assets/ship-nerf.gif)

#### HOTDOG
![Text](./assets/ship-dietnerf.gif) ![Alt Text](./assets/ship-nerf.gif)


##  Hugging Face Hub Repo URL: 

We will also upload our project on the Hugging Face Hub Repository. 
[https://huggingface.co/flax-community/putting-nerf-on-a-diet/](https://huggingface.co/flax-community/putting-nerf-on-a-diet/)

Our JAX/Flax implementation currently supports:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"><span style="font-weight:bold">Platform</span></th>
    <th class="tg-0lax" colspan="2"><span style="font-weight:bold">Single-Host GPU</span></th>
    <th class="tg-0lax" colspan="2"><span style="font-weight:bold">Multi-Device TPU</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">Type</span></td>
    <td class="tg-0lax">Single-Device</td>
    <td class="tg-0lax">Multi-Device</td>
    <td class="tg-0lax">Single-Host</td>
    <td class="tg-0lax">Multi-Host</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">Training</span></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">Evaluation</span></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
    <td class="tg-0lax"><img src="http://storage.googleapis.com/gresearch/jaxnerf/check.png" alt="Supported" width=18px height=18px></td>
  </tr>
</tbody>
</table>

## Installation

```bash
# Clone the repo
svn export https://github.com/google-research/google-research/trunk/jaxnerf
# Create a conda environment, note you can use python 3.6-3.8 as
# one of the dependencies (TensorFlow) hasn't supported python 3.9 yet.
conda create --name jaxnerf python=3.6.12; conda activate jaxnerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
pip install -r jaxnerf/requirements.txt
# [Optional] Install GPU and TPU support for Jax
# Remember to change cuda101 to your CUDA version, e.g. cuda110 for CUDA 11.0.
pip install --upgrade jax jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# install flax and flax-transformer
pip install flax transformer[flax]
```

## Dataset
Download the datasets from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download the `nerf_synthetic.zip` and unzip them
in the place you like. Let's assume they are placed under `/tmp/jaxnerf/data/`.

## How to use
```
python -m train \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \ % e.g., nerf_synthetic/lego
  --train_dir=/PATH/TO/THE/PLACE/YOU/WANT/TO/SAVE/CHECKPOINTS \
  --config=configs/CONFIG_YOU_LIKE
```
You can toggle the semantic loss by “use_semantic_loss” in configuration files.

## Rendered examples by 8-shot learned Diet-NeRF
- Lego
- Chair

## Rendered examples by occluded 14-shot learned NeRF and Diet-NeRF
This result is on the quite initial state and expected to be improved.

### Training poses
<img width="1400" src="https://user-images.githubusercontent.com/26036843/126111980-4f332c87-a7f0-42e0-a355-8e77621bbca4.png">

### Rendered novel poses
<img width="800" src="https://user-images.githubusercontent.com/26036843/126113080-a6a48f3d-2629-4efc-a740-fe908ca6b5c3.png">


## Demo
[https://huggingface.co/spaces/flax-community/DietNerf-Demo](https://huggingface.co/spaces/flax-community/DietNerf-Demo)

## Our Teams

| Teams            | Members                                                                                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NeRF Team        | Leader : [JaeYoung Chung](https://github.com/robot0321), Members : [Stella Yang](https://github.com/codestella), [Alex Lau](https://github.com/riven314), [Haswanth Aekula](https://github.com/hassiahk), [Hyunkyu Kim](https://github.com/minus31)           |
| CLIP Team        | Leader : [Sunghyun Kim](https://github.com/MrBananaHuman), Members : [Seunghyun Lee](https://github.com/sseung0703), [Sasikanth Kotti](https://github.com/ksasi), [Khali Sifullah](https://github.com/khalidsaifullaah)                                 |
| Cloud TPU Team   | Leader : [Alex Lau](https://github.com/riven314), Members : [JaeYoung Chung](https://github.com/robot0321), [Sunghyun Kim](https://github.com/MrBananaHuman), [Aswin Pyakurel](https://github.com/masapasa)                                                     |
| Project Managing | [Stella Yang](https://github.com/codestella) To Watch Our Project Progress, Please Check [Our Project Notion](https://www.notion.so/Putting-NeRF-on-a-Diet-e0caecea0c2b40c3996c83205baf870d) |

## References
This project is based on “JAX-NeRF”.
```
@software{jaxnerf2020github,
  author = {Boyang Deng and Jonathan T. Barron and Pratul P. Srinivasan},
  title = {{JaxNeRF}: an efficient {JAX} implementation of {NeRF}},
  url = {https://github.com/google-research/google-research/tree/master/jaxnerf},
  version = {0.0},
  year = {2020},
}
```

This project is based on “JAX-NeRF”.
```
@misc{jain2021putting,
      title={Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis}, 
      author={Ajay Jain and Matthew Tancik and Pieter Abbeel},
      year={2021},
      eprint={2104.00677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
[Apache License 2.0](https://github.com/codestella/putting-nerf-on-a-diet/blob/main/LICENSE)
