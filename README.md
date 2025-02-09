# AutoCAR codebase

[Project Website](https://autocar.zyh.science) | [Paper(TBD)]() | [![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This is the official projection repo of paper *Sparse and Transferable 3D Dynamic Vascular Reconstruction For Instantaneous Diagnosis*.
<!-- | [Static 3D Viewer](https://zhuyh19-autocar-synthetic-data-preview.static.hf.space) | [Dynamic 3D Viewer](https://autocar.zyh.science/dynamic_full.html) 
<video height="512" controls autoplay>
    <source src="teaser.mp4" type="video/mp4" >
</video> -->

https://github.com/user-attachments/assets/824e57a0-3bb4-4f66-ac27-0313a1dc87db

<!-- <iframe
	src="https://zhuyh19-autocar-synthetic-data-preview.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe> -->
<!-- [![Title](Image URL)](Link URL) -->

## Resources List

- training dataset: [Download link](https://1drv.ms/u/s!AutE3SVE1oKchJIM9CwIoGtZ5jMY0w?e=s1gPeh)
- synthetic evaluation dataset: [Download link](https://1drv.ms/u/s!AutE3SVE1oKchJF8ttxgXHtXNOTlbQ?e=feGD3t)
- realworld dataset: [Download link](https://1drv.ms/u/s!AutE3SVE1oKchJE3yQtkdYId-W2MJw?e=sNbzyQ)
- pretrained weights: [Download link](https://1drv.ms/u/s!AutE3SVE1oKchJIM9CwIoGtZ5jMY0w?e=UH8maE)
- zipped file of training dataset, pretrain weights and sample test set (used in (#reproduction)): [Download link](https://1drv.ms/u/s!AutE3SVE1oKchJE4cTH2uraT8v4wfw?e=vaq7nM)
- predictions of existing and proposed method(105GB): [Download link](https://1drv.ms/f/s!AutE3SVE1oKchJIAOApDjBBgoc7Hmg?e=nR026i)
- Sample predictions of existing and proposed method(200MB): [Download link](https://1drv.ms/u/s!AutE3SVE1oKchJIIJMW7D7uZXyfF1w?e=uGbOXf)


## Reproduction

Hints:
1. The Sparse Backward Projection Modules are defined in `src/modules/ray_casting.py`. Other related network modules are also defined in this folder, such as pytorch3d rendering modules, 2D and 3D backbone.
2. The graph-related algorithms are defined in the `src/geometry` folder, which includes graph structure generation in `curvenetwork.py`, deformation graph in `deformation_graph.py`, and 3D-2D matching/3D-3D matching in `correspondence.py`. Multi-view images and video can be found in `multiview-image.py`.
3. The entry point for training is `src/train.py`, while the entry point for inference is `src/inference.py`.
4. For reproduction, please refer to [Reproduction Section](#reproduction)
5. For results inspection, please refer to [Project Website](https://autocar.zyh.science)

### Environment and Libraries

The Sparse Operator Library requires the Cpp and NVCC compilers. Please ensure that you have the correct compiler and CUDA driver version. It is recommended to use the same CUDA and Torch versions as follows:

Tested Environment: Ubuntu 20.04, Nvidia 2080Ti, CUDA 11.3

#### CUDA
1. Make sure that the Nvidia driver is installed and its version is greater than 530. You can check this by running `nvidia-smi`.
2. Install CUDA 11.3, which is a dependency for the Minkowski Engine. Installing it from conda, for example, `conda install cudatoolkit`, is not valid as nvcc is required to build `ME` from source. Follow these steps:
    ```
    wget -c https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
    sudo sh cuda_11.3.1_465.19.01_linux.run --silent --toolkit --installpath=/usr/local/cuda-11.3
    sudo reboot # reboot to take effect
    # check if nvcc is installed successfully
    /usr/local/cuda-11.3/bin/nvcc -V
    ```
3. Run `sh install_env.sh`.
4. Validate the installation by running:
    ```
    nvcc -V
    ```
#### Libraries

Install conda environment
```{shell}
    conda create -y -n autocar python=3.8.13
    conda activate autocar

    # copy config cuda, BE CAREFUL HERE, PLEASE ENSURE cudatoolkit=11.3 or consistent with the NVCC compiler.
    conda install -y pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
    conda install -y openblas-devel -c anaconda

    # pytorch 3d
    conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -y -c bottler nvidiacub
    conda install -y pytorch3d -c pytorch3d

    ## Demos and examples need local build
    pip install k3d notebook simpleitk pydicom pylibjpeg \
        ipyvolume tensorboardx h5py  pandas scikit-image \
        matplotlib imageio plotly opencv-python \
        numpy scipy  scikit-learn imageio-ffmpeg tqdm black install opencv-contrib-python open3d "napari[all]" trimesh \
        hydra-core hydra-colorlog hydra-optuna-sweeper lightning==1.9 torchmetrics rootutils pre-commit rich pytest \
        "tensorboard<2.16" "protobuf<5" opencv-contrib-python opencv-python opencv-python-headless opencv-contrib-python-headless
```

Compiling the Minkowski Engine
```{shell}
    export CUDA_VERSION=11.3
    export PATH=$PATH:/usr/local/cuda-11.3/bin
    export CPATH=$CPATH:/usr/local/cuda-11.3/include
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.3

    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    # if you are using CUDA 12 or newer, please follows the solution in https://github.com/NVIDIA/MinkowskiEngine/issues/543
    # to modify some include files of MinkowskiEngine
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### Download Training/Testing Data and Weights

Download the preprocessed dataset by following these steps:

    ```{shell}
    # cd <project-root>
    # this link is expired because the author has successfully graduated and has no access to Tsinghua Cloud anymore.
    # wget -O autocar_data.zip https://cloud.tsinghua.edu.cn/f/d5357eb6d49a4009b009/?dl=1
    # It's currently hosted in Onedrive, where direct download link is not allowed. 
    # Please go the following link to download maunlly.
    # https://1drv.ms/u/s!AutE3SVE1oKchJE4cTH2uraT8v4wfw?e=vaq7nM
    unzip autocar_data.zip
    # check folder structure:
    # - <project-root>/data
    #   - imagecas_left_surface/*.ply: Left artery surface
    #   - imagecas_right_surface/*.ply: Right artery surface
    #   - testset: test set
    #   - testset_result: exemplar results
    #   - filelist_train/val.txt: pre-shuffled split
    #   - weights_left.ckpt: weights for Left artery reconstruction
    ```

For unprocessed dataset and furhter validation, please download at [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT#imagecas-a-large-scale-dataset-and-benchmark-for-coronary-artery-segmentation-based-on-computed-tomo)

For visualize and parse real_world dataset:
```{shell}
# visualize the dataset
python -m http.server realworld_dataset
# this link is expired because the author has successfully graduated and has no access to Tsinghua Cloud anymore.
# wget -O autocar_realworld_data.zip https://cloud.tsinghua.edu.cn/f/d25f88acc10c41a4b685/?dl=1
# It's currently hosted in Onedrive, where direct download link is not allowed. 
# Please go the following link to download maunlly.
# https://1drv.ms/u/s!AutE3SVE1oKchJE3yQtkdYId-W2MJw?e=sNbzyQ

unzip autocar_realworld_data.zip

# parse the dataset: index.html
# check folder structure:
# - <project-root>/realworld_data
#    - cases/case_<idx>_camera_poses.json: a list of camera poses,including rotation matrix R and translational vector t, for each view
#    - cases/case_<idx>_view_1.mp4: XA videos for view 1
#    - cases/case_<idx>_view_2.mp4: XA videos for view 2
```

### Inference Only

```

export CUDA_VERSION=11.3
export PATH=$PATH:/usr/local/cuda-11.3/bin
export CPATH=$CPATH:/usr/local/cuda-11.3/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.3

python src/inference.py experiment=exp_left ckpt_path="data/weights_left.ckpt"
python src/inference.py experiment=exp_right ckpt_path="data/weights_right.ckpt"

# The interactive result will be shown and the result files will be saved to ./logs/eval/<DATE-TIME>/
# Configurations can be modified in ./configs/eval.yaml
```

### Training from Scratch

```

export CUDA_VERSION=11.3
export PATH=$PATH:/usr/local/cuda-11.3/bin
export CPATH=$CPATH:/usr/local/cuda-11.3/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.3

python src/train.py experiment=exp_left
python src/train.py experiment=exp_right
# Different configurations can be found in ./configs/train.yaml, such as multi-gpu. Single GPU is used by default.
```

## Licence

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
