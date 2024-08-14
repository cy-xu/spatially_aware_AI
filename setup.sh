set -e

conda create -n cf -y
conda activate cf

# open3d only has builds for python <=3.10
conda install python=3.10 pip -y

pip install \
    torch \
    matplotlib \
    numpy \
    opencv-python \
    scikit-image \
    scipy \
    tqdm \
    h5py \
    pandas \
    open_clip_torch \
    open3d \
    pillow


pip install \
    black \
    ipdb \
    ipython \
    line_profiler

https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip