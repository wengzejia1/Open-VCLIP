# Create Environment

conda create -n ivnet python=3.8.13 pip
conda activate ivnet

# Install pytorch 1.11.0
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

# Dependencies
```bash
pip install scipy
pip install pandas
pip install scikit-learn
pip install ftfy
pip install regex
```

# Install other dependencies and PySlowFast

more details can be viewed in [`PySlowFast_Install`](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md)

- **Ensure GCC >= 4.9**

```bash
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
conda install av -c conda-forge
pip install -U iopath
pip install psutil
pip install opencv-python
pip install tensorboard
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
pip install 'git+https://github.com/facebookresearch/fairscale'
```

**Build SlowFast in OpenVCLIP**

```
export PYTHONPATH=/path/to/OpenVCLIP/slowfast:$PYTHONPATH
cd OpenVCLIP
python setup.py build develop
```



