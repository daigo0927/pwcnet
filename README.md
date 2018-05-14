# PWC-Net_tf
PWC-Network with TensorFlow

# Acknowledgments
- [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch): framework, data transformers, loss functions, and many details about flow estimation.
- [nameloss-Chatoyant/PWC-Net_pytorch](https://github.com/nameless-Chatoyant/PWC-Net_pytorch.git): Referenced implmentation.


## A visualization of estimated optical flow

![optical flow](https://github.com/daigo0927/PWC-Net_tf/blob/master/figure/flow_0000.pdf)

**Working confirmed. I hope this helps you.**  
Unofficial implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**. [arXiv](https://arxiv.org/abs/1709.02371)


# Usage
- Requirements
    - Python 3.6+
    - PyTorch 0.4.0 (mainly in in data handling)
    - TensorFlow 1.8

- `model_1000epoch/model_399.ckpt` is fully trained by [SintelClean](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) dataset. The model_**399** is because I trained this splittingly (with no particular meaning),
so this is actually obtained after 1000 epochs.

## Training (the case SintelClean)

```
python train.py --dataset SintelClean --dataset_dir path/to/MPI-Sintel-complete 
```

```
# Start with learned checkpoint
python train.py --dataset SintelClean --dataset_dir path/to/MPI-Sintel-complete --resume model_1000epoch/model_399.ckpt
```

After running above script, utilize GPU-id is asked, (-1:CPU). You can use other learning configs (like `--n_epoch` or `--batch_size`) see all arguments in `train.py`, regards.

