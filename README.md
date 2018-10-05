# PWC-Net_tf
PWC-Network with TensorFlow
![estimated flow](https://i.imgur.com/DfN8glU.png)

input image0, estimated flow at each 5 scale, ground truth flow, input image1


# Acknowledgments
- [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net): original implementation (by Caffe / PyTorch).
- [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch): framework, data transformers, loss functions, and many details about flow estimation.
- [nameloss-Chatoyant/PWC-Net_pytorch](https://github.com/nameless-Chatoyant/PWC-Net_pytorch.git): Referenced implementation.


**Working confirmed. I hope this helps you.**  
**Adding PWCDCNet: advanced model! I'm going to make the trained parameters avaliable ASAP! :)**  
Unofficial implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**. [arXiv](https://arxiv.org/abs/1709.02371)


# Usage
- Requirements
    - Python 3.6+
    - PyTorch 0.4.0 (mainly in in data handling)
    - TensorFlow 1.8+

<!-- - `model_3000epoch/model_3007.ckpt` is fully trained by [SintelClean](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) dataset. -->

## Training (the case SintelClean)

```
# Training from scratch
python train.py --dataset SintelClean --dataset_dir /path/to/MPI-Sintel-complete 
```

```
# Start with learned checkpoint
python train.py --dataset SintelClean --dataset_dir /path/to/MPI-Sintel-complete --resume /path/to/model.ckpt
```

After running above script, utilize GPU-id is asked, (-1:CPU). You can use other learning configs (like `--n_epoch` or `--batch_size`) see all arguments in `train.py`, regards.

## Testing (inferring optical flow) by paired images

```
python test.py --input_images /path/to/image_0 /path/to/image_1 --resume /path/to/model.ckpt
```
```
# for multiple images (using wild-card for 001.png, 002.png, 003.png, ...)
python test_continuous.py -i /path/to/images* -r /path/to/model.ckpt
```
