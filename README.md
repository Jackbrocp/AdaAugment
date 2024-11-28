# AdaAugment

The implementation of "AdaAugment: A Tuning-Free and Adaptive Approach to Enhance Data Augmentation".

## Requirements

* python >= 3.6
* PyTorch >= 1.1.0
* Torch vision >= 0.3.0

## Datasets

[CIFAR-10]: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[CIFAR-100]: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
[CIFAR home page]: https://www.cs.toronto.edu/~kriz/cifar.html

The datasets can be directly downloaded at [CIFAR-10] and [CIFAR-100].

For further information, please check [CIFAR home page].

## Training

```bash 
python train.py --conf <config file path> --dataset CIFAR10 --dataset_path <dataset path> --cutout_length 16
# or
python train.py --conf <config file path> --dataset CIFAR100 --dataset_path <dataset path> --cutout_length 8
```

For `CIFAR10`, parameter `cutout length` is set to **16**, and **8** for `CIFAR100`.

You can also **add other parameters** according to your needs.
