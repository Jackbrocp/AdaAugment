# AdaAugment

This is the implementation of "AdaAugment: A Tuning-Free and Adaptive Approach to Enhance Data Augmentation". 

## Requirements

* python >= 3.6
* PyTorch >= 1.1.0
* Torch vision >= 0.3.0

## Updates

* 2024/11/27: Initial release

## Datasets

[CIFAR-10]: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[CIFAR-100]: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
[CIFAR home page]: https://www.cs.toronto.edu/~kriz/cifar.html

The datasets can be directly downloaded at [CIFAR-10] and [CIFAR-100].

For further information, please check [CIFAR home page].

## Training Examples

```bash 
python train.py --conf <config file path> --dataset CIFAR10 --dataset_path <dataset path>
# or
python train.py --conf <config file path> --dataset CIFAR100 --dataset_path <dataset path>
```

You can also **add other parameters** according to your needs.

## Acknowledge

Part of our implementation is adopted from the [TrivialAugment](https://github.com/automl/trivialaugment) repositories.

## Citation
If you find this repository useful in your research, please cite our paper:

'
@article{yang2024adaaugment,
  title={AdaAugment: A Tuning-Free and Adaptive Approach to Enhance Data Augmentation},
  author={Yang, Suorong and Li, Peijia and Xiong, Xin and Shen, Furao and Zhao, Jian},
  journal={arXiv preprint arXiv:2405.11467},
  year={2024}
}
'

