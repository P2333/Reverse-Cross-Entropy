# Reverse Cross Entropy Training
Reverse Cross Entropy Training (RCE) is a novel training method, which can learn more distinguished feature representations for detecting adversarial examples.
Technical details are specified in:

[Towards Robust Detection of Adversarial Examples](http://papers.nips.cc/paper/7709-towards-robust-detection-of-adversarial-examples.pdf) (NeurIPS 2018)

Tianyu Pang, Chao Du, Yinpeng Dong and Jun Zhu

## Training
We provide codes for training [ResNet](https://github.com/tensorflow/models/tree/master/research/resnet) on MNIST and CIFAR-10. Our codes are based on [Tensorflow](https://github.com/tensorflow). 

<b>Prerequisite:</b>
1. Install TensorFlow 1.9.0.

2. Download [MNIST](http://ml.cs.tsinghua.edu.cn/~tianyu/mnist_dataset.zip)/[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) dataset.

<b>How to run:</b>

An example of using RCE to train a ResNet-32 on MNIST:

```shell
python train.py --train_data_path='mnist_dataset/data_train.bin' \
                --log_root=models_mnist/resnet32 \
                --train_dir=models_mnist/resnet32/train \
                --dataset='mnist' \
                --num_gpus=1 \
                --num_residual_units=5 \
                --mode=train \
                --Optimizer='mom' \
                --total_steps=20000 \
                --RCE_train=True
```

## Test in the Normal Setting

An example of test trained ResNet-32 in the normal setting (test set) on MNIST:

```shell
python test_nor.py --eval_data_path='mnist_dataset/data_test.bin' \
                --log_root=models_mnist/resnet32 \
                --eval_dir=models_mnist/resnet32/eval \
                --dataset='mnist' \
                --num_gpus=1 \
                --num_residual_units=5 \
                --mode=eval \
                --RCE_train=True
```

## Test in the Adversarial Setting

The code of attacks is forked from [Zhitao Gong](https://github.com/gongzhitaao/tensorflow-adversarial)

An example of attacking a trained Resnet-32 by FGSM on MNIST:

```shell
python test_adv.py --eval_data_path='cifar10_dataset/test_batch.bin' \
                               --log_root=models_mnist/resnet32 \
                               --dataset='mnist' \
                               --num_gpus=1 \
                               --num_residual_units=5 \
                               --Optimizer='mom' \
                               --mode=attack \
                               --RCE_train=True \
                               --attack_method='fgsm' \
                               --eval_batch_count=5
```

The `attack_method` 
