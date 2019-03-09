# advGAN_pytorch
a Pytorch implementation of the paper "Generating Adversarial Examples with Adversarial Networks" (advGAN).

## training the target model

```shell
python3 train_target_model.py
```

## training the advGAN

```shell
python3 main.py
```

## testing adversarial examples

```shell
python3 test_adversarial_examples.py
```

## results

**attack success rate** in the MNIST test set: **99%**

**NOTE:** My implementation is a little different from the paper, because I add a clipping trick.
