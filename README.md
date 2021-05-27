# BMI-NMT
Source code for the ACL 2021 short paper [Bilingual Mutual Information Based Adaptive Training for Neural Machine Translation](https://arxiv.org/abs/2105.12523).

## Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Requirements](#requirements)
* [Citation](#citation)

## Introduction
+ Implementation

Implemented based on [THUMT-TensorFlow](https://github.com/THUNLP-MT/THUMT), an open-source toolkit for neural machine translation developed by the Natural Language Processing Group at Tsinghua University which was implemented strictly referring to [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).

+ Data

[WMT14 English-German](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh)

[WMT19 Chinese-English](https://drive.google.com/file/d/1LvUPsIZ_xRwuB1vHlvi1COeZEOxfbYy0/view?usp=sharing) 

## Usage
Note: The usage is on the top of THUMT, for more details, please refer to the user manual of THUMT.
+ Calculating BMI
```
python mi_calculate.py
```

+ Training
```
sh ende_mi.sh
```

## Requirements
+ Python version \>=3.6
+ Tensorflow version \>=1.12

## Citation

Please cite the following paper if you use the code:

```
@InProceedings{Xu2021bmi,
  author    = {Yangyifan Xu, Yijin Liu, Fandong Meng, Jiajun Zhang, Jinan Xu, Jie Zhou},
  title     = {Bilingual Mutual Information Based Adaptive Training for Neural Machine Translation},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year      = {2021}
}
```
