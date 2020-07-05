## Meta-Transfer Learning for Code-Switched Speech Recognition
### Genta Indra Winata, Samuel Cahyawijaya, Zhaojiang Lin, Zihan Liu, Peng Xu, Pascale Fung

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the implementation of our papers accepted in [ACL](https://www.aclweb.org/anthology/2020.acl-main.348/) 2020.

This code has been written using PyTorch. If you use any source codes or datasets included in this toolkit in your work, please cite the following papers.
```
@inproceedings{winata-etal-2020-meta,
    title = "Meta-Transfer Learning for Code-Switched Speech Recognition",
    author = "Winata, Genta Indra  and
      Cahyawijaya, Samuel  and
      Lin, Zhaojiang  and
      Liu, Zihan  and
      Xu, Peng  and
      Fung, Pascale",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.348",
    pages = "3770--3776",
}
```

## Abstract
An increasing number of people in the world today speak a mixed-language as a result of being multilingual. However, building a speech recognition system for code-switching remains difficult due to the availability of limited resources and the expense and significant effort required to collect mixed-language data. We therefore propose a new learning method, meta-transfer learning, to transfer learn on a code-switched speech recognition system in a low-resource setting by judiciously extracting information from high-resource monolingual datasets. Our model learns to recognize individual languages, and transfer them so as to better recognize mixed-language speech by conditioning the optimization on the code-switching data. Based on experimental results, our model outperforms existing baselines on speech recognition and language modeling tasks, and is faster to converge.

## Data
- SEAME Phase II datasets
- HKUST
- CommonVoice v3 (Nov 2019)
Kindly check the /data/ directory to check the data split and labels.

## Model Architecture
<img src="img/model.png" width=40%/>

## Setup
- Install PyTorch (Tested in PyTorch 1.0 and Python 3.6)
- Install library dependencies (requirement.txt)

## Bug Report
Feel free to create an issue or send email to giwinata@connect.ust.hk
