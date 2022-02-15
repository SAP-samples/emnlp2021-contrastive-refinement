# Towards Zero-shot Commonsense Reasoning with Self-supervised Refinement of Language Models
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/emnlp2021-contrastive-refinement)](https://api.reuse.software/info/github.com/SAP-samples/emnlp2021-contrastive-refinement)

In this repository we will provide the source code for the paper [*Towards Zero-shot Commonsense Reasoning with Self-supervised Refinement of Language Models*](https://arxiv.org/abs/2109.05105) to be presented at  [EMNLP 2021](https://2021.emnlp.org/). The code is in parts based on the code from [Huggingface Tranformers](https://github.com/huggingface/transformers) and the paper [A Surprisingly Robust Trick for Winograd Schema Challenge](https://github.com/vid-koci/bert-commonsense).

## Abstract

![Schematic Illustration of Contrastive Refinement](https://raw.githubusercontent.com/SAP-samples/emnlp2021-contrastive-refinement/main/img/refinement_task.png)

Can we get existing language models and refine them for zero-shot commonsense reasoning? 
This paper presents an initial study exploring the feasibility of zero-shot commonsense reasoning for the Winograd Schema Challenge by formulating the task as self-supervised refinement of a pre-trained language model. In contrast to previous studies that rely on fine-tuning annotated datasets, we seek to boost conceptualization via loss landscape refinement. To this end, we propose a novel self-supervised learning approach that refines the language model utilizing a set of linguistic perturbations of similar concept relationships. Empirical analysis of our conceptually simple framework demonstrates the viability of zero-shot commonsense reasoning on multiple benchmarks.

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)


## Requirements
- [Python](https://www.python.org/) (version 3.6 or later)
- [PyTorch](https://pytorch.org/)
- [Huggingface Tranformers](https://github.com/huggingface/transformers)


## Download and Installation

1. Install the requiremennts:

```
conda install --yes --file requirements.txt
```

or

```
pip install -r requirements.txt
```

2. Clone this repository and install dependencies:
```
git clone https://github.com/SAP-samples/emnlp2021-attention-contrastive-learning
cd emnlp2021-attention-contrastive-learning
pip install -r requirements.txt
```

3. Create 'data' sub-directory and download files for PDP, WSC challenge, KnowRef, DPR and WinoGrande:
```
mkdir data
wget https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/PDPChallenge2016.xml
wget https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml
wget https://raw.githubusercontent.com/aemami1/KnowRef/master/Knowref_dataset/knowref_test.json
wget http://www.hlt.utdallas.edu/~vince/data/emnlp12/train.c.txt
wget http://www.hlt.utdallas.edu/~vince/data/emnlp12/test.c.txt
wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
unzip winogrande_1.1.zip
rm winogrande_1.1.zip
cd ..
```

4.  Get the Pertubed-WSC dataset of the paper [The Sensitivity of Language Models and Humans to Winograd Schema Perturbations](https://arxiv.org/pdf/2005.01348.pdf).
```
wget https://raw.githubusercontent.com/mhany90/perturbed-wsc/release/data/dataset/enhanced_wsc.tsv
```

5. Train
```
python refine_lm.py -do_train --do_eval --shuffle_batch --output_experimentname --R_weight=0.5 --C_weight=75 --C_loss= --C_weight=0.5 --D_weight=0.5  --adam_epsilon=1e-08 --bert_model=bert-large-uncased --eval_task_name dpr --cls_hidden_dropout=0.1 --cls_hidden_size=128 --data_dir=data/ --eval_batch_size=10 --eval_steps=5 --gen_perturbations=1 --learning_rate=5e-05 --matrix_band=4 --max_grad_norm=1 --max_seq_length=40 --num_train_epochs=10 --output_dir=model_output/experiment/ --ranking_steps=50 --reg_perturbations=2 --schedule=cosine --train_batch_size=10 --warmup_steps=250
```


## How to obtain support

[Create an issue](https://github.com/SAP-samples/emnlp2021-contrastive-refinement/issues) in this repository if you find a bug or have questions about the content.
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Citations
If you use this code in your research,
please cite:

```
@inproceedings{klein-nabi-2021-towards,
    title = "Towards Zero-shot Commonsense Reasoning with Self-supervised Refinement of Language Models",
    author = "Klein, Tassilo  and
      Nabi, Moin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.688",
    pages = "8737--8743",
}
```


## License
Copyright (c) 2021 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
