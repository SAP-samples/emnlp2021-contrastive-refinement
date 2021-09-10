# Towards Zero-shot Commonsense Reasoning with Self-supervised Refinement of Language Models
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


In this repository we will provide the source code for the paper *Towards Zero-shot Commonsense Reasoning with Self-supervised Refinement of Language Models* to be presented at  [EMNLP 2021](https://2021.emnlp.org/). The code is in parts based on the code from [Huggingface Tranformers](https://github.com/huggingface/transformers) and the paper [A Surprisingly Robust Trick for Winograd Schema Challenge](https://github.com/vid-koci/bert-commonsense).

## Abstract

![Schematic Illustration of Contrastive Refinement](https://raw.githubusercontent.com/SAP-samples/emnlp2021-contrastive-refinement/main/img/refinement_task.png)

Can we get existing language models and refine them for zero-shot commonsense reasoning? 
This paper presents an initial study exploring the feasibility of zero-shot commonsense reasoning for the Winograd Schema Challenge by formulating the task as self-supervised refinement of a pre-trained language model. In contrast to previous studies that rely on fine-tuning annotated datasets, we seek to boost conceptualization via loss landscape refinement. To this end, we propose a novel self-supervised learning approach that refines the language model utilizing a set of linguistic perturbations of similar concept relationships. Empirical analysis of our conceptually simple framework demonstrates the viability of zero-shot commonsense reasoning on multiple benchmarks.

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements

## Download and Installation

## Known Issues

## How to obtain support

[Create an issue](https://github.com/SAP-samples/emnlp2021-contrastive-refinement/issues) in this repository if you find a bug or have questions about the content.
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing

## License
Copyright (c) 2021 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
