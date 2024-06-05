# Bitmind Subnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

- [Introduction](#introduction)
- [Setup](#setup)
- [Mining](#mining)
- [Validating](#validating)
- [License](#license)

## Introduction

**IMPORTANT**: If you are new to Bittensor, please review the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section.

Introducing Bittensor Subnet X (Deep Fake Image Detection Subnet): A Platform for Identifying and Mitigating Deep Fake Image Content.

The Deep Fake Image Detection Subnet is at the vanguard of cybersecurity and digital authenticity, providing a robust solution tailored for developers and researchers in the fields of AI and image analysis. This platform is intricately designed to offer accurate, real-time detection of deep fake images through API usage, leveraging the decentralized Bittensor network. It is a pivotal tool in the quest to ensure the integrity of digital media by facilitating the incentivized detection and mitigation of manipulated content.

Our initiative marks a significant advancement in securing digital environments, addressing the growing concerns over deep fake technology. By focusing on delivering consistent and reliable detection results, we aim to alleviate the issues of misinformation and digital deception that pervade modern media. The quality and reliability of the Deep Fake Image Detection Subnet are seamlessly integrated within the Bittensor network, enabling developers to utilize multiple subnets and modalities by integrating directly with an existing validator or via an API key from BitMind.

The Deep Fake Image Detection Subnet also employs advanced AI models to enhance its detection capabilities, continuously evolving to counteract new methods of image manipulation. This approach involves generating and analyzing synthetic data to improve detection algorithms, akin to the strategies used by leading AI researchers. By iterating on model outputs and utilizing data augmentation techniques, we ensure our detection models are both sophisticated and efficient, mirroring the performance of state-of-the-art AI technologies. This democratizes access to high-quality detection tools, encouraging widespread adoption and innovation.

By harnessing synthetic data, the Deep Fake Image Detection Subnet overcomes traditional data collection challenges, accelerating the development of robust and adaptive AI models. This platform is your gateway to mastering digital authenticity, offering the unique opportunity to train your models with data that captures the intricacies of deep fake detection. With our advanced tools and methodologies, you're not just identifying deep fakes; you're safeguarding the truth, providing a pathway to creating AI models that uphold the highest standards of digital integrity.

Join us at the Deep Fake Image Detection Subnet, your partner in maintaining digital authenticity and leading the fight against misinformation. Be part of the solution and stay at the forefront of innovation with our cutting-edge detection tools – Defending Truth, Empowering the Future!

## Setup

### Local Subtensor Setup

To run locally follow Bittensor's <a href="https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_staging.md">Running on Staging docs</a> to get a local version of Bittensor running

- After cloning the subtensor repository (step 3), make sure to checkout the main branch before running the subsequent build step (step 4)<br>
  `git checkout main`
- If you're getting `eth-typing` warnings about ChainIds, run:<br>
  `pip install --force-reinstall eth-utils==2.1.1`

### Before you proceed

Before you proceed with the installation of the subnet, note the following:

**IMPORTANT**: We **strongly recommend** before proceeding that you test both subtensor and all API keys. Ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary)

**IMPORTANT:** Make sure you are aware of the minimum compute requirements for bitmind subnet. See the [Minimum compute YAML configuration](./min_compute.yml).

### Installation

Before starting make sure you have pm2, nano and any other useful tools installed.

```bash
apt update -y && apt-get install git -y && apt install python3-pip -y && apt install npm -y && npm install pm2@latest -g  && apt install nano
```

It is recommended you use a virtual environment to install the necessary requirements.

```
conda create -n deepfake python=3.10 ipython
conda activate deepfake
```

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```bash
git clone https://github.com/bitmind-ai/bitmind-subnet.git && cd bitmind-subnet && pip
conda create -n bitmind python=3.10 ipython
conda activate bitmind
export PIP_NO_CACHE_DIR=1
pip install -r requirements.txt
pip install -e .
```

Prior to proceeding, ensure you have a registered hotkey on subnet XX testnet. If not, run the command

```bash
btcli s register --netuid XX --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

### Getting Data

1. Create a Kaggle Account
2. Get a Kaggle API Key - but kaggle.json in `$HOME/.kaggle/` directory
3. Run:

```python
python base_miner/get_data.py
```

You can also manually download the dataset here: [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
Create a directory called `data` in base_miner and move the `test` and `train` image directories into the `data` directory.

### Training a Model

1. Modify `model.py` to improve performance of the base model
2. Run:

```python
python base_miner/model.py
```

### Model Prediction / Inference

```python
python base_miner/predict.py
```

## Mining

You can launch your miners via pm2 using the following command.

```bash
pm2 start ./neurons/miner.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>
```

### Testnet Example

```bash
pm2 start ./neurons/miner.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default --axon.port 8091
```


## Validating

You can launch your validator via pm2 using the following command.

```bash
pm2 start ./neurons/validator.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid XX --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

### Testnet Example

```bash
pm2 start ./neurons/validator.py --interpreter $HOME/miniconda3/envs/deepfake/bin/python3 -- --netuid 168 --subtensor.network test --wallet.name default --wallet.hotkey default
```

---

## License

This repository is licensed under the MIT License.

```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
