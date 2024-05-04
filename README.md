# Bitmind Subnet

## Local Testing Setup
1. Clone and install this repository in a virtual environemnt:
    ```
    git clone git@github.com:BitMind-AI/bitmind-subnet.git
    ```
    or with https
   ```
   git clone https://github.com/BitMind-AI/bitmind-subnet.git
   ```
   ```
    cd bitmind-subnet
    conda create -n bitmind python=3.10 ipython
    conda activate bitmind
    export PIP_NO_CACHE_DIR=1
    python3 -m pip install -r requirements.txt
    python3 -m pip install -e .
   ```
3. Follow Bittensor's <a href="https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_staging.md">Running on Staging docs</a> with the following modifications:
  - **Modified step 3**
     ```
    git clone https://github.com/opentensor/subtensor.git
    git checkout main
    ```
  - **Skip step 6**

*note*: If you are getting `eth-typing` warnings about ChainIds, run:<br>
   `pip install --force-reinstall eth-utils==2.1.1`


## Miner Training Quickstart
Note - To test the miner, you can use the provided `mining_models/deepfake_detection_model.h5`.

### Getting Data

1. Create a Kaggle Account
2. Get a Kaggle API Key - but kaggle.json in `$HOME/.kaggle/` directory
3. Run:

```python
python base_miner/get_data.py
```

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

## Start Miner
```
python ./neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default --logging.debug
```

## Start Validator

In local testing, set the --neuron.vpermit_tao_limit to something far lower than its default so that all miners will pay attention to all validators (without having to mint a bunch of tao)

```
python ./neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey validator_hot --logging.trace --logging.info --logging.debug --neuron.vpermit_tao_limit 500
```
