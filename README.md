# Bitmind Subnet

## Local Testing Setup
Follow Bittensor's <a href="https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_staging.md">Running on Staging docs</a> to get a local version of Bittensor running
   - After cloning the subtensor repository (step 3), make sure to run `git checkout main` before running the subsequent build step (step 4)
   - If you're getting `eth-typing` warnings about ChainIds, run `pip install --force-reinstall eth-utils==2.1.1`


## Start Miner
Initial testing being done with base miner from:
https://huggingface.co/spaces/Wvolf/CNN_Deepfake_Image_Detection/tree/main
Clone this repo and move the model file into the root of this repository before running miner or validator

```
python ./neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default --logging.debug
```

## Start Validator
In local testing, set the --neuron.vpermit_tao_limit to something far lower than its default so that all miners will pay attention to all validators (without having to mint a bunch of tao)
```
python ./neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey validator_hot --logging.trace --logging.info --logging.debug --neuron.vpermit_tao_limit 500
```
