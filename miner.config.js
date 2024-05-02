module.exports = {
  apps: [
    {
      name: "miner",
      script: "python3",
      args: "./neurons/miner.py --netuid 93 --logging.debug --logging.trace --subtensor.network test --wallet.name ken_coldkey --wallet.hotkey ken_hotkey --axon.port 8091 --model bitmind6_lstm.h5",
    },
  ],
};
