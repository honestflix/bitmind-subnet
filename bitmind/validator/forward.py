# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: dubm
# Copyright © 2023 Bitmind

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

from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np
import time
import torch
import base64
import requests
import joblib

from bitmind.utils.uids import get_random_uids
from bitmind.protocol import ImageSynapse, prepare_image_synapse
from bitmind.validator.reward import get_rewards



async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    if np.random.rand() > .5:
        print('sampling real image')
        sample = self.real_dataset.sample(k=1)[0]
        label = 0
    else:
        print('generating fake image')
        sample = self.random_image_generator.generate(k=1)[0]
        label = 1

    image = sample['image'] 

    print(f"Querying {len(miner_uids)} miners...")
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=prepare_image_synapse(image=image),
        deserialize=True
    )

    rewards = get_rewards(label=label, responses=responses)

    # debug outputs for rewards
    print(f'{"real" if label == 0 else "fake"} image | source: {sample["id"]}')
    for i, pred in enumerate(responses):
        print(f'Miner uid: {miner_uids[i]} | prediction: {pred} | correct: {np.round(pred) == label} | reward: {rewards[i]}')

    bt.logging.info(f"Received responses: {responses}")
    bt.logging.info(f"Scored responses: {rewards}")
    
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    rewards = rewards.to('cuda')
    miner_uids = miner_uids.to('cuda')
    self.update_scores(rewards, miner_uids)
