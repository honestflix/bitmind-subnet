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

import bittensor as bt
import numpy as np
import torch
import base64
import os

from bitmind.utils.uids import get_random_uids
from bitmind.protocol import ImageSynapse
from bitmind.validator.reward import get_rewards, reward


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    image_dir = '/Users/duys/proj/bitmind/bitmind-subnet/bitmind/data'
    #image_files = ['pope_FAKE.jpg', 'tahoe_REAL.jpg']

    """
    placeholder logic for getting labels and sampling real and fake images
    
    TODO set up models for generating fake images
    TODO data augmentation for real images
    """
    prompt = "An astronaut riding a green horse"
    images = self.diffuser(prompt=prompt).images[0]
    print(images)

    image_files = np.array(os.listdir(image_dir))
    labels = np.array([
        1 if f.split('_')[-1].split('.')[0] == 'FAKE'
        else 0 for f in image_files
    ])

    k_fake = 10
    k_real = 10
    fake_idx = np.random.choice(np.where(labels == 0)[0], k_fake)
    real_idx = np.random.choice(np.where(labels == 1)[0], k_real)
    idx = np.concatenate([real_idx, fake_idx])
    np.random.shuffle(idx)

    image_files = image_files[idx]
    labels = torch.FloatTensor(labels[idx])

    b64_images = []
    for image_file in image_files:
        image_abspath = os.path.join(image_dir, image_file)
        with open(image_abspath, "rb") as fin:
            b64_image = base64.b64encode(fin.read()).decode('utf-8')
            b64_images.append(b64_image)

    #for uid in miner_uids:
    #    print("miner", uid, ":", self.metagraph.axons[uid])

    # The dendrite client queries the network.
    responses = await self.dendrite(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a query. TODO: do all miners get the same query?
        synapse=ImageSynapse(images=b64_images, predictions=[]),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )

    for image, label, pred in zip(image_files, labels, responses[0]):
        s = '[INCORRECT]' if np.round(pred) != label else '\t  '
        print(s, image, label, pred)

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")
    # TODO why is self passed here in the bittensor template? overkill for moving response to device
    rewards, metrics = get_rewards(labels=labels, responses=responses)
    print(rewards)
    print(metrics)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
