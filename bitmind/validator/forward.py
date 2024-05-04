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

from io import BytesIO
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
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    k_gen = 5
    k_real = 5

    # get generated images, either from diffuser model if gpu is available, otherwise from cifake datset
    # TODO: expand datset sources
    if self.gpu > 0:
        prompts = []  # TODO prompt generation
        gen_images = []
        for prompt in prompts:
            gen_images.append(self.diffuser(prompt=prompt).images[0])
    else:
        print("Sampling generated images from dataset...")
        gen_images_idx = np.random.choice(self.gen_images_idx, k_gen, replace=False)

    print("Sampling real images from dataset...")
    real_images_idx = np.random.choice(self.real_images_idx, k_gen, replace=False)
    images_idx = np.concatenate([real_images_idx, gen_images_idx])
    labels = np.array([0] * k_real + [1] * k_gen)

    images_idx_labels = list(zip(images_idx, labels))
    np.random.shuffle(images_idx_labels)

    images_idx = np.array([s[0] for s in images_idx_labels])
    labels = torch.FloatTensor([int(s[1]) for s in images_idx_labels])

    print("Encoding...")
    b64_images = []
    for image_idx in images_idx:
        image = self.dataset[int(image_idx)]['image']
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        b64_images.append(base64.b64encode(buffered.getvalue()))


    #for uid in miner_uids:
    #    print("miner", uid, ":", self.metagraph.axons[uid])

    print("Querying miners...")
    # The dendrite client queries the network.
    responses = await self.dendrite(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a query.
        synapse=ImageSynapse(images=b64_images, predictions=[]),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )

    for image, label, pred in zip(images_idx, labels, responses[0]):
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
