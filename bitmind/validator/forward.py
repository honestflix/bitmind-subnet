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
from bitmind.validator.reward import get_rewards, reward



async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # TODO put this in a config
    total_images = 10
    k_gen = np.random.randint(int(.35*total_images), int(.65*total_images))
    k_real = total_images - k_gen

    # TODO create RealGen dataset class
    real_images = self.real_dataset.sample(k_real)
    labels = [0] * len(real_images)
    if self.gpu > 0:
        gen_images = self.random_image_generator.generate(k=k_gen)
        images = real_images + gen_images
        labels += [1] * len(gen_images)
    else:
        bt.logging.warning('UNABLE TO GENERATE IMAGES')
        images = real_images

    images_labels = list(zip(images, labels))
    np.random.shuffle(images_labels)

    image_samples = [s[0] for s in images_labels]
    images = [sample['image'] for sample in image_samples]
    labels = torch.FloatTensor([int(s[1]) for s in images_labels])

    #for uid in miner_uids:
    #    print("miner", uid, ":", self.metagraph.axons[uid])

    print(f"Querying miners with {len(images)} images...")
    # The dendrite client queries the network.
    responses = await self.dendrite(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a query.
        synapse=prepare_image_synapse(images=images, predictions=[]),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )

    miner_idx = np.argmax([len(r) for r in responses])
    miner = responses[miner_idx]

    for image_sample, label, pred in zip(image_samples, labels, miner):
        s = '[INCORRECT]' if np.round(pred) != label else '\t  '
        print(s, image_sample['id'], label, pred)
        self.results['challenge'].append(self.challenge)
        self.results['image'].append(image_sample['id'])
        self.results['label'].append(label)
        self.results['prediction'].append(pred)
        self.results['correct'].append(np.round(pred) == label)
    self.challenge += 1

    joblib.dump(self.results, 'results.pkl')

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")
    # TODO why is self passed here in the bittensor template? overkill for moving response to device
    rewards, metrics = get_rewards(labels=labels, responses=responses)
    print('Miner Rewards:', rewards[miner_idx])
    #print(metrics)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    rewards = rewards.to('cuda')
    miner_uids = miner_uids.to('cuda')
    self.update_scores(rewards, miner_uids)
