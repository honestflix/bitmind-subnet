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

from transformers import set_seed
from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np
import torch
import base64
import requests
import random
import re

from bitmind.utils.uids import get_random_uids
from bitmind.protocol import ImageSynapse
from bitmind.validator.reward import get_rewards, reward


def download_image(url):
    print(f'downloading {url}')
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            print(f"Failed to download image: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def get_b64_image_from_dataset(dataset, index):

    sample = dataset[int(index)]
    image_bytes = None

    if 'url' in sample:
        image_bytes = download_image(sample['url'])
    elif 'image' in sample:
        if isinstance(sample['image'], Image.Image):
            image_bytes = BytesIO()
            sample['image'].save(image_bytes, format="JPEG")

    if image_bytes is None:
        return None

    return base64.b64encode(image_bytes.getvalue())


def generate_prompt(generator, starting_text, ideas):
    seed = random.randint(100, 1000000)
    set_seed(seed)

    if starting_text == "":
        starting_text: str = ideas[random.randrange(0, len(ideas))].replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

    response = generator(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
    response_list = []
    for x in response:
        resp = x['generated_text'].strip()
        if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
            response_list.append(resp+'\n')

    response_end = "\n".join(response_list)
    response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
    response_end = response_end.replace("<", "").replace(">", "")

    if response_end != "":
        return response_end


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

    # get generated images, either from diffuser model if gpu is available, otherwise from local dataset
    if self.gpu > 0:
        print("Generating images...")
        prompts = [generate_prompt(
            self.prompt_generator, "A realistic image", self.ideas_text)]
        gen_images = []
        for prompt in prompts:
            print("Prompt:", prompt)
            gen_images.append(self.diffuser(prompt=prompt).images[0])
    else:
        print("Sampling generated images from dataset...")
        gen_images_idx = np.random.choice(
            list(range(0, len(self.gen_dataset))), k_gen, replace=False)
        gen_b64_images = [
            get_b64_image_from_dataset(self.gen_dataset, i)
            for i in gen_images_idx
        ]
        gen_image_names = [
            self.gen_dataset[int(i)]['name'] for i in gen_images_idx
        ]

    print("Sampling real images from dataset...")
    real_images_idx = np.random.choice(
        list(range(0, len(self.real_dataset))), k_real, replace=False)
    real_b64_images = [
        get_b64_image_from_dataset(self.real_dataset, i)
        for i in real_images_idx
    ]
    real_image_names = [
        self.real_dataset[int(i)]['url'] for i in real_images_idx
    ]
    real_image_names = [
        n for i, n in enumerate(real_image_names)
        if real_b64_images[i] is not None
    ]
    real_b64_images = [img for img in real_b64_images if img is not None]

    names = real_image_names + gen_image_names
    b64_images = np.concatenate([real_b64_images, gen_b64_images], axis=0)
    labels = np.array([0] * len(real_b64_images) + [1] * len(gen_b64_images))

    images_labels_names = list(zip(b64_images, labels, names))
    np.random.shuffle(images_labels_names)

    b64_images = [s[0] for s in images_labels_names]
    labels = torch.FloatTensor([int(s[1]) for s in images_labels_names])
    names = [s[2] for s in images_labels_names]


    for uid in miner_uids:
        print("miner", uid, ":", self.metagraph.axons[uid])

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

    for image_name, label, pred in zip(names, labels, responses[0]):
        s = '[INCORRECT]' if np.round(pred) != label else '\t  '
        print(s, image_name, label, pred)

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")
    # TODO why is self passed here in the bittensor template? overkill for moving response to device
    rewards, metrics = get_rewards(labels=labels, responses=responses)
    print(rewards)
    print(metrics)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
