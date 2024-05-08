from typing import List
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np
import requests
import base64


def download_image(url):
    print(f'downloading {url}')
    response = requests.get(url)
    if response.status_code == 200:
        image_data =  BytesIO(response.content)
        return Image.open(image_data)

    else:
        print(f"Failed to download image: {response.status_code}")
        return None


class RealImageDataset:

    def __init__(
            self,
            huggingface_datasets: List[str]=['dalle-mini/open-images']
    ):
        self.huggingface_datasets = huggingface_datasets
        self.sources = {
            name: load_dataset(name, split='validation')
            for name in huggingface_datasets
        }
        self.sampled_images_idx = defaultdict(list)

    def _get_image(self, data_source, index):
        """

        """
        sample = self.sources[data_source][int(index)]
        if 'url' in sample:
            image = download_image(sample['url'])
            image_id = sample['url']
        elif 'image' in sample:
            if isinstance(sample['image'], Image.Image):
                image = sample['image']
            else:
                raise NotImplementedError

            image_id = ''
            if 'name' in sample:
                image_id = sample['name']
        else:
            raise NotImplementedError

        return {
            'image': image,
            'id': image_id,
        }

    def sample(self, k=1):
        """
        """
        data_source = np.random.choice(self.huggingface_datasets, 1)[0]
        print(f"Sampling {k} real images from {data_source}...")

        dataset = self.sources[data_source]
        sampled_images = []
        while k > 0:
            attempts = len(dataset) // 2
            for _ in range(attempts):
                image_idx = np.random.randint(0, len(dataset))
                if data_source not in self.sampled_images_idx or image_idx not in self.sampled_images_idx[data_source]:
                    break
            try:
                image = self._get_image(data_source, image_idx)
                if image['image'] is not None:
                    sampled_images.append(image)
                    self.sampled_images_idx[data_source].append(image_idx)
                    k -= 1
            except Exception as e:
                print(e)
                continue

        return sampled_images
