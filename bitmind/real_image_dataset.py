from typing import List
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np
import requests
import base64


def download_image(url):
    print(f'downloading {url}')
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(type(response.content))
            return BytesIO(response.content)
        else:
            print(f"Failed to download image: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
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

    def sample_images(self, k=1, sampled_images=[], sampled_images_idx={}):
        """
        """
        if k == 0:
            return sampled_images

        data_source = np.random.choice(self.huggingface_datasets, 1)[0]
        bt.logging.info(f"Sampling {k} real images from {data_source}...")

        dataset = self.sources[data_source]
        while True:
            image_idx = np.random.randint(0, len(dataset))
            if image_idx not in sampled_images_idx[data_source]:
                break

        try:
            image = self._get_image(data_source, image_idx)
            sampled_images.append(image)
            sampled_images_idx[data_source].append(image_idx)
            k -= 1
        except Exception as e:
            print(e)

        return self.sample_images(k, sampled_images, sampled_images_idx)