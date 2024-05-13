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
    #print(f'downloading {url}')
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)

    else:
        #print(f"Failed to download image: {response.status_code}")
        return None


def load_huggingface_dataset(name):
    if 'imagefolder' in name:
        _, directory = name.split(':')
        return load_dataset(path='imagefolder', data_dir=directory, split='train')
    else:
        return load_dataset(name, split='validation')


class RealImageDataset:

    def __init__(
            self,
            huggingface_dataset_names: List[str]=['dalle-mini/open-images']
    ):
        self.huggingface_dataset_names = huggingface_dataset_names
        self.data_sources = {
            name: load_huggingface_dataset(name)
            for name in huggingface_dataset_names
        }
        self.sampled_images_idx = defaultdict(list)

    def __getitem__(self, index):
        # TODO get from multiple source options
        source = np.random.choice(self.huggingface_dataset_names, 1)[0]
        return self._get_image(source, index)
    def __len__(self):
        # todo factor in multidataset
        return len(self.data_sources[self.huggingface_dataset_names[0]])

    def _get_image(self, data_source, index):
        """

        """
        sample = self.data_sources[data_source][int(index)]
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

        # check for/remove alpha channel if download didnt 404
        if image is not None and 'A' in image.mode:
            image = image.convert('RGB')

        return {
            'image': image,
            'id': image_id,
        }

    def sample(self, k=1):
        """
        """
        data_source = np.random.choice(self.huggingface_dataset_names, 1)[0]
        #print(f"Sampling {k} real images from {data_source}...")

        dataset = self.data_sources[data_source]
        sampled_images = []
        while k > 0:
            attempts = len(dataset) // 2
            for i in range(attempts):
                image_idx = np.random.randint(0, len(dataset))
                if data_source not in self.sampled_images_idx or image_idx not in self.sampled_images_idx[data_source]:
                    break
                if i >= attempts:
                    self.sampled_images_idx[data_source] = []
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
