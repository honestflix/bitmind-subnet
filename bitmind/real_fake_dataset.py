import numpy as np


class RealFakeDataset:

    def __init__(
        self,
        real_image_dataset,
        fake_image_dataset,
        image_generator=None,
        transforms=None
    ):

        self.real_image_dataset = real_image_dataset
        self.fake_image_dataset = fake_image_dataset
        self.image_generator = image_generator
        self.transforms = transforms

    def __getitem__(self, item):

        source = self.real_image_dataset
        label = 0.
        if np.random.rand() > .5:
            source = self.fake_image_dataset
            label = 1.

        image = source.sample(1)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return min(len(self.real_image_dataset), len(self.fake_image_dataset))









