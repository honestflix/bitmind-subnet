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

    def __getitem__(self, index):

        if np.random.rand() > .5:
            source = self.fake_image_dataset
            image = source[index]['image']
            label = 1.
        else:
            source = self.real_image_dataset
            image = source.sample(1)[0]['image']
            label = 0.

        #image = source[index]['image']
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return min(len(self.real_image_dataset), len(self.fake_image_dataset))









