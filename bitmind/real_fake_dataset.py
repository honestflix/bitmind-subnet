import numpy as np


class RealFakeDataset:

    def __init__(
        self,
        real_image_datasets,
        fake_image_datasets,
        image_generator=None,
        transforms=None
    ):

        self.real_image_datasets = real_image_datasets
        self.fake_image_datasets = fake_image_datasets
        self.image_generator = image_generator
        self.transforms = transforms

        self._history = {
            'source': [],
            'index': [],
            'label': [],
            'image': []
        }

    def __getitem__(self, index):

        if len(self._history['index']) > index:
            label = self._history['label'][index]
            source = self._history['source'][index]
            image = self._history['image'][index]
            #image = source[index]['image']
        else:
            if np.random.rand() > .5:
                source = self.fake_image_datasets[np.random.randint(0, len(self.fake_image_datasets))]
                image = source[index]['image']
                label = 1.
            else:
                source = self.real_image_datasets[np.random.randint(0, len(self.real_image_datasets))]
                #image = source.sample(1)[0]['image']
                imgs, idx = source.sample(1)
                image = imgs[0]['image']
                index = idx[0]
                label = 0.
                if image is None:
                    print('NONE', source, source.huggingface_dataset_name, label, index)

            self._history['source'].append(source)
            self._history['label'].append(label)
            self._history['index'].append(index)
            self._history['image'].append(image)

        #print(source.huggingface_dataset_name, index, label)
        #image = source[index]['image']
        try:
            if self.transforms is not None:
                image = self.transforms(image)
        except Exception as e:
            print(e)
            print(source.huggingface_dataset_name, label, index)
            print(image)
 

        return image, label


    def __len__(self):
        real_dataset_min = min([len(ds) for ds in self.real_image_datasets])
        fake_dataset_min = min([len(ds) for ds in self.fake_image_datasets])
        return min(fake_dataset_min, real_dataset_min)
