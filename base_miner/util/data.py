import torchvision.transforms as transforms
from bitmind.real_fake_dataset import RealFakeDataset
from bitmind.image_dataset import ImageDataset
from bitmind.data_sources import DATASET_META


def load_datasets():
    """
    
    Returns:

    """
    splits = ['train', 'validation', 'test']

    fake_datasets = {split: [] for split in splits}
    for split in splits:
        for dataset_meta in DATASET_META['fake']:
            dataset = ImageDataset(dataset_meta['name'], split, dataset_meta['create_splits'])
            fake_datasets[split].append(dataset)
            print(f"Loaded {dataset_meta['name']}[{split}], len={len(dataset)}")

    real_datasets = {split: [] for split in splits}
    for split in splits:
        for dataset_meta in DATASET_META['real']:
            dataset = ImageDataset(dataset_meta['name'], split, dataset_meta['create_splits'])
            real_datasets[split].append(dataset)
            print(f"Loaded {dataset_meta['name']}[{split}], len={len(dataset)}")

    return real_datasets, fake_datasets


def create_real_fake_datasets(real_datasets, fake_datasets):
    """

    Args:
        real_datasets:
        fake_datasets:

    Returns:

    """
    def CenterCrop():
        def fn(img):
            m = min(img.size)
            return transforms.CenterCrop(m)(img)

        return fn

    transform = transforms.Compose([
        CenterCrop(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
    ])

    train_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['train'],
        fake_image_datasets=fake_datasets['train'],
        transforms=transform)

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        transforms=transform)

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        transforms=transform)

    return train_dataset, val_dataset, test_dataset
