import torchvision.transforms as transforms
import torch


random_image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    transforms.RandomRotation(20)
])
