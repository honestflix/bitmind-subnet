import torchvision.transforms as transforms


def CenterCrop():
    def fn(img):
        m = min(img.size)
        return transforms.CenterCrop(m)(img)
    return fn


# these transforms should match the tranforms used during your model's training,
# sans any data augmentation
transform = transforms.Compose([
    CenterCrop(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
])


def predict(model, image):
    image = transform(image).unsqueeze(0).float()
    out = model(image).sigmoid().flatten().tolist()
    return out[0]