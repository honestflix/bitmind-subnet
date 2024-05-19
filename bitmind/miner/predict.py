import torchvision.transforms as transforms


def CenterCrop():
    def fn(img):
        m = min(img.size)
        return transforms.CenterCrop(m)(img)
    return fn

transform = transforms.Compose([
    CenterCrop(),
    #transforms.Lambda(lambda img: CenterCrop()(img)),
    #transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
    #transforms.Lambda(lambda t: t.float() / 255.),
    #transforms.Normalize( mean=MEAN['imagenet'], std=STD['imagenet'] ),
])


def predict(model, image):
    image = transform(image).unsqueeze(0).float()
    out = model(image).sigmoid().flatten().tolist()
    return out[0]