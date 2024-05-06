from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from torchvision import datasets, transforms
import numpy as np

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)

class Simple_CNN(nn.Module):
    def __init__(self, class_num, pretrain=False):
        super(Simple_CNN, self).__init__()
        nc = 3
        nf = 64
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(nf * 8, class_num, bias=True)
        )
        self.pretrain = pretrain

    def forward(self, input):
        embedding = self.main(input)
        feature = self.pool(embedding)
        feature = feature.view(feature.shape[0], -1)
        cls_out = self.classification_head(feature)
        if not self.pretrain:
            cls_out = F.softmax(cls_out)
        return cls_out, embedding

class SupConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, backbone, head='mlp', dim_in=512, feat_dim=128):
        super(SupConNet, self).__init__()
        self.backbone=backbone
        if head=='linear':
            self.head=nn.Linear(dim_in, feat_dim)
        elif head=='mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        cls_out, embedding = self.backbone(x)
        feat = self.backbone.pool(embedding)
        feat = feat.view(feat.shape[0], -1)
        feat = F.normalize(self.head(feat), dim=1)
        return cls_out, feat
    

# Define the loss function
def loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets)

# Define the optimizer
def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer

# Define the training loop
def train_model(model, model_name, train_loader, val_loader, epochs=10):
    # train_dl, val_dl = create_data_loader(train_ds, val_ds)
    optimizer = get_optimizer(model)
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
        model.eval()
        with torch.no_grad():
            val_loss = []
            for i, (images, labels) in enumerate(val_loader):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = loss_fn(outputs, labels)
                val_loss.append(loss.item())
            print(f'Epoch: {epoch}, Val Loss: {np.mean(val_loss)}')
    torch.save(model.state_dict(), f'mining_models/{model_name}.pt')
    return model

def create_dl():
    # Define transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.ToTensor(),         # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
    ])

    # Define paths to your dataset directories
    train_data_path = 'base_miner/data/train'
    test_data_path = 'base_miner/data/test'

    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

    # Define batch size for training and testing
    batch_size = 64

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check the length of the datasets and the number of batches
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    return train_loader, test_loader


if(__name__=='__main__'):
    train_loader, test_loader = create_dl()    
    netE = Simple_CNN(2)
    model = SupConNet(netE)
    print(model)    
    base_model_history = train_model(model, 'torch_model', train_loader, test_loader, epochs=5)
