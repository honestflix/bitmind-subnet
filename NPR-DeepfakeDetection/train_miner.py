import os
import sys
import time
import random
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from util import Logger

from bitmind.real_fake_dataset import RealFakeDataset
from bitmind.real_image_dataset import RealImageDataset
#from bitmind.random_image_generator import RandomImageGenerator
import torchvision.transforms as transforms
import torch


def load_datasets():
    splits = ['train', 'validation', 'test']
    real_datasets = {split: [] for split in splits}
    real_image_dataset_meta = [
        {'name': 'dalle-mini/open-images', 'create_splits': False},
        {'name': 'merkol/ffhq-256', 'create_splits': True},
        {'name': 'jlbaker361/flickr_humans_20k', 'create_splits': True},
        {'name': 'saitsharipov/CelebA-HQ', 'create_splits': True}
    ]
    
    for split in splits:
        for dataset_meta in real_image_dataset_meta:
            dataset = RealImageDataset(dataset_meta['name'], split, dataset_meta['create_splits'])
            real_datasets[split].append(dataset)
            print(f"Loaded {dataset_meta['name']}[{split}], len={len(dataset)}")
    
    
    fake_datasets = {split: [] for split in splits}
    fake_image_dataset_meta = [
        {'name': 'imagefolder:../bitmind/data/RealVisXL_V.40', 'create_splits': False}
    ]
    
    for split in splits:
        for dataset_meta in fake_image_dataset_meta:
            dataset = RealImageDataset(dataset_meta['name'] + '/' + split, None, dataset_meta['create_splits'])
            fake_datasets[split].append(dataset)
            print(f"Loaded {dataset_meta['name']}[{split}], len={len(dataset)}")

    return real_datasets, fake_datasets

def create_real_fake_datasets(real_datasets, fake_datasets):
    MEAN = {
        "imagenet":[0.485, 0.456, 0.406],
        "clip":[0.48145466, 0.4578275, 0.40821073]
    }

    STD = {
        "imagenet":[0.229, 0.224, 0.225],
        "clip":[0.26862954, 0.26130258, 0.27577711]
    }

    def CenterCrop():
        def fn(img):
            m = min(img.size)
            return transforms.CenterCrop(m)(img)
        return fn
    
    transform = transforms.Compose([
        CenterCrop(),
        #transforms.CenterCrop(224),
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


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=True)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


def main():
    opt = TrainOptions().parse()
    seed_torch(100)
    
    #Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    real_datasets, fake_datasets = load_datasets()
    train_dataset, val_dataset, test_dataset = create_real_fake_datasets(real_datasets, fake_datasets)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=lambda d: tuple(d))
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=lambda d: tuple(d))
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=lambda d: tuple(d))

    model = Trainer(opt)

    early_stopping_epochs = 10
    best_val_acc = 0
    n_epoch_since_improvement = 0
    model.train()

    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
    
        for i, data in enumerate(train_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
    
            model.set_input(data)
            model.optimize_parameters()
    
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
        
            if model.total_steps % opt.loss_freq == 0:
                train_writer.add_scalar('loss', model.loss, model.total_steps)
            
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
    
        # Validation
        model.eval()
        acc, ap = validate(model.model, val_loader)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)

        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        if acc > best_val_acc:
            model.save_networks('best')
            best_val_acc = acc
        else:
            n_epoch_since_improvement += 1
            if n_epoch_since_improvement >= early_stopping_epochs:
                break

        model.train()
    
    model.eval()
    acc, ap = validate(model.model, test_loader)[:2]
    print("(Test) acc: {}; ap: {}".format(acc, ap))
    model.save_networks('last')


if __name__ == '__main__':
    main()


