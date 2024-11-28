import os
import time
import yaml
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms 

from A2C import A2C
from Network import *
from config import get_arg
from Dataset import Dataset_online, CIFAR100Dataset
from normalization import batch_Normalization

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

args = get_arg()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
cuda = True if torch.cuda.is_available() else False

with open(args.conf) as f:
    cfg = yaml.safe_load(f)

best_acc = 0
best_epoch = 0
acc_list = []
momentum = args.momentum
epoches = cfg['epoch']
batch = cfg['batch']

args.max_step = epoches


def setup_seed(seed):
    if seed == None:
        seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(args.seed)

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])

transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                ])

if args.dataset == 'CIFAR10':
    trainset = Dataset_online(args.dataset_path, train=True, transform=transform, aug=args.aug)
    testset = Dataset_online(args.dataset_path, train=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    trainset = CIFAR100Dataset(args.dataset_path, train=True, fine_label=True, transform=transform, aug=args.aug)
    testset = CIFAR100Dataset(args.dataset_path, train=False, fine_label=True, transform=transform_test)

train_loader=DataLoader(dataset=trainset, batch_size=cfg['batch'], shuffle=True, num_workers=8, pin_memory=True)
test_loader=DataLoader(dataset=testset, batch_size=cfg['batch'], shuffle=False, num_workers=8, pin_memory=True)
start_epoch = 0

model = get_model(cfg['model']['type'], num_classes=num_class(args.dataset.lower()))

if len(args.gpus) > 1:
    model = torch.nn.DataParallel(model, device_ids=np.arange(len(args.gpus.split(','))).tolist()).cuda()

agent = A2C(args)

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}/{}'.format(args.dataset, args.resume))
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if cfg['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['lr'],
            momentum=momentum,
            weight_decay=cfg['optimizer']['decay'],
            nesterov=cfg['optimizer']['nesterov']
        )

lr_schduler_type = cfg['lr_schedule']['type']

if lr_schduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'], eta_min=0.)
elif lr_schduler_type == 'step':
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr_schedule']['milestones'],gamma=cfg['lr_schedule']['gamma'])

if cfg['lr_schedule']['warmup'] != '' and cfg['lr_schedule']['warmup']['epoch'] > 0:
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier = cfg['lr_schedule']['warmup']['multiplier'],
        total_epoch = cfg['lr_schedule']['warmup']['epoch'],
        after_scheduler = scheduler
    )

loss = nn.CrossEntropyLoss(reduction='none')
balance_factor = 1.

if args.use_reward_norm:
    reward_norm = batch_Normalization()

def train_first_epoch(net, agent):
    net.eval()
    for i, data in enumerate(train_loader, 0):
        idx, inputs, labels = data

        inputs, labels = inputs.cuda(), labels.cuda()
        outputs, feature = net(inputs, labels)
        state = torch.squeeze(feature.detach())
        magnitude = agent.action(state)
        magnitude = magnitude.squeeze()

        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())


def train(net, agent, epoch):
    global optimizer
    global scheduler
    global loss
    global balance_factor

    net.train()
    balance_factor = balance_factor - 1 / epoches
    print('balance_factor:{}'.format(balance_factor))
    training_loss=0.0
    training_none_loss = 0.0
    training_full_loss = 0.0
    training_magnitude=0.0
    total = len(train_loader.dataset)
    correct = 0

    total_reward = 0
    for i, data in enumerate(train_loader, 0):
        idx, none_aug_inputs, ada_aug_inputs, full_aug_inputs, labels = data
        none_aug_inputs, ada_aug_inputs, full_aug_inputs, labels = none_aug_inputs.cuda(), ada_aug_inputs.cuda(), full_aug_inputs.cuda(), labels.cuda()
        
        ada_outputs, ada_feature = net(ada_aug_inputs, labels)
        with torch.no_grad():
            none_outputs, none_feature = net(none_aug_inputs, labels)
            full_outputs, full_feature = net(full_aug_inputs, labels)
        
        none_state = torch.squeeze(none_feature.detach())
        ada_state = torch.squeeze(ada_feature.detach())

        ada_loss = loss(ada_outputs, labels)
        none_loss = loss(none_outputs, labels)
        full_loss = loss(full_outputs, labels)

        magnitude = agent.action(none_state)

        magnitude = magnitude.squeeze()
        trainset.set_MAGNITUDE(idx, magnitude.detach().cpu())

        reward = (1 - balance_factor) * (ada_loss.detach() - none_loss.detach()) + \
                balance_factor * (full_loss.detach() - ada_loss.detach())

        if args.use_reward_norm:
            reward = reward_norm(reward)
        total_reward += reward
        
        optimizer.zero_grad()
        ada_loss = ada_loss.mean()
        ada_loss.backward()
        optimizer.step()
        agent.update(none_state, magnitude, ada_state, reward)
            
        training_loss += ada_loss.item()
        training_none_loss += none_loss.mean().item()
        training_full_loss += full_loss.mean().item()
        training_magnitude += magnitude.mean().item()
        predicted = ada_outputs.max(1).indices
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            loss_mean = training_loss / (i + 1)
            magnitude_mean = training_magnitude / (i + 1)
            trained_total = (i + 1) * len(labels)
            acc = 100. * correct/trained_total
            progress = 100. * trained_total/total
            aver_reward = total_reward / (i + 1)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.3f} Acc: {:.2f} Mag: {:.2f}  Average Reward: {:.6f}'.format(epoch,
                trained_total, total, progress, loss_mean, acc, magnitude_mean, aver_reward))
    
def test(net, epoch):
    global best_acc
    global best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs, targets)
            predicted = outputs.max(1).indices
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct * 100. /total
    print('EPOCH:{}, ======================ACC:{}===================='.format(epoch, acc))
    acc_list.append(acc)
    if acc>=best_acc:
        best_acc = acc
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if args.save_model:
            torch.save(state, './checkpoint/{}/{}_{}.pth'.format(args.dataset,cfg['model']['type'],args.seed))
    print('BEST EPOCH:{},BEST ACC:{}'.format(best_epoch, best_acc))


if __name__ =='__main__':
    train_first_epoch(model, agent)
    trainset.is_magnitude = True
    for epoch in tqdm(range(start_epoch, epoches)):
        train(model, agent, epoch)
        agent.lr_decay(epoch)
        test(model, epoch)
        scheduler.step()