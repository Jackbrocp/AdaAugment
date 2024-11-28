from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import pickle
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from scipy.stats import entropy
import torch.nn.functional as F
from augmentation import CIFAR10Policy_magnitude
from organize_transform import make_transform, make_magnitude_transform_AA, make_magnitude_transform_RA, make_magnitude_transform_TA

class Dataset_online(Dataset):
    base_folder = ''
    preix = 'adv_'
    train_list=[[preix+'data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                [preix+'data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                [preix+'data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                [preix+'data_batch_4', '634d18415352ddfa80567beed471001a'],
                [preix+'data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],]
    test_list = [['test_batch', '40351d587109b95175f43aff81a1287e'],]
    meta = {'filename': 'batches.meta',
            'key': 'label_names',
            'md5': '5ff9c542aee3614f3951f8cda6e48888',}
    transform_list = []
    def __init__(self, root, train=True, transform=None, target_transform=None, aug=None):
        super(Dataset_online, self).__init__()
        self.train = train
        self.root = root
        self.data = []
        self.targets = []
        self.filename_list = []
        self.transform = transform
        self.target_transform = target_transform
        if aug == 'autoaugment':
            self.make_magnitude_transform = make_magnitude_transform_AA
        elif aug == 'trivialaugment':
            self.make_magnitude_transform = make_magnitude_transform_TA
        else:
            self.make_magnitude_transform = make_magnitude_transform_RA

        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if self.train:
                    self.filename_list.extend(entry['filenames'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self._load_meta()
        self.MAGNITUDE = torch.zeros(50000)
        self.is_magnitude = False
        self.full_transform = self.make_magnitude_transform(magnitude=1, cutout_length=16)
        self.none_transform = self.transform
        self.warmup_transform = make_transform('cifar10', aug, length=16)[0]

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        """if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')"""
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def set_transform(self, magnitude, cutout_length):
        transform_list = []
        for i in range(len(magnitude)):
            transform_list.append(self.make_magnitude_transform(magnitude=magnitude, cutout_length=cutout_length)) 
        self.transform_list = transform_list
        return 
    
    def set_MAGNITUDE(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude

    def __getitem__(self, index:int):
        if self.train:
            img, target = self.data[index], int(self.targets[index]) 
        else:
            img, target = self.data[index], int(self.targets[index])
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            if self.is_magnitude:
                t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=16)
                normalized_img = t(img)
                full_aug_img = self.full_transform(img)
                none_aug_img = self.none_transform(img)
                return index, none_aug_img, normalized_img, full_aug_img, target
            else:   
                normalized_img = self.warmup_transform(img)
            return index, normalized_img, target 
        else:
            normalized_img = self.transform(img)
            return normalized_img, target
        
    def __len__(self):
        return len(self.data)
    
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class Dataset_online_faster(Dataset):
    # base_folder='cifar-10-batches-py'
    base_folder='advMask_Dataset'
    preix='adv_'
    train_list=[
        [preix+'data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        [preix+'data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        [preix+'data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        [preix+'data_batch_4', '634d18415352ddfa80567beed471001a'],
        [preix+'data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list=[
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(self,root,train=True,ori_transform=None,transform=None, target_transform=None, pointlist=None):
        super(Dataset_online_faster,self).__init__()
        self.train=train
        self.root=root
        self.data = []
        self.targets=[]
        self.filename_list=[]
        self.ori_transform = ori_transform
        self.transform=transform
        self.target_transform=target_transform
        if self.train:
            data_list = self.train_list
            self.pointlist = pointlist
        else:
            data_list = self.test_list
        
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if self.train:
                    self.filename_list.extend(entry['filenames'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self._load_meta()
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        """if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')"""
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    
    def __getitem__(self, index:int):
        if self.train:
            img, target, filename = self.data[index], int(self.targets[index]), self.filename_list[index]
        else:
            img, target = self.data[index], int(self.targets[index])
        # img =  Image.fromarray((img*255).astype(np.uint8))
        if self.ori_transform is not None:
            ori_img = self.ori_transform(img)
        if self.transform is not None :
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            if self.ori_transform is None:
                return img, target, filename
            else:
                return ori_img, img, target, filename
        else:
            return img, target
    def __len__(self):
        return len(self.data)
    
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
 
 
class MyDataset(Dataset):
    # base_folder='cifar-10-batches-py'
    base_folder='myDataset'
    preix='aug_'

    train_list=[
        ['data_'+preix+'batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_'+preix+'batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_'+preix+'batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_'+preix+'batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_'+preix+'batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
  
    test_list=[
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(self,root,train=True,transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        self.train=train
        self.root=root
        self.data = []
        self.targets=[]
        self.filename_list=[]

        self.transform=transform
        self.target_transform=target_transform
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if self.train:
                    self.filename_list.extend(entry['filenames'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))            
        self._load_meta()
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        """if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')"""
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    
    def __getitem__(self, index:int):
        # img, target, filename = self.data[index], self.targets[index], self.filename_list[index]
        img, target = self.data[index], self.targets[index]
        # img =  Image.fromarray((img*255).astype(np.uint8))
        if self.transform is not None :
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target 
    def __len__(self):
        return len(self.data)
    
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

 
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, fine_label=True, transform=None, aug=None):
        if train:
            self.data,self.labels,self.filename_list=load_CIFAR_100(root,train,fine_label=fine_label)
        else:
            self.data,self.labels = load_CIFAR_100(root,train,fine_label=fine_label)
        self.transform = transform
        self.train = train
        self.is_magnitude = False
        if aug == 'autoaugment':
            self.make_magnitude_transform = make_magnitude_transform_AA
        else:# aug == 'trivialaugment':
            self.make_magnitude_transform = make_magnitude_transform_TA
        self.MAGNITUDE = torch.zeros(50000)
        self.full_transform = self.make_magnitude_transform(magnitude=1, cutout_length=8)
        self.transform = self.transform
        self.none_transform = self.transform
        self.warmup_transform, self.test_transform = make_transform('cifar100', aug, length=8)
        print('warmup_transform',self.warmup_transform)
    def set_MAGNITUDE(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], int(self.labels[index]) 
            if self.is_magnitude:
                t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=8)
                # t = self.warmup_transform
                normalized_img = t(img)
                full_aug_img = self.full_transform(img)
                none_aug_img = self.none_transform(img)
                return index, none_aug_img, normalized_img, full_aug_img, target
            else:
                normalized_img = self.warmup_transform(img)
                return index, normalized_img, target
        else:
            img, target = self.data[index], int(self.labels[index])
            normalized_img = self.test_transform(img)
            return normalized_img, target
    def __len__(self):
        return len(self.data)

def load_CIFAR_100(root, train=True, fine_label=True):
    if train:
        filename = root + 'my_train'
    else:
        filename = root + 'test'
 
    with open(filename, 'rb')as f:
        datadict = pickle.load(f,encoding='bytes')
 
        if train:
            # [50000, 32, 32, 3]
            X = datadict['data']
            filename_list = datadict['filenames']
            X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict['labels']
            Y = np.array(Y)
            return X, Y, filename_list
        else:
            # [10000, 32, 32, 3]
            X = datadict[b'data']
            filename_list = datadict[b'filenames']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict[b'fine_labels']
            Y = np.array(Y)
            return X, Y
 