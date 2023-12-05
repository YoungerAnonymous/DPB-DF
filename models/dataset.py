import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MVTecAD(Dataset):
    def __init__(self, root, training=True, validation=False, train_val_split=1.0, resize=256, imagesize=224, ret_masks=False):
        self.root = root
        self.training = training
        self.ret_masks = ret_masks
        self.imagesize = imagesize
        images_dir = os.path.join(root, "train" if training else "test")
        masks_dir = os.path.join(root, "ground_truth")
        self.categories = ["good"] + sorted(filter(lambda x: x != 'good' and not x.startswith('.'), os.listdir(images_dir)))
        self.data = []
        self.targets = []
        self.masks = []
        self.img_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor()
        ])
        self.n_samples = []
        for i, category in enumerate(self.categories):
            category_dir = os.path.join(images_dir, category)
            images = [os.path.join(category_dir, image) for image in sorted(filter(lambda x: not x.startswith('.'), os.listdir(category_dir)))]
            if training and train_val_split > 0 and train_val_split < 1:
                split = int(len(images) * train_val_split)
                images = images[:split] if not validation else images[split:]
                assert len(images) > 0
            self.data += images
            self.targets += [i] * len(images)
            self.n_samples.append(len(images))
            if category == 'good':
                masks = [None] * len(images)
            else:
                masks = [os.path.join(masks_dir, category, self.get_mask_name(os.path.basename(image))) for image in images]
            self.masks += masks
        
    @staticmethod
    def get_mask_name(imagename):
        number, ext = os.path.splitext(imagename)
        return number + '_mask' + ext
    
    def __getitem__(self, index):
        x = Image.open(self.data[index]).convert("RGB")
        x = self.img_transforms(x)
        y = self.targets[index]
        if not self.ret_masks:
            return x, y
        if self.masks[index] is not None:
            m = self.mask_transforms(Image.open(self.masks[index]))
        else:
            m = torch.zeros([1, self.imagesize, self.imagesize])
        return x, y, m

    def __len__(self):
        return len(self.data)


class AITEX(MVTecAD):
    def __init__(self, root, training=True, validation=False, train_val_split=1.0, resize=256, imagesize=224, ret_masks=False):
        super().__init__(root, training, validation, train_val_split, resize, imagesize, ret_masks)


class CILSampler(Sampler):
    def __init__(self, dataset, base_class, incre_class, accumulate=False, shuffle=True, init_task=0):
        self.targets = np.array(dataset.targets)
        self.task = init_task
        self.base_class = base_class
        self.incre_class = incre_class
        self.accumulate = accumulate
        self.shuffle =shuffle
        self.classes = np.unique(self.targets)
        self.n_classes = self.classes.shape[0]
        assert (self.n_classes - self.base_class) % self.incre_class == 0
        self.n_tasks = (self.n_classes - self.base_class) // self.incre_class + 1
        self.samples = []
        for c in self.classes:
            self.samples.append(np.squeeze(np.argwhere(self.targets == c), axis=1))

    def set_task(self, task):
        self.task = task
    
    def get_class_range(self):
        if self.task == 0:
            class_index = range(self.base_class)
        else:
            if self.accumulate:
                class_index = range(self.base_class + self.task * self.incre_class)
            else:
                class_index = range(self.base_class + (self.task - 1) * self.incre_class, self.base_class + self.task * self.incre_class)
        return class_index

    def __iter__(self):
        samples = []
        for i in self.get_class_range():
            samples += self.samples[i].tolist()
        if self.shuffle:
            np.random.shuffle(samples)
        for sample in samples:
            yield sample
    
    def __len__(self):
        n_samples = 0

        for i in self.get_class_range():
            n_samples += self.samples[i].shape[0]
        return n_samples
