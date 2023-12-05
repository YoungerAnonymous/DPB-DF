import os
import random
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import *
import pandas as pd
from sklearn import metrics

def get_dataloaders(
        root, 
        d_name, 
        validation=False, 
        train_val_split=1.0, 
        base_class=1,
        incre_class=1,
        batch_size=32,
        resize=256,
        imagesize=224,
        num_workers=0,
    ):
    if "mvtec" in d_name:
        if validation and train_val_split > 0 and train_val_split < 1:
            train_dataset = MVTecAD(root, True, False, train_val_split, resize, imagesize)
            test_dataset = MVTecAD(root, True, True, train_val_split, resize, imagesize)
        else:
            train_dataset = MVTecAD(root, training=True, resize=resize, imagesize=imagesize)
            test_dataset = MVTecAD(root, training=False, resize=resize, imagesize=imagesize)
    elif d_name == 'AITEX':
        if validation and train_val_split > 0 and train_val_split < 1:
            train_dataset = AITEX(root, True, False, train_val_split, resize, imagesize)
            test_dataset = AITEX(root, True, True, train_val_split, resize, imagesize)
        else:
            train_dataset = AITEX(root, training=True, resize=resize, imagesize=imagesize)
            test_dataset = AITEX(root, training=False, resize=resize, imagesize=imagesize)
    else:
        raise NotImplementedError()
    
    train_sampler = CILSampler(train_dataset, base_class, incre_class)
    test_sampler = CILSampler(test_dataset, base_class, incre_class, accumulate=True, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader, test_dataloader

def create_index(dimension: int, device, id: bool = True):
    if device.type == "cuda":
        resource = faiss.StandardGpuResources()
        resource.noTempMemory()
        index = faiss.GpuIndexFlatL2(
            resource, dimension, faiss.GpuIndexFlatConfig()
        )
        return faiss.IndexIDMap2(index)
    index = faiss.IndexFlatL2(dimension)
    return faiss.IndexIDMap2(index) if id else index

def patch_vote(candidates: np.ndarray, dim=-1, exception: int = None):
    """
    Choose the winner who has the most of votes.

    Args:
        candidates: [N x k] N candidates with k voter.
        dim: The dimension where voter are.
        exception: The class id which is required to get all the votes.
    """
    if candidates.shape[dim] == 1:
        return np.squeeze(candidates, dim)
    max_class_num = np.max(candidates) + 1
    count = []
    for i in range(max_class_num):
        i_col = np.sum((candidates == i).astype(float), dim)
        count.append(i_col)
    count = np.stack(count, dim)
    if exception is not None:
        max_votes = candidates.shape[dim]
        _count = np.take(count, exception, dim)
        _count[np.sum((candidates == exception).astype(int), axis=-1) < max_votes] = -1
        count = np.insert(np.delete(count, exception, dim), exception, _count, dim)  
    return np.argmax(count, dim)

def set_torch_device(gpu_ids):
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def save_results(save_dir, results: dict, method_name='Default', incre=False):
    save_path_1 = os.path.join(save_dir, "results.csv")
    fields = list(results.keys())
    if not os.path.exists(save_path_1):
        df = pd.DataFrame(columns=["Method", *fields]).set_index("Method")
    else:
        df = pd.read_csv(save_path_1, index_col=0)
    for i in fields:
        df.loc[method_name, i] = results[i][-1]
    df.to_csv(save_path_1)

    if incre:
        save_path_2 = os.path.join(save_dir, 'incremental_learning.csv')
        df = pd.DataFrame({"Session": range(len(results[fields[0]])), **results})
        df.to_csv(save_path_2, index=False)
