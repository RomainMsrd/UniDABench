import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import os, sys
import numpy as np
import random


class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs, encoder):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        self.return_index = dataset_configs.da_method in ["DANCE", "UniOT", "UniJDOT", "UniJDOT_THR_NOJOINT", "UniJDOT_NoJoint", "UniJDOT_THR"]

        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = np.array(dataset["labels"]).copy()
        #print(len(y_data))

        #Extend Encoder if necessary (new classes)
        if not encoder is None:
            self.encoder = encoder.copy()
            #diff = len(np.unique(y_data)) - len(encoder)
            diff = abs((max(y_data)+1) - len(self.encoder))# - len(encoder)
            if diff > 0:
                print("Private Target Classes Detected")
                self.encoder = np.concatenate([self.encoder, np.zeros(diff, dtype=int) - 1])
            for gt in np.unique(y_data):
                #print("gt : ", max(self.encoder) + 1 if self.encoder[gt] == -1 else self.encoder[gt])
                self.encoder[gt] = max(self.encoder) + 1 if self.encoder[gt] == -1 else self.encoder[gt]
            #Encode Labels
            y_data = self.encoder[y_data]
            #print(self.encoder)
            """self.decoder = self.encoder.copy()
            self.decoder = {v: k for i, v in enumerate(self.encoder)}"""
            self.decoder = self.encoder.copy()
            for i, k in enumerate(self.encoder):
                self.decoder[k] = i

        #print(np.unique(y_data))
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        if self.return_index:
            return x, y, index
        return x, y

    def __len__(self):
        return self.len

#TODO: Remove print
def get_label_encoder(data_path, domain_id, dataset_configs, pri_cl, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))
    print(dataset_file['labels'].shape, dataset_file['samples'].shape)

    if len(pri_cl) != 0:
        dataset_file = remove_private_class(dataset_file, pri_cl)
    print(type(dataset_file['labels']), type(torch.zeros(5)))
    if type(dataset_file['labels']) == type(torch.zeros(5)):
        dataset_file["labels"] = dataset_file["labels"].numpy().astype(int)
    else:
        dataset_file["labels"] = dataset_file["labels"].astype(int)
    uni = np.unique(dataset_file["labels"])
    print(uni)
    #print("Uni : ", uni)
    if len(uni) != dataset_configs.num_classes:
        print("Private Classes Detected in Source")
        dataset_configs.num_classes = len(uni)
    encoder = np.zeros(max(uni) + 1, dtype=int) - 1
    for i, k in enumerate(uni):
        encoder[k] = i

    return encoder


def remove_private_class(dataset_file, private_class):
    mask = np.isin(dataset_file['labels'], private_class, invert=True)
    dataset_file['labels'] = dataset_file['labels'][mask]
    dataset_file['samples'] = dataset_file['samples'][mask]
    assert len(mask) != len(dataset_file['labels'])
    return dataset_file


def data_generator(data_path, domain_id, dataset_configs, hparams, encoder, pri_cl, dtype, src=False):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))
    if type(dataset_file['labels']) == type(torch.zeros(5)):
        dataset_file["labels"] = dataset_file["labels"].numpy().astype(int)
    else:
        dataset_file["labels"] = dataset_file["labels"].astype(int)

    if len(pri_cl) != 0:
        dataset_file = remove_private_class(dataset_file, pri_cl)

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs, encoder)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    if dataset_configs.src_balanced and dtype != "test" and src:
        print("Source Mini-Batch is Balanced during training !")
        class_sample_count = np.array(
            [len(np.where(dataset.y_data == t)[0]) for t in np.unique(dataset.y_data)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in dataset.y_data])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=hparams["batch_size"],
                                                  drop_last=drop_last,
                                                  sampler=sampler,
                                                  num_workers=4)
        return data_loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=4)
    return data_loader


def data_generator_old(data_path, domain_id, dataset_configs, hparams):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, dataset_configs)
    test_dataset = Load_Dataset(test_dataset, dataset_configs)

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader


#TODO : Remove
def few_shot_data_generator(data_loader, dataset_configs, encoder, num_samples=5):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    NUM_SAMPLES_PER_CLASS = num_samples
    NUM_CLASSES = len(torch.unique(y_data))

    counts = [y_data.eq(i).sum().item() for i in range(NUM_CLASSES)]
    samples_count_dict = {i: min(counts[i], NUM_SAMPLES_PER_CLASS) for i in range(NUM_CLASSES)}

    samples_ids = {i: torch.where(y_data == i)[0] for i in range(NUM_CLASSES)}
    selected_ids = {i: torch.randperm(samples_ids[i].size(0))[:samples_count_dict[i]] for i in range(NUM_CLASSES)}

    selected_x = torch.cat([x_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)
    selected_y = torch.cat([y_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    if not encoder is None:
        encoder = np.arange(len(encoder))
    few_shot_dataset = Load_Dataset(few_shot_dataset, dataset_configs, encoder)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)

    return few_shot_loader

