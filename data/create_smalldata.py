from data_loader import ImagerLoader
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

import numpy as np

import time

# TODO: Set this as command line args
batch_size = 16
workers = 8
total_ingredients = 30167 # Found by loading vocab.bin

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
        ImagerLoader("./data/images/",
            transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256), # we get only the center of that rescaled
            transforms.RandomCrop(224), # random crop within the center crop 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), data_path="./data/lmdbs/", partition='train', sem_reg=None),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        ImagerLoader("./data/images/",
            transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ]), data_path="./data/lmdbs/", partition='val', sem_reg=None),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)


num_batches = len(train_loader)
data = []

for i, (input, target) in enumerate(train_loader):
    start = time.time()
    # input has 5 components
    # x = img
    # y1 = recipe step data  / torch.Size([1, 20, 1024]) ()
    # y2 = number of steps / torch.Size([1])
    # z1 = ingredient id / torch.Size([1, 20])
    # z2 = number of ingredients / torch.Size([1])

    img_tensor = input[0] # torch.Size([1, 3, 224, 224])
    current_batch_size = int(img_tensor.size()[0])

    num_ingredients = int(input[4].data[0]) # torch.Size([1]) (number of ingredients?)
    ingredient_idx = input[3] # torch.Size([1, 20]) (ingredient id?)

    # Create one-hot vectors (multilabel binarization - ingredients present will have 1 at their indices)
    labels = torch.zeros((current_batch_size, total_ingredients))
    for batch in range(current_batch_size):
        for j in range(num_ingredients):
            labels[batch, int(ingredient_idx[batch, j])] = 1
        
        data.append([img_tensor[batch,:].numpy(), labels[batch, :].numpy()])
        assert(data[-1][0].shape == data[0][0].shape)
        assert(data[-1][1].shape == data[0][1].shape)

    if len(data) >= 100:
        print("Saving smalldata...")
        data_arr = np.array(data)
        np.save("smalldata.npy", data_arr)
        break

    # print(img_tensor.size(), labels.size()) # torch.Size([batch_size, 3, 224, 224]) torch.Size([batch_size, 30167])
    time_taken = time.time() - start
    print(f"Batch {i} / {num_batches}, Time taken: {time_taken}s")

print("Done")