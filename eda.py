import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import pickle
import numpy as np
from data import get_tensor_from_data, get_class_mapping
import time
import pickle
from data_storage_class import result_storage
import pathlib

import json
from tqdm import tqdm

from sklearn.metrics import f1_score

# TODO: Set this as command line args
batch_size = 32
workers = 16 # How many cores to use to load data
dev_mode = False # Set this to False when training on Athena
data_dir = "./dataset"
snapshots_dir = "./snapshots"
pathlib.Path(snapshots_dir).mkdir(exist_ok=True) # Create snapshot directory if it doesn't exist

#############################
# TOGGLES HERE
learning_rate = 0.01
momentum_mod = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class_mapping = get_class_mapping()
num_classes = len(class_mapping)
total_epochs = 10
#############################

if dev_mode:
    # Load from .npy file and batch manually
    data_arr = np.load(f"{data_dir}/smalldata.npy") # shape: (112, 2)
    num_batches = int(np.ceil(data_arr.shape[0] / batch_size))
    img_split = np.array_split(data_arr[:,0], num_batches)
    label_split = np.array_split(data_arr[:,1], num_batches)
    train_loader = [(img_split[i], label_split[i]) for i in range(num_batches)]
    val_loader = train_loader # you want to overfit anyway for testing.
    test_loader = val_loader    
else:    
    from data_loader import ImagerLoader
    train_loader = torch.utils.data.DataLoader(
            ImagerLoader(f"{data_dir}/images/",
                transforms.Compose([
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(256), # we get only the center of that rescaled
                transforms.RandomCrop(224), # random crop within the center crop 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ]), data_path=f"{data_dir}/lmdbs/", partition='train', sem_reg=None),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            ImagerLoader(f"{data_dir}/images/",
                transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ]), data_path=f"{data_dir}/lmdbs/", partition='val', sem_reg=None),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            ImagerLoader(f"{data_dir}/images/",
                transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ]), data_path=f"{data_dir}/lmdbs/", partition='test', sem_reg=None),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

#####################################################################
num_batches = len(train_loader) # now we can get your batches since dataset is chosen

#initialise your model here
target_model = torchvision.models.resnet50(pretrained=True)
nf = target_model.fc.in_features
target_model.fc = torch.nn.Linear(nf, num_classes)

optimiser = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum_mod)  # TOGGLES HERE.
target_model = target_model.to(device)
storage = result_storage(False,batch_size,num_classes,num_batches)
test_result = result_storage(True,batch_size,num_classes,num_batches)

ingredient_mapping = {}
with open('dataset/det_ingrs.json') as f:
    recipe_list = json.load(f)

for recipe in recipe_list:
    ingredients = [x["text"] for x in recipe["ingredients"]]
    ingredient_mapping[recipe["id"]] = ingredients

np.save("ingredient_mapping.npy", ingredient_mapping)
###########################################################

best_f1 = 0

ingredient_labels = []
ingredients = []

for i, (input, target) in enumerate(train_loader):
    start = time.time()
    img_tensor, labels, recipe_ids = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)

    for j in range(int(labels.size()[0])):
        nonzero = np.nonzero(labels[j].cpu().numpy())
        ingredient_labels.append(nonzero)

    for recipe in recipe_ids:
        ingredients.append(ingredient_mapping[recipe])

    # BIG DATA IS PROCESSED
    # Now we have a (16, 3, 224, 224) Tensor of images, and a (16, 30167) Tensor of labels
    # We can do d e e p l e a r n i n g
    # what a god.
    time_taken = time.time() - start
    print(f"Batch {i} / {num_batches}, time taken: {time_taken}s", flush=True)

    if i >= 100:
        break

ingredient_info = {"ingredient_text": ingredients, "ingredient_labels": ingredient_labels}
np.save("ingredient_info.npy", ingredient_info)

with torch.no_grad():
    target_model.eval()
    num_val_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start = time.time()
        #############################################################################################
        img_tensor, labels, recipe_ids = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)

        for j in range(int(labels.size()[0])):
            nonzero = np.nonzero(labels[j].cpu().numpy())
            ingredient_labels.append(nonzero)

        for recipe in recipe_ids:
            ingredients.append(ingredient_mapping[recipe])

        time_taken = time.time() - start
        print(f"Batch {i} / {num_val_batches}, time taken: {time_taken}s", flush=True)
        #############################################################################################                

        if i >= 100:
            break

print("="*20)
print("Starting test...")

target_model.eval()

with torch.no_grad():
    test_start = time.time()
    num_test_batches = len(test_loader)
    for i, (input, target) in enumerate(test_loader):
        start = time.time()
        img_tensor, labels, recipe_ids = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)

        for j in range(int(labels.size()[0])):
            nonzero = np.nonzero(labels[j].cpu().numpy())
            ingredient_labels.append(nonzero)

        for recipe in recipe_ids:
            ingredients.append(ingredient_mapping[recipe])

        time_taken = time.time() - start
        print(f"Test batch {i} / {num_test_batches}, time Taken: {time_taken}", flush=True)
        
        if i >= 100:
                break

print("Done")

ingredient_info = {"ingredient_text": ingredients, "ingredient_labels": ingredient_labels}
np.save("ingredient_info.npy", ingredient_info)

