import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from PIL import Image
import pickle
import numpy as np

import time
import pathlib
import argparse

from data import get_tensor_from_data, get_class_mapping, get_class_weights
from data_storage_class import result_storage

from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=32,
                   help='batch size for training')
parser.add_argument('--num_epochs', metavar='num_epochs', type=int, default=10,
                   help='number of train epochs')
parser.add_argument('--max_stagnation', metavar='max_stagnation', type=int, default=3,
                   help='number of validation epochs that do not improve to consider before early stopping')
parser.add_argument('--lr', metavar='lr', type=float, default=0.1,
                   help='base learning rate')
parser.add_argument('--momentum', metavar='momentum', type=float, default=0.9,
                   help='momentum for SGD')
parser.add_argument('--adam', metavar='adam', type=bool, default=False,
                   help='whether to use Adam as the optimization algorithm')
parser.add_argument('--workers', metavar='workers', type=int, default=16,
                   help='number of cores to use for data loading')
parser.add_argument('--pos_weight', metavar='pos_weight', type=int, default=10,
                   help='weight for positive label in BCELoss')
parser.add_argument('--use_class_weights', metavar='use_class_weights', type=bool, default=False,
                   help='whether to weight to account for class imbalance')
parser.add_argument('--dev_mode', metavar='dev_mode', type=bool, default=False,
                   help='whether to run in development mode')
parser.add_argument('--data_dir', metavar='data_dir', type=str, default="./dataset",
                   help='root directory containing train / val / test data')
parser.add_argument('--snapshots_dir', metavar='snapshots_dir', type=str, default="./snapshots",
                   help='root directory to store model states')
parser.add_argument('--model_name', metavar='model_name', type=str, default="resnet",
                   help='model to use [resnet / densenet]')

args = parser.parse_args()
print(args)

#############################
# Load hyperparameters

batch_size = args.batch_size
workers = args.workers # How many cores to use to load data
dev_mode = args.dev_mode # Set this to False when training on Athena

if dev_mode:
    batch_size = 8

pos_weight = args.pos_weight

data_dir = args.data_dir
snapshots_dir = args.snapshots_dir
pathlib.Path(snapshots_dir).mkdir(exist_ok=True) # Create snapshot directory if it doesn't exist

learning_rate = args.lr
momentum_mod = args.momentum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class_mapping, class_weights = get_class_mapping()
use_class_weights = args.use_class_weights

if not use_class_weights:
    class_weights = torch.ones(class_weights.shape).to(device)
else:
    class_weights = torch.Tensor(class_weights).to(device)

num_classes = len(class_mapping)
print("Num classes:", num_classes)
total_epochs = args.num_epochs
max_stagnation = args.max_stagnation

model_name = f"best_{args.model_name}_b{batch_size}_posw{pos_weight}_lr{learning_rate}_stag{max_stagnation}_epochs{total_epochs}.pth"
#############################

def calc_loss(probs, target, pos_weight=1):
    """
    probs (torch.Tensor): Size (N x C) containing probability for each class to be classified
    target (torch.Tensor): Size (N x C) containing 1 if a class is present and 0 otherwise
    pos_weight (float, optional): The weight to assign to the cost of a positive error relative to a negative error
    """
    return (class_weights * pos_weight * -target * torch.log(probs) - (1 - target) * class_weights * torch.log(1 - probs)).mean()

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

if args.model_name == "resnet":
    target_model = torchvision.models.resnet50(pretrained=True)
    nf = target_model.fc.in_features
    target_model.fc = torch.nn.Linear(nf, num_classes)
else:
    target_model = torchvision.models.densenet161(pretrained=True)
    nf = target_model.classifier.in_features
    target_model.classifier = torch.nn.Linear(nf, num_classes)

if args.adam:
    optimizer = torch.optim.adam(target_model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
else:
    optimizer = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum_mod)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# criterion = torch.nn.BCELoss()

target_model = target_model.to(device)
storage = result_storage(False,batch_size,num_classes,num_batches)
test_result = result_storage(True,batch_size,num_classes,num_batches)
###########################################################

best_f1 = 0
stagnating_epochs = 0 # Incremented when validation F1 score does not increase

for epochs in range(total_epochs):
    epoch_start = time.time()
    target_model.train()
    total_f1 = 0
    total_samples = 0

    for i, (input, target) in enumerate(train_loader):
        start = time.time()
        img_tensor, labels, recipe_id = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)
        # BIG DATA IS PROCESSED
        # Now we have a (16, 3, 224, 224) Tensor of images, and a (16, 30167) Tensor of labels
        # We can do d e e p l e a r n i n g
        # what a god.

        #################################################
        #TRAINING BIT
        scheduler.step()
        optimizer.zero_grad()
        img_tensor = img_tensor.to(device)
        target = labels.to(device)

        output = target_model(img_tensor.float())
        probs = torch.sigmoid(output)
        preds = probs > 0.5

        preds_arr = preds.cpu().numpy()
        total_samples += preds_arr.shape[0]

        labels_arr = labels.cpu().numpy()
        for j in range(preds_arr.shape[0]):
            total_f1 += f1_score(labels_arr[j,:], preds_arr[j,:], average='macro')

        if pos_weight != 1:
            loss = calc_loss(probs, target, pos_weight=pos_weight)
        else:
            loss = calc_loss(probs, target)

        num_nonzero = float(torch.sum(preds > 0)) / preds_arr.shape[0]

        # storage.store_train_loss(epochs,loss)
        #################################################

        loss.backward()
        optimizer.step()
        time_taken = time.time() - start
        print("Epoch {epochs}, batch {i} / {num_batches}, loss: {loss:.5f}, num_nonzero: {num_nonzero}, F1: {f1:.5f} lr: {lr:.5f} time taken: {time_taken:.3f}s".format(
                epochs=epochs,
                i=i,
                num_batches=num_batches,
                loss=loss.item(),
                num_nonzero=num_nonzero,
                f1=total_f1 / total_samples,
                lr=scheduler.get_lr()[0],
                time_taken=time_taken
            ), flush=True)
    print(f"Average train F1: {total_f1 / total_samples}, total train time: {time.time() - epoch_start}s")
    print("="*20)
    print("Starting validation...")

    total_f1 = 0
    total_samples = 0

    with torch.no_grad():
        target_model.eval()
        val_start = time.time()
        num_val_batches = len(val_loader)
        for i, (input, target) in enumerate(val_loader):
            start = time.time()
            #############################################################################################
            img_tensor, labels, recipe_id = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)
            output = torch.sigmoid(target_model(img_tensor.float())) #sigmoid values. since it's binary cross entropy.

            if pos_weight != 1:
                loss = calc_loss(output, labels, pos_weight=pos_weight)
            else:
                loss = calc_loss(output, labels)

            # answers = labels.cpu().numpy() #obtain a numpy version of answers.

            preds = output > 0.5

            preds_arr = preds.cpu().numpy()

            labels_arr = labels.cpu().numpy()
            total_samples += preds_arr.shape[0]

            for j in range(preds_arr.shape[0]):
                total_f1 += f1_score(labels_arr[j,:], preds_arr[j,:], average='macro')

            # storage.data_entry(answers,loss,epochs,output)
            # again, store losses
            time_taken = time.time() - start
            print("Validation epoch {epochs}, batch {i} / {num_batches}, loss: {loss:.5f}, F1: {f1}, time taken: {time_taken:.3f}s".format(
                epochs=epochs,
                i=i,
                num_batches=num_val_batches,
                loss=loss.item(),
                f1=total_f1 / total_samples,
                time_taken=time_taken
            ), flush=True)

            #############################################################################################                
            # calculate accuracy
        average_f1 = total_f1 / total_samples
        print("=" * 20)
        print(f"Validation F1 score for this epoch: {average_f1}")
        print(f"Time taken: {time.time() - val_start}s")
        # print("Validation Accuracy for this epoch")
        # storage.accuracy_calculation_epoch(epochs) #calculate accuracies.
        # latest = storage.accuracies(epochs)
        # for threshold_value in latest.keys():
        #     print("Threshold of {} :  {}".format(threshold_value,latest[threshold_value]))

        # Save best performing model
        if average_f1 > best_f1:
            print("New best model!")
            best_f1 = average_f1
            torch.save(target_model.state_dict(), f'{snapshots_dir}/{model_name}')
            stagnating_epochs = 0
        else:
            stagnating_epochs += 1
            print(f"Did not improve. Stagnation: {stagnating_epochs}/{max_stagnation}")

        ###############################
        # Early stop if required. 
        if stagnating_epochs >= max_stagnation:
            break
        ###############################

print("="*20)
print("Starting test...")
print("Loading best model...")
saved_state_dict = torch.load(f"{snapshots_dir}/{model_name}", map_location='cpu')
target_model.load_state_dict(saved_state_dict, strict=False)

correct =0
target_model.eval()
total_f1 = 0
total_samples = 0

with torch.no_grad():
    test_start = time.time()
    num_test_batches = len(test_loader)
    for i, (input, target) in enumerate(test_loader):
        start = time.time()
        img_tensor, labels, recipe_id = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)
        img_tensor = img_tensor.to(device)
        labels = labels.to(device)
        ################################################
        output = torch.sigmoid(target_model(img_tensor.float()))

        if pos_weight != 1:
            loss = calc_loss(output, labels, pos_weight=pos_weight)
        else:
            loss = calc_loss(output, labels)

        preds = (output > 0.5)
        preds_arr = preds.cpu().numpy()

        labels_arr = labels.cpu().numpy()
        total_samples += preds_arr.shape[0]
        for j in range(preds_arr.shape[0]):
            total_f1 += f1_score(labels_arr[j,:], preds_arr[j,:], average='macro')
        # test_result.data_entry(answers,loss,epochs,output)
        time_taken = time.time() - start

        print("Test epoch {epochs}, batch {i} / {num_batches}, loss: {loss:.5f}, F1: {f1} time taken: {time_taken:.3f}s".format(
            epochs=epochs,
            i=i,
            num_batches=num_test_batches,
            loss=loss.item(),
            f1=total_f1 / total_samples,
            time_taken=time_taken
        ), flush=True)

        ################################################
    ############################################################
    # print("Test Accuracy for this epoch")
    # test_result.accuracy_calculation_epoch(epochs) #calculate accuracies.
    # latest = test_result.accuracies(epochs)
    # for threshold_value in latest.keys():
    #     print("Threshold of {} :  {}".format(threshold_value,latest[threshold_value]))
    ############################################################
    average_f1 = total_f1 / total_samples
    print("=" * 20)
    print(f"Best validation F1 score: {best_f1}")
    print(f"Test F1 score: {average_f1}")
    print(f"Time taken: {time.time() - test_start}s")

print("Done")


