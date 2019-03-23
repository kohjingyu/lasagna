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

from sklearn.metrics import f1_score

# TODO: Set this as command line args
batch_size = 32
workers = 16 # How many cores to use to load data
dev_mode = True # Set this to False when training on Athena
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
###########################################################

best_f1 = 0

for epochs in range(total_epochs):
    epoch_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        start = time.time()
        img_tensor, labels, recipe_id = get_tensor_from_data(input, target, class_mapping, dev_mode=dev_mode)
        # BIG DATA IS PROCESSED
        # Now we have a (16, 3, 224, 224) Tensor of images, and a (16, 30167) Tensor of labels
        # We can do d e e p l e a r n i n g
        # what a god.

        #################################################
        #TRAINING BIT
        target_model.train() # set train mode
        optimiser.zero_grad()
        img_tensor = img_tensor.to(device)
        target = labels.to(device)
        output = torch.sigmoid(target_model(img_tensor.float()))
        result = torch.nn.functional.binary_cross_entropy(output,target)
        storage.store_train_loss(epochs,result)
        #################################################

        result.backward()
        optimiser.step()
        time_taken = time.time() - start
        print(f"Epoch {epochs}, batch {i} / {num_batches}, loss: {result.item()}, time taken: {time_taken}s", flush=True)
    print("Total train time: {}s".format(time.time() - epoch_start))
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
            results = torch.nn.functional.binary_cross_entropy(output,labels) #calculate results.
            # answers = labels.cpu().numpy() #obtain a numpy version of answers.

            preds = (output > 0.5).cpu().numpy()
            labels_arr = labels.cpu().numpy()
            total_samples += preds.shape[0]

            for j in range(preds.shape[0]):
                total_f1 += f1_score(labels_arr[j,:], preds[j,:], average='macro')

            # storage.data_entry(answers,results,epochs,output)
            # again, store losses
            time_taken = time.time() - start
            print(f"Batch {i} / {num_val_batches}, loss: {results.item()}, time taken: {time_taken}s", flush=True)
            #############################################################################################                
        # calculate accuracy
        average_f1 = total_f1 / total_samples
        print(f"F1 score for this epoch: {average_f1}")
        print(f"Time taken: {time.time() - val_start}s")
        # print("Validation Accuracy for this epoch")
        # storage.accuracy_calculation_epoch(epochs) #calculate accuracies.
        # latest = storage.accuracies(epochs)
        # for threshold_value in latest.keys():
        #     print("Threshold of {} :  {}".format(threshold_value,latest[threshold_value]))

        # Save best performing model
        if average_f1 > best_f1:
            best_f1 = average_f1
            torch.save(target_model.state_dict(), f'{snapshots_dir}/best_resnet50.pth')

        ###############################
        ###############################
        ###############################
        # TODO: Early stop if required. 
        ###############################
        ###############################
        ###############################
            
    pickle.dump(storage,open("val_loss_class.pkl","wb"))
    pickle.dump(test_result,open("test_results.pkl","wb"))    
    print("Done 1 epoch. Pickles dumped again.")
    
print("="*20)
print("Starting test...")
print("Loading best model...")
saved_state_dict = torch.load(f"{snapshots_dir}/best_resnet50.pth", map_location='cpu')
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
        results = torch.nn.functional.binary_cross_entropy(output,labels)

        preds = (output > 0.5).cpu().numpy()
        labels_arr = labels.cpu().numpy()
        total_samples += preds.shape[0]
        for j in range(preds.shape[0]):
            total_f1 += f1_score(labels_arr[j,:], preds[j,:], average='macro')
        # test_result.data_entry(answers,results,epochs,output)
        time_taken = time.time() - start
        print(f"Test batch {i} / {num_test_batches}, time Taken: {time_taken} , test loss: {results.item()}", flush=True)
        ################################################
    ############################################################
    # print("Test Accuracy for this epoch")
    # test_result.accuracy_calculation_epoch(epochs) #calculate accuracies.
    # latest = test_result.accuracies(epochs)
    # for threshold_value in latest.keys():
    #     print("Threshold of {} :  {}".format(threshold_value,latest[threshold_value]))
    ############################################################
    average_f1 = total_f1 / total_samples
    print(f"F1 score for this epoch: {average_f1}")
    print(f"Time taken: {time.time() - test_start}s")

print("Done")


