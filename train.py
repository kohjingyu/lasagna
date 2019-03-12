import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import pickle
import numpy as np
from data import get_tensor_from_data
import time
import pickle
from data_storage_class import result_storage
# TODO: Set this as command line args
batch_size = 16
workers = 8 # How many cores to use to load data
dev_mode = False # Set this to False when training on Athena
data_dir = "./dataset"

#############################
# TOGGLES HERE
learning_rate = 0.01
momentum_mod = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
Epochal_loss = 0
num_classes = 30167 # Found by loading vocab.bin
total_epochs = 30
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
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ]), data_path=f"{data_dir}/lmdbs/", partition='val', sem_reg=None),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

#####################################################################
num_batches = len(train_loader) # now we can get your batches since dataset is chosen
#initialise your model here
# target_model = torchvision.models.squeezenet1_0(num_classes=num_classes)
target_model = torchvision.models.resnet34(pretrained=True)
nf = target_model.fc.in_features
target_model.fc = torch.nn.Linear(nf, num_classes)
optimiser = torch.optim.SGD(target_model.parameters(), lr=learning_rate, momentum=momentum_mod)  # TOGGLES HERE.
target_model = target_model.to(device)
storage = result_storage(False,batch_size,num_classes,num_batches)
test_result = result_storage(True,batch_size,num_classes,num_batches)
###########################################################


for epochs in range(total_epochs):
    epoch_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        start = time.time()
        img_tensor, labels = get_tensor_from_data(input, target, dev_mode=dev_mode)
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
    with torch.no_grad():
        target_model.eval()
        num_val_batches = len(val_loader)
        for i, (input, target) in enumerate(val_loader):
            start = time.time()
            #############################################################################################
            img_tensor, labels = get_tensor_from_data(input, target, dev_mode=dev_mode)
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)
            # or whatever God Koh Jing Yu did. i'm not even sure how much i've defaced it.
            # did you hear he's a god and has actual coding standards
            output = torch.sigmoid(target_model(img_tensor.float())) #sigmoid values. since it's binary cross entropy.
            results = torch.nn.functional.binary_cross_entropy(output,labels) #calculate results.
            answers = labels.cpu().numpy() #obtain a numpy version of answers.
            storage.data_entry(answers,results,epochs,output)
            # again, store losses
            time_taken = time.time() - start
            print("Time Taken: {}   Batch loss: {}".format(time_taken,results.item()))
            #############################################################################################                
        # calculate accuracy
        print("Validation Accuracy for this epoch")
        storage.accuracy_calculation_epoch(epochs) #calculate accuracies.
        latest = storage.accuracies(epochs)
        for threshold_value in latest.keys():
            print("Threshold of {} :  {}".format(threshold_value,latest[threshold_value]))

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
# TODO: Perform test
correct =0
target_model.eval()
with torch.no_grad():
    num_test_batches = len(test_loader)
    for i, (input, target) in enumerate(test_loader):
        start = time.time()
        img_tensor, labels = get_tensor_from_data(input, target, dev_mode=dev_mode)
        img_tensor = img_tensor.to(device)
        labels = labels.to(device)
        ################################################
        output = torch.sigmoid(target_model(img_tensor.float()))
        results = torch.nn.functional.binary_cross_entropy(output,labels)
        test_result.data_entry(answers,results,epochs,output)
        time_taken = time.time() - start
        print("Epoch {}, batch{}, Time Taken: {} , Test Loss - {}".format(epochs,i,time_taken,results.item()))
        ################################################
    ############################################################
    print("Test Accuracy for this epoch")
    test_result.accuracy_calculation_epoch(epochs) #calculate accuracies.
    latest = test_result.accuracies(epochs)
    for threshold_value in latest.keys():
        print("Threshold of {} :  {}".format(threshold_value,latest[threshold_value]))
    ############################################################
print("Done")


