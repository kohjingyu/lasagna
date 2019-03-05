import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import pickle
import numpy as np

import time

# TODO: Set this as command line args
batch_size = 16
workers = 8 # How many cores to use to load data
dev_mode = True # Set this to False when training on Athena
total_ingredients = 30167 # Found by loading vocab.bin


#############################
learning_rate = 0.01
momentum_mod = 0.01
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
Epochal_loss=0
total_epochs=30
train_losses={}
val_losses = {}
val_accuracies = []
test_losses = {}
#############################



if not dev_mode:
    from data_loader import ImagerLoader
# Initialise Squeezenet from pytorch.


#####################################################################
Squeezenet_model = torchvision.models.squeezenet1_0(num_classes=30167)
optimiser = torch.optim.SGD(Squeezenet_model.parameters(), lr=learning_rate, momentum=momentum_mod)
Squeezenet_model = Squeezenet_model.to(device)
#####################################################################



if dev_mode:
    # Load from .npy file and batch manually
    data_arr = np.load("./data/smalldata.npy") # shape: (100, 2)
    num_batches = int(np.ceil(data_arr.shape[0] / batch_size))

    img_split = np.array_split(data_arr[:,0], num_batches)
    label_split = np.array_split(data_arr[:,1], num_batches)
    train_loader = [(img_split[i], label_split[i]) for i in range(num_batches)]
else:
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
for epochs in range(total_epochs):
    train_losses[epochs] = []
    val_losses[epochs]= [] #initialise a list in the entry of this epoch number.
    test_losses[epochs] = []
    
    for i, (input, target) in enumerate(train_loader):
        Squeezenet_model.train()
        start = time.time()

        # FOR LMDB FILES:
        # input has 5 components
        # x = img
        # y1 = recipe step data  / torch.Size([1, 20, 1024]) ()
        # y2 = number of steps / torch.Size([1])
        # z1 = ingredient id / torch.Size([1, 20])
        # z2 = number of ingredients / torch.Size([1])
        
        if dev_mode:
            input_arr = np.array([input[i] for i in range(input.shape[0])]) # (16, 3, 224, 224)
            target_arr = np.array([target[i] for i in range(target.shape[0])]) # (16, 30167)
            img_tensor = torch.from_numpy(input_arr) # torch.Size([1, 3, 224, 224])
            labels = torch.from_numpy(target_arr)
        else:
            img_tensor = input[0] # torch.Size([1, 3, 224, 224])
            num_ingredients = int(input[4].data[0]) # torch.Size([1]) (number of ingredients?)
            ingredient_idx = input[3] # torch.Size([1, 20]) (ingredient id?)
            current_batch_size = int(img_tensor.size()[0])

            # Create one-hot vectors (multilabel binarization - ingredients present will have 1 at their indices)
            labels = torch.zeros((current_batch_size, total_ingredients))

            for batch in range(current_batch_size):
                for j in range(num_ingredients):
                    labels[batch, int(ingredient_idx[batch, j])] = 1

        # BIG DATA IS PROCESSED
        # Now we have a (16, 3, 224, 224) Tensor of images, and a (16, 30167) Tensor of labels
        # We can do d e e p l e a r n i n g

        # TODO: Feed data into model, calculate loss and do backpropagation
        
        
        
        #################################################
        Squeezenet_model.eval()
        optimiser.zero_grad()
        img_tensor = img_tensor.to(device)
        target = labels.to(device)
        output = torch.sigmoid(Squeezenet_model(img_tensor))
        result = torch.nn.functional.binary_cross_entropy(output,target)
        #################################################
        
        
        
        result.backward()
        optimiser.step()
        train_losses[epochs].append(result.item()) #append to dictionary of losses
        time_taken = time.time() - start
        print("Epoch {}, batch{}, Training Loss - {}".format(epochs,i,result.item()))
        print(f"Batch {i} / {num_batches}, Time taken: {time_taken}s")
        

    # TODO: Perform validation
        print("="*20)
        print("Starting validation...")
        # TODO: Perform validation
        correct =0
        with torch.no_grad():
            num_val_batches = len(val_loader)
            for i, (input, target) in enumerate(val_loader):
                start = time.time()
                img_tensor, labels = get_tensor_from_data(input, target, dev_mode=dev_mode)
                
                
                
                
                ########################################################
                output = torch.sigmoid(Squeezenet_model(img_tensor))
                predicted = torch.where(output>0.5,1,0)
                target = labels.to(device)
                correct += (predicted == test_labels2[0]).sum()
                results = torch.nn.functional.binary_cross_entropy(output,target)
                time_taken = time.time() - start
                val_losses[epochs].append(results.item())
                ########################################################
                
                
                
                
                
                print("Epoch {}, batch{}, Validation Loss - {}".format(epochs,i,result.item()))
                
                print(f"Validation batch {i} / {num_val_batches}, Time taken: {time_taken}s")
            print("Validation Accuracy {}".format(correct/len(val_loader)))
            val_accuracies.append(correct/len(val_loader))
            # TODO: Calculate validation accuracy or something
            # Early stop if required

    print("="*20)
    print("Starting test...")
    # TODO: Perform test
    correct =0
    Squeezenet_model.eval()
    with torch.no_grad():
        num_test_batches = len(test_loader)
        for i, (input, target) in enumerate(test_loader):
            start = time.time()
            img_tensor, labels = get_tensor_from_data(input, target, dev_mode=dev_mode)
            
            
            ################################################
            output = torch.sigmoid(Squeezenet_model(img_tensor))
            results = torch.nn.functional.binary_cross_entropy(output,labels)
            test_losses[epochs].append(results.item())
            correct += (predicted == test_labels2[0]).sum()
            time_taken = time.time() - start
            print("Epoch {}, batch{}, Test Loss - {}".format(epochs,i,result.item()))
            ################################################
            
            
            
            
            # print(f"Test batch {i} / {num_test_batches}, Time taken: {time_taken}s")
        # TODO: Calculate test accuracy
        
        
        
        ############################################################
        print("Test accuracy: {}".format(correct/len(test_loader)))
        test_accuracies.append(correct/len(test_loader))
        ############################################################
        
        
    print("Done")
    
    
###################
pickle.dump(train_losses,open("trainloss.pkl","wb"))
import matplotlib.pyplot as plt
holder = []
for i in train_losses:
    holder.append(sum(train_losses[i]))
plt.plot(holder)
plt.show()
# pickle.dump(val_losses,open("valloss.pkl","wb"))
# pickle.dump(test_losses,open("testloss.pkl","wb"))
torch.save(Squeezenet_model.state_dict(),"SqueezenetDict.pt")
###################



