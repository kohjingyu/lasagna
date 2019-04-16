import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

# To avoid OSError with some images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import resnet
import pickle
import numpy as np
import time
from data import get_class_mapping, get_ingredient_mapping
import pickle
import json
import sys

def display_help():
    print("""⢠⠤⣤⠀⠤⡤⠄⢠⡤⢤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢸⠲⣏⠀⢀⡇⠀⢸⡗⠚⢀⣤⣶⣾⣿⣷⣶⣤⣄⠀⠀⣀⣤⣤⣴⣦⣤⡀⠀⠀⠀⠀⠀⠀⠀
    ⠈⠀⠈⠀⠉⠉⠁⠈⠁⣴⣿⣿⣿⡿⠿⣛⣛⠻⠿⣧⢻⣿⣿⣿⣿⣿⣿⣿⣄⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣫⣵⣾⣿⣿⣿⡿⠷⠦⠔⣶⣶⣶⣶⣶⠶⠶⠶⠤⡀⠀⠀⠀⠀
   ⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⠿⠛⢁⣀⣌⣿⣿⣷⣶⣈⠿⣒⣒⣭⣭⣭⣭⣑⣒⠄⠀⠀
  ⠀⠀⠀⠀⠀⠀⣠⡎⣾⣿⣿⣿⣿⢫⣡⡥⠶⠿⣛⠛⠋⠳⢶⣶⣾⣜⣫⣭⣷⠖⡁⠀⠐⢶⣯⡆⠀
    ⠀⠀⠀⣰⣿⣷⣿⣿⣿⣿⣿⣷⣖⢟⡻⢿⠃⢸⠱⠶⠀⠿⠟⡻⠿⣿⡏⠀⠅⠛⠀⣘⠟⠁⠀ ⠀
    ⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣮⣥⣤⣴⣤⣦⠄⣠⣾⣿⡻⠿⠾⠿⠿⠟⠛⠁⠀⠀
    ⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣭⣶⣿⣿⣿⣿⣿⣷⣿⣿⣿⣧⡀⠀
    ⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀ 
    ⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢩⡤⠶⠶⠶⠦⠬⣉⣛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⣋⣡⠀⠀
    ⠀ ⠀⠘⣿⣿⣿⣿⣿⣿⣟⢿⣧⣙⠓⢒⣚⡛⠳⠶⠤⢬⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⡄⠀
    ⠀⠀⠀ ⠀⠀⠈⠻⢿⣿⣿⣿⣿⣶⣽⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣤⣤⣤⣤⣤⣤⡥⠄
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠒⠭⢭⣛⣛⡻⠿⠿⠿⠿⣿⣿⣿⣿⣿⠿""")
    print("python trialScript.py <image to be categorised> <resnet / densenet> <model PATH>") #<OPTIONAL:correct labels>")
    print("It's resnet50 or densenet121 btw")
    print("Sorry")


outputdir = "./output"
import pathlib
pathlib.Path(outputdir).mkdir(exist_ok=True) # Create snapshot directory if it doesn't exist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
if len(sys.argv) < 3:
    display_help()

image_path = sys.argv[1]
class_mapping = get_class_mapping()
ingredient_mapping = get_ingredient_mapping()
class_mapping,_ = get_class_mapping()

num_classes = len(class_mapping)


#############################
image_transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
# initialise model
if sys.argv[2] =="resnet":
    print(sys.argv[2])
    target_model = resnet.resnet50(num_classes=num_classes, pretrained=False)
elif sys.argv[2] =="densenet":
    target_model = torchvision.models.densenet121(pretrained=True)
else:
    display_help()
    quit()


if sys.argv[2] == "resnet":
    nf = target_model.fc.in_features
    target_model.fc = torch.nn.Linear(nf, num_classes)
    print(nf,num_classes)
else:
    nf = target_model.classifier.in_features
    target_model.classifier = torch.nn.Linear(nf, num_classes)

saved_state_dict = torch.load(sys.argv[3], map_location='cpu')
target_model.load_state_dict(saved_state_dict,strict=False)
target_model = target_model.to(device)
target_model.eval()

print("\n\nModel loaded!")
with torch.no_grad():
    start = time.time()
    im = Image.open(image_path)
    img_tensor = image_transform(im).float().unsqueeze(0).to(device)
    ################################################
    output0, output2 = target_model(img_tensor)
    print('output2 size: {}'.format(output2.size()))
    output = torch.sigmoid(output0)
    preds = (output > 0.5).cpu().numpy()
    preds = preds[0,:] # Because it's a batch of 1. Maybe not the best way to do it...
    pred_idx = np.nonzero(preds)[0]

    time_taken = time.time() - start

    ingredients = []
    for idx in pred_idx:
        ingredients.append((ingredient_mapping[idx],output2[0,idx]))

    print("Ingredients", ingredients)
    print(f"Time Taken: {time_taken}", flush=True)
    # json.dump(preds.tolist(),open(outputdir+"/"+image_path+"_output.json","w"))
    # print("output complete")
    print("""──────────▀█───────────────────▀█─
──────────▄█───────────────────▄█─
──█████████▀───────────█████████▀─
───▄██████▄─────────────▄██████▄──
─▄██▀────▀██▄─────────▄██▀────▀██▄
─██────────██─────────██────────██
─██───██───██─────────██───██───██
─██────────██─────────██────────██
──██▄────▄██───────────██▄────▄██─
───▀██████▀─────────────▀██████▀──
───────────█████████████──────────""")

    ################################################

    # if len(sys.argv)>4:
    #     answers = json.load(sys.argv[4])
    #     from sklearn.metrics import f1_score
    #     "f1 score: {}".format(f1_score(labels_arr, preds, average='macro'))
    #     TP = np.sum(np.logical_and(predicted == 1, answers== 1))
    #     TN = np.sum(np.logical_and(predicted == 0, answers == 0))
    #     FP = np.sum(np.logical_and(predicted == 1, answers == 0))
    #     FN = np.sum(np.logical_and(predicted == 0, answers== 1))
    #     print(f"TP: {TP}     TN: {TN}     FP:{FP}     FN:{FN}")
