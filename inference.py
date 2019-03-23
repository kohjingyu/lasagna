import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import pickle
import numpy as np
import time
from data import get_class_mapping
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
num_classes = len(class_mapping)
#############################
image_transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
# initialise model
if sys.argv[2] =="resnet":
    target_model = torchvision.models.resnet50(pretrained=True)
elif sys.argv[2] =="densenet":
    target_model = torchvision.models.densenet121(pretrained=True)
else:
    display_help()
    quit()


if sys.argv[2] == "resnet":
    nf = target_model.fc.in_features
    target_model.fc = torch.nn.Linear(nf, num_classes)
else:
    nf = target_model.classifier.in_features
    target_model.classifier = torch.nn.Linear(nf, num_classes)

saved_state_dict = torch.load(sys.argv[3], map_location='cpu')
target_model.load_state_dict(saved_state_dict)
target_model = target_model.to(device)
target_model.eval()

print("\n\nModel loaded!")
with torch.no_grad():
    start = time.time()
    im = Image.open(image_path)
    img_tensor = image_transform(im).float().unsqueeze(0).to(device)
    ################################################
    output = torch.sigmoid(target_model(img_tensor))
    preds = (output > 0.5).cpu().numpy()
    time_taken = time.time() - start
    print("Ingredients", np.nonzero(preds))
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
