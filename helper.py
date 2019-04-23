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

def get_results_for_image(image_path, root_dir=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_mapping = get_class_mapping(root_dir=root_dir)
    ingredient_mapping = get_ingredient_mapping(root_dir=root_dir)
    class_mapping,_ = get_class_mapping(root_dir=root_dir)

    num_classes = len(class_mapping)


    #############################
    image_transform = transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    target_model = resnet.resnet50(num_classes=num_classes, pretrained=False)
    nf = target_model.fc.in_features
    target_model.fc = torch.nn.Linear(nf, num_classes)

    saved_state_dict = torch.load(root_dir + "snapshots/best_resnet50_b32_posw10_lr0.1_stag3_epochs10.pth", map_location='cpu')
    target_model.load_state_dict(saved_state_dict,strict=False)
    target_model = target_model.to(device)
    target_model.eval()

    with torch.no_grad():
        start = time.time()
        im = Image.open(root_dir + "images/samples/" + image_path)
        img_tensor = image_transform(im).float().unsqueeze(0).to(device)
        ################################################
        output0, output2 = target_model(img_tensor)
        output = torch.sigmoid(output0)
        preds = (output > 0.5).cpu().numpy()
        preds = preds[0,:] # Because it's a batch of 1. Maybe not the best way to do it...
        pred_idx = np.nonzero(preds)[0]

        time_taken = time.time() - start

        ingredients = []
        for idx in pred_idx:
            ingredients.append((ingredient_mapping[idx],output2[0,idx]))

        return ingredients
