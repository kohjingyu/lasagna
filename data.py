import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

import os

import numpy as np
import json

class IngredientDataset(Dataset):
    def __init__(self, data_path, ingredients_path, data_root):
        """
        Args:
            data_path (string): path to .npy file of food data
            ingredients_path (string): path to .npy file of ingredient data
        """

        self.data_root = data_root

        # Transforms
        self.to_tensor = transforms.ToTensor()

        self.data = np.load(data_path)
        self.ingredients_info = np.load(ingredients_path).item()
        self.num_ingredients = len(self.ingredients_info)

        self.image_arr = []
        self.label_arr = []

        # Convert to one-hot vectors
        for d in self.data:
            filepath = self.get_file_path(d["id"])

            if os.path.isfile(filepath):
                print(filepath, "exists")
                self.image_arr.append(d["id"])

                result = np.zeros((self.num_ingredients))
                ingredients = d["ingredients"]
                
                for i in ingredients:
                    result[i] = 1
                self.label_arr.append(result)
            else:
                print(filepath, "doesn't exist")

        self.data_len = len(self.image_arr)

    def get_file_path(self, image_id):
        return self.data_root + "/".join(list(image_id[:4])) + "/" + image_id + ".jpg"

    def __getitem__(self, index):
        # Get image name 
        filename = self.image_arr[index]
        # Convert to how our directory is organized
        filename = self.get_file_path(filename)
        print(filename)

        # Open image
        img = Image.open(filename)

        # TODO: Perform image augmentations here
        # We may want to crop the image to a fixed size or something

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img)

        # Get label
        label = self.label_arr[index]

        return (img, label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    custom_loader = IngredientDataset('smalldata/ingredients_vectorized.npy', 'smalldata/ingredients_mapping.npy', "smalldata/train/")

    for i, (images, labels) in enumerate(custom_loader):
        pass