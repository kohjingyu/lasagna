import numpy as np
import torch

def get_tensor_from_data(input, target, dev_mode=False):
    if dev_mode:
        input_arr = np.array([input[i] for i in range(input.shape[0])]) # (16, 3, 224, 224)
        target_arr = np.array([target[i] for i in range(target.shape[0])]) # (16, 30167)
        img_tensor = torch.from_numpy(input_arr) # torch.Size([1, 3, 224, 224])
        labels = torch.from_numpy(target_arr)
    else:
        img_tensor = input[0] # torch.Size([1, 3, 224, 224])
        num_ingredients = input[4] # torch.Size([1]) (number of ingredients?)
        ingredient_idx = input[3] # torch.Size([1, 20]) (ingredient id?)

        current_batch_size = int(img_tensor.size()[0])

        # Create one-hot vectors (multilabel binarization - ingredients present will have 1 at their indices)
        labels = torch.zeros((current_batch_size, total_ingredients))

        for batch in range(current_batch_size):
            for j in range(int(num_ingredients[batch])):
                labels[batch, int(ingredient_idx[batch, j])] = 1

    return img_tensor, labels
