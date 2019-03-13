import numpy as np
import torch

def get_class_mapping():
    """
    Returns the new class mapping by filtering out unused (or long tailed) classes
    """

    label_counts = np.load("data/label_count.npy")
    nonzero_idx = np.nonzero(label_counts)[0]

    mapping = {}
    for idx in nonzero_idx:
        mapping[idx] = len(mapping)

    return mapping

def get_tensor_from_data(input, target, mapping, dev_mode=False):
    total_classes = len(mapping)

    if dev_mode:
        input_arr = np.array([input[i] for i in range(input.shape[0])]) # (16, 3, 224, 224)
        target_arr = np.zeros((target.shape[0], total_classes)) # (16, 30167)

        # Convert to new mapping
        for i in range(target.shape[0]):
            nonzero = np.nonzero(target_arr)[0]
            for j in nonzero:
                target_arr[i, mapping[j]] = 1

        img_tensor = torch.from_numpy(input_arr) # torch.Size([1, 3, 224, 224])
        labels = torch.from_numpy(target_arr).type(torch.FloatTensor)
    else:
        img_tensor = input[0] # torch.Size([1, 3, 224, 224])
        num_ingredients = input[4] # torch.Size([1]) (number of ingredients?)
        ingredient_idx = input[3] # torch.Size([1, 20]) (ingredient id?)

        current_batch_size = int(img_tensor.size()[0])

        # Create one-hot vectors (multilabel binarization - ingredients present will have 1 at their indices)
        labels = torch.zeros((current_batch_size, total_classes))

        for batch in range(current_batch_size):
            for j in range(int(num_ingredients[batch])):
                current_ingredient_idx = int(ingredient_idx[batch, j])
                mapped_idx = mapping[current_ingredient_idx]
                labels[batch, mapped_idx] = 1

    return img_tensor, labels
