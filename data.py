import numpy as np
import torch

def get_ingredient_mapping(root_dir="./"):
    ingredient_mapping = np.load(root_dir + "data/ingredient_mapping.npy")

    return ingredient_mapping

def get_class_weights(root_dir="./"):
    class_weights = np.load(root_dir + "data/class_weights.npy")
    class_weights /= np.min(class_weights)

    return class_weights

def get_class_mapping(root_dir="./"):
    """
    Returns the new class mapping by filtering out unused (or long tailed) classes
    """

    label_counts = np.load(root_dir + "data/label_count.npy")
    label_counts[1] = 1 # Ignore the rubbish label
#    top_k = 1500 # Only use top k labels

#    label_counts[np.argsort(-label_counts)[top_k:]] = 0
    nonzero_idx = np.nonzero(label_counts)[0]

    mapping = {}
    counts = []

    for idx in nonzero_idx:
        mapping[idx] = len(mapping)
        counts.append(label_counts[idx])

    weights = 1 / np.array(counts)
    weights /= np.max(weights)

    return mapping, weights

def get_tensor_from_data(input, target, mapping, recipe_quantity_mapping, dev_mode=False):
    total_classes = len(mapping)

    if dev_mode:
        input_arr = np.array([input[i] for i in range(input.shape[0])]) # (16, 3, 224, 224)
        target_arr = np.zeros((target.shape[0], total_classes)) # (16, 3748)

        # Convert to new mapping
        for i in range(target.shape[0]):
            nonzero = np.nonzero(target_arr)[0]
            for j in nonzero:
                if j in mapping:
                    target_arr[i, mapping[j]] = 1

        img_tensor = torch.from_numpy(input_arr) # torch.Size([1, 3, 224, 224])
        labels = torch.from_numpy(target_arr).type(torch.FloatTensor)
        recipe_id = ""
    else:
        img_tensor = input[0] # torch.Size([1, 3, 224, 224])
        num_ingredients = input[4] # torch.Size([1]) (number of ingredients?)
        ingredient_idx = input[3] # torch.Size([1, 20]) (ingredient id?)
        recipe_id = input[5]

        current_batch_size = int(img_tensor.size()[0])

        # Create one-hot vectors (multilabel binarization - ingredients present will have 1 at their indices)
        labels = torch.zeros((current_batch_size, total_classes))
        labels2 = torch.zeros((current_batch_size, total_classes))

        for batch in range(current_batch_size):
            for j in range(int(num_ingredients[batch])):
                current_ingredient_idx = int(ingredient_idx[batch, j])
                if current_ingredient_idx in mapping and current_ingredient_idx > 1:
                    mapped_idx = mapping[current_ingredient_idx]
                    labels[batch, mapped_idx] = 1
            riqm = recipe_quantity_mapping[recipe_id[batch]]
            nonzero_riqm = riqm[0]
            labels2[batch,nonzero_riqm] = torch.FloatTensor(riqm[1])
        labels2 = labels2*labels
    return img_tensor, labels, labels2, recipe_id
