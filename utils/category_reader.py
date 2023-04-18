import numpy as np


def read_categories(filename):
    category_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            splitted_line = line.strip().rstrip('\n').split("\t")
            category_id, category_name = splitted_line[0], ''.join(splitted_line[1:])
            category_dict[category_id] = category_name
        return category_dict

def aggregate_category_freq(category_dict, prediction):
    fr_dict = {}
    unique, fr_unique = np.unique(prediction)
    for key,fr in zip(unique, fr_unique):
        fr_dict[category_dict[key]] = fr_unique
    return fr_dict