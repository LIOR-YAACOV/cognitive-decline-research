from torchvision.datasets.folder import find_classes
import os


def classes_to_indices(root_dir):
    classes, class_to_idx = find_classes(root_dir)
    return classes, class_to_idx


if __name__ == '__main__':
    directory = os.getcwd() + '/TU-berlin-dataset'
    classes_list, class_to_idx_mapping = classes_to_indices(directory)
    print([class_to_idx_mapping[curr_class] for curr_class in classes_list])
