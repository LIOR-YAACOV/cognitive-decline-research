import os
import torch
import random
import argparse
import yaml

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt


def moca_to_class(score):
    if score <= 20:
        return 0
    elif score <= 26:
        return 1
    else:
        return 2
        
def moca0_to_12(score):
    if score < 18:
      score = float(score)/3.0 + 12.0
    return score


class AlzData(torch.utils.data.Dataset):
    def __init__(self,
                 image_list_file,
                 size=512,
                 mode="train",
                 augment_args=None,
                 target_label="MOCA_CLASSES_3"
                 ):
        super(AlzData, self).__init__()

        self.img_mean = 0.9904036772625554
        self.std = 0.06184305670932849

        self.target_label = target_label
        self.size = size
        self.mode = mode

        self.augment_args = augment_args

        lines = []

        self.images = []
        self.labels = []

        with open(image_list_file, "r") as f:
            for line in f.readlines():
                lines.append(line)

        for i in range(len(lines)):
            values = str.split(lines[i], ' ')
            if len(values) == 3:
                values = values[1:]
            if self.target_label == "MOCA_CLASSES_3":
                self.labels.append(int(values[0]))
            else:
                self.labels.append(float(values[0]))

            img_name = values[-1].replace("\n", "")
            self.images.append(os.path.join(os.getcwd(), "alzheimer_2024_07_18_blended_1_512", img_name))

    def __len__(self):
        return len(self.images)

    def get_augmented_image(self, img):
        img = transforms.Resize([self.size, self.size])(img)
        img = transforms.ToTensor()(img)
        if not self.mode == "train":
            return transforms.Normalize(mean=self.img_mean, std=self.std)(img)

        else:
            if self.augment_args is not None:
                brightness = self.augment_args['brightness']
                img = tf.adjust_brightness(img, random.uniform(1 - brightness, 1 + brightness))

                gamma = self.augment_args['gamma']
                img = tf.adjust_gamma(img, random.uniform(1, gamma))

                gamma = self.augment_args['rotate']
                rotate_degree = 2 * random.random() * gamma - gamma

                scale = self.augment_args['scale']
                scale_f = (scale[1] - scale[0]) * random.random() + scale[0]

                translate = self.augment_args['translate']
                tu = 2 * random.random() * translate[0] - translate[0]
                tv = 2 * random.random() * translate[1] - translate[1]

                shear = self.augment_args['shear']
                shear_x = 2 * random.random() * shear - shear
                shear_y = 2 * random.random() * shear - shear

                h_flip_prob = self.augment_args['hflip']
                if random.random() > h_flip_prob:
                    img = tf.hflip(img)

                img = tf.affine(img,
                                translate=[tu, tv],
                                scale=scale_f,
                                angle=rotate_degree,
                                shear=[shear_x, shear_y],
                                interpolation=tf.InterpolationMode.BILINEAR)  # BILINEAR NEAREST)

            return transforms.Normalize(mean=self.img_mean, std=self.std)(img)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.target_label == "MOCA_CLASSES_3":
            label = moca_to_class(label)
        elif self.target_label == "MOCA_0_18_TO_12_18":
            label = moca0_to_12(label)
        else:
            label = float(label)

        img_file = self.images[idx]
        img = Image.open(img_file)
        tensor_img = self.get_augmented_image(img)

        return tensor_img, label, img_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_list',
                        type=str,
                        help='path of file with images list')
    parser.add_argument("--augment_args_file",
                        type=str,
                        help="files containing augmentation arguments",
                        default="")
    args = parser.parse_args()

    if not os.path.isfile(args.files_list):
        print(f'no image list file found')
        exit()

    alz_data = AlzData(image_list_file=args.files_list)

    print(f"size of alzheimer dataset {len(alz_data)}")

    img_tensor, lbl, image = alz_data[0]

    image.close()

    if not os.path.isfile(args.augment_args_file):
        print(f'augment argument file found')
        exit()

    with open(args.augment_args_file) as fp:
        config_file = yaml.load(fp, Loader=yaml.FullLoader)

    alz_data_2 = AlzData(image_list_file=args.files_list,
                          augment_args=config_file['augmentations'],
                          target_label="MOCA_REGRESSION_ALL")

    print(f"size of alzheimer dataset {len(alz_data)}")

    img_tensor1, lbl1, image1 = alz_data_2[0]

    print(lbl1)

    plt.imshow(image1, cmap='gray')
    plt.axis('off')
    plt.show()


