import numpy as np
import cv2
import os
from tqdm import tqdm
import scipy.misc
import imageio

def color2annotation(input_path, output_path, new_img_path2):

    # image = scipy.misc.imread(input_path)
    # imread is deprecated in SciPy 1.0.0, and will be removed in 1.2.0. Use imageio.imread instead.
    image = imageio.v2.imread(input_path)
    image2 = imageio.v2.imread(input_path.replace('_mask.png', '_sat.jpg'))
    image = (image >= 128).astype(np.uint8)
    image = 4 * image[:, :, 0] + 2 * image[:, :, 1] + image[:, :, 2]
    cat_image = np.zeros((2448,2448), dtype=np.uint8)
    cat_image[image == 3] = 1  # (Cyan: 011) Urban land
    cat_image[image == 6] = 2  # (Yellow: 110) Agriculture land
    cat_image[image == 5] = 3  # (Purple: 101) Rangeland
    cat_image[image == 2] = 4  # (Green: 010) Forest land
    cat_image[image == 1] = 5  # (Blue: 001) Water
    cat_image[image == 7] = 6  # (White: 111) Barren land
    cat_image[image == 0] = 0  # (Black: 000) Unknown


    # scipy.misc.imsave(output_path, cat_image)
    imageio.imsave(output_path, cat_image)
    imageio.imsave(new_img_path2, image2)
    pass

rgb_label_path = 'train'
new_img_path = 'DeepGlobe2018/land_train'
onechannel_path = 'DeepGlobe2018/onechannel_label'

filelist = os.listdir(rgb_label_path)
file_names = np.array([file.split('_')[0] for file in filelist if file.endswith('.png')], dtype=object)

for filename in tqdm(file_names):
    mask_path = os.path.join(rgb_label_path, filename + '_mask.png')
    label_path = os.path.join(onechannel_path, filename + '_label.png')
    new_img_path2 = os.path.join(new_img_path, filename + '_sat.jpg')
    color2annotation(mask_path, label_path, new_img_path2)
