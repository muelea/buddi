import glob
import os 
import cv2
import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--m1', type=str, default='baseline/fit_baseline')
parser.add_argument('--m2', type=str, default='diffusion_prior/fit_diffusion_prior')
parser.add_argument('--result_folder', type=str, default='results/optimization')

args = parser.parse_args()

RESULT_FOLDER = args.result_folder 
M1_FOLDER = args.m1 #'baseline/fit_baseline'
M2_FOLDER = args.m2 #'diffusion_prior/fit_diffusion_prior'

# read images in M1 result folder 
m1_images = glob.glob(os.path.join(RESULT_FOLDER, M1_FOLDER, 'images', '*.png'))
# read images in M2 result folder
m2_images = glob.glob(os.path.join(RESULT_FOLDER, M2_FOLDER, 'images', '*.png'))
# find image names that esists in both folders
image_names = list(set([os.path.basename(x) for x in m1_images]).intersection([os.path.basename(x) for x in m2_images]))

# create folder for comparison
M1_OUT = M1_FOLDER.replace('/', '_')
M2_OUT = M2_FOLDER.replace('/', '_')
comparison_folder = os.path.join(RESULT_FOLDER, f'comparison_{M1_OUT}_{M2_OUT}')
if not os.path.exists(comparison_folder):
    os.makedirs(comparison_folder)

# stack the images and save them to the comparison folder
for image_name in image_names:
    m1_image = cv2.imread(os.path.join(RESULT_FOLDER, M1_FOLDER, 'images', image_name))
    m2_image = cv2.imread(os.path.join(RESULT_FOLDER, M2_FOLDER, 'images', image_name))
    stacked_image = np.vstack((m1_image, m2_image))
    cv2.imwrite(os.path.join(comparison_folder, image_name), stacked_image)
