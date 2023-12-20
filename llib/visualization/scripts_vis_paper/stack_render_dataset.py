import argparse
import os
import os.path as osp
import sys
import numpy as np
import cv2


def parse_args():

    parser = argparse.ArgumentParser(description='Render a dataset')
    parser.add_argument('--root-folder', help='folder with images to be stacked')
    parser.add_argument('--methods', nargs="*", help='methods to be stacked')
    parser.add_argument('--views', nargs="*", help='image views to be stacked')
    parser.add_argument('--small-image-size', default=224, 
        help='size of one image in the stack. Final image will be of size 1 + small_image_size * len(methods) * len(views)')
    parser.add_argument(
        "--selection",
        default=None,
        nargs="*",
        help="select specific images in dataset by name, e.g. girls_113749_0",
    )
    parser.add_argument('--out-fn', default='stacked_supmat', help='output filename')
    parser.add_argument('--header-text', nargs="*", help='text to be added to the header (BEV, VAE, Ours etc.)')
    args = parser.parse_args()
    return args

def process_method(folder, image_names, views, size=244, crop_x=['side_01','side_05','top']):
    """
    Read images from a method folder
    ----------
    Returns:
        all_images: list of list of views of images
    """
    all_images = []
    for img_name in image_names:
        images = []
        for view in views:
            img_path = osp.join(folder, img_name, f'{img_name}_{view}.png')
            img = cv2.imread(img_path)
            img = cv2.resize(img, (size, size))
            if view in crop_x:
                img = img[:, 50:size-50, :]
            images.append(img)
        all_images.append(images)
    return all_images

def main(args):

    # read image names from files if provided
    selection = args.selection
    if len(selection) == 1 and selection[0].endswith(".txt"):
        with open(selection[0], "r") as f:
            selection = [line.strip() for line in f.readlines()]

    views = args.views
    methods = args.methods
    header_text = args.header_text
    size = args.small_image_size

    # read image names from folders if provided
    all_images = []

    # read the source image without overlay from one folder
    base_method = methods[0]
    base_method_folder = osp.join(args.root_folder, base_method)
    for img in selection:
        img_path = osp.join(base_method_folder, img, f'{img}_src_img_crop.png')
        img = cv2.resize(cv2.imread(img_path), (size, size))
        all_images.append([img])

    # read the rest of the images from the other folders
    for method in methods:
        method_folder = osp.join(args.root_folder, method)
        method_all_images = process_method(method_folder, selection, views, size)
        for idx, item in enumerate(method_all_images):
            all_images[idx] += item

    # create header
    header = 255 * np.ones((int(size / 2), size, 3))
    # add text to header
    # get cv2 font (times new roman
    font = cv2.FONT_HERSHEY_COMPLEX
    crop_size = all_images[0][2].shape[1]
    for mm_text, mm in zip(header_text, methods):
        mwidth = size + crop_size*(len(views) - 1)
        mheader = 255 * np.ones((int(size / 2), mwidth, 3))
        # put text on mheader, center text in mheader
        textsize = cv2.getTextSize(mm_text, font, 1, 2)[0]
        textX = (mwidth - textsize[0]) // 2
        textY = (int(size / 2) + textsize[1]) // 2
        cv2.putText(mheader, mm_text, (textX, textY), font, 1, (0, 0, 0), 2)
        header = np.hstack([header, mheader])
    images = np.vstack([header] + [np.hstack(x) for x in all_images])
    cv2.imwrite(f"{args.root_folder}/{args.out_fn}.png", images)

if __name__ == "__main__":
    args = parse_args()
    main(args)