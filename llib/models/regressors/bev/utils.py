import numpy as np
import cv2

def padding_image(image):
    h, w = image.shape[:2]
    side_length = max(h, w)
    pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    top, left = int((side_length - h) // 2), int((side_length - w) // 2)
    bottom, right = int(top+h), int(left+w)
    pad_image[top:bottom, left:right] = image
    image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
    return pad_image, image_pad_info
    
def img_preprocess(image, input_size=512, return_pad_img=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_image, image_pad_info = padding_image(image)
    input_image = torch.from_numpy(cv2.resize(pad_image, (input_size,input_size), interpolation=cv2.INTER_CUBIC))[None].float()
    if return_pad_img:
        return input_image, image_pad_info, pad_image
    else:
        return input_image, image_pad_info

def bbox_preprocess(bbox, image_pad_info, pad_image, image_size=512):
    bbox[1] += image_pad_info[0]
    bbox[3] += image_pad_info[0]
    bbox[0] += image_pad_info[2]
    bbox[2] += image_pad_info[2]
    
    bbox[[1,3]] = bbox[[1,3]] / pad_image.shape[1] * image_size
    bbox[[0,2]] = bbox[[0,2]] / pad_image.shape[0] * image_size

    return bbox
