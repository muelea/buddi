import numpy as np

def iou(a, b):
    """
    Compute intersection over union for two bounding boxes. Bounding
    box format: [x_lower_left, y_lower_left, x_upper_right, y_upper_right]
    a: 1x4 array defining bounding box
    b: 1x4 array defining bounding box
    return IoU value
    """

    c0x = max(a[0], b[0])
    c1x = min(a[2], b[2])
    c0y = max(a[1], b[1])
    c1y = min(a[3], b[3])

    x_length_inner = max(0, c1x - c0x)
    y_length_inner = max(0, c1y - c0y)
    intersection = x_length_inner * y_length_inner

    a_size = (a[2] - a[0]) * (a[3] - a[1])
    b_size = (b[2] - b[0]) * (b[3] - b[1])
    union = a_size + b_size - intersection

    IoU = intersection / union 
    
    return IoU


def iou_matrix(a, b):
    """
    Computer the intersection over untion of two arrays of bounding boxes. 
    Bounding box format: [x_lower_left, y_lower_left, x_upper_right, y_upper_right]
    a: Nx4 array of N bounding boxes
    b: Mx4 array of N bounding boxes 
    return matrix of size NxM of IoU values
    """

    iou_matrix = np.zeros((a.shape[0], b.shape[0]))

    for a_idx, a_bb in enumerate(a):
        for b_idx, b_bb in enumerate(b):
            iou_matrix[a_idx, b_idx] = iou(a_bb, b_bb)

    # find best match for each bounding box in a
    best_match = iou_matrix.argmax(1)
    
    return iou_matrix, best_match
