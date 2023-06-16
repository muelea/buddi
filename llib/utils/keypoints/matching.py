import numpy as np 

COST_CONST = 999.99 # cost for missing keypoints, inf doesn't work

def keypoint_cost(kp1, kp2, kp1_conf, kp2_conf, norm=1.0, conf_thres=0.6):
    """ 
    Compute the cost of the keypoints, considering 
    the confidence of each keypoint and a norm factor.
    """
   
    # only consider keypoints with confidence > 0.0
    mask = (kp1_conf > conf_thres) * (kp2_conf > conf_thres)

    # normalize keypoints by image size
    kp1, kp2 = kp1 / norm, kp2 / norm

    if mask.sum() > 1:
        # euclidean distance
        kp1kp2vec = kp1[mask][:,:2] - kp2[mask][:,:2]
        dist = np.linalg.norm(kp1kp2vec, axis=1)
        cost = dist.mean()
    else:
        cost = COST_CONST

    return cost

def keypoint_cost_matrix(kpts1, kpts2, norm=1.0, unique_best_matches=False):
    """
    Compute cost matrix between two sets of keypoints. 
    Return the best matching keypoints and the cost.
    """
    matrix = np.zeros((len(kpts1), len(kpts2)))
    for idx1, kp1 in enumerate(kpts1):
        for idx2, kp2 in enumerate(kpts2):
            matrix[idx1, idx2] = keypoint_cost(
                kp1[:,:2], kp2[:,:2], kp1[:,2], kp2[:,2], norm=norm
            )
    # for each keypoint in kpts1 find the best match in kpts2
    keypoint_cost_best_match = matrix.argmin(1)
    if unique_best_matches:
        if matrix.shape[0] == 2:
            best_match_counts = np.unique(keypoint_cost_best_match, return_counts=True)
            if np.any(best_match_counts[1] != 1):
                
                # create all possible combinations of matches
                all_matches = np.array(np.meshgrid(*[np.arange(len(kpts1)), np.arange(len(kpts2))])).T.reshape(-1,2)
                # filter rows with duplicated elements
                has_duplicate = np.array([np.any(np.unique(x, return_counts=True)[1] > 1) for x in all_matches])
                all_matches = all_matches[~has_duplicate]
                costs = []
                for match in all_matches:
                    # gather all elements of matrix by column index specified in match
                    elements = [matrix[ridx, cidx] for ridx, cidx in enumerate(match)]
                    costs.append(sum(elements))
                # get the best match
                keypoint_cost_best_match = all_matches[np.argmin(np.array(costs))]
            else:
                print("unique best matches only possible when max. two people on image. Ignoring.")

    # ignore keypoints with not enough matches
    if not unique_best_matches: 
        ignore_mask = (matrix == COST_CONST).all(1)
        if ignore_mask.sum() > 0:
            keypoint_cost_best_match[ignore_mask] = -1
    
    return matrix, keypoint_cost_best_match