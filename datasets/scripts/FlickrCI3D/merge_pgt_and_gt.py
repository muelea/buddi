import pickle
import os, json
import torch

"""
After downloading the FlickrCI3D PGT and FlickrCI3D images and contact maps, 
we merge them to avoid processing everything twice.
"""

GT_FOLDER = "datasets/original/FlickrCI3D_Signatures"
PGT_FOLDER = "datasets/processed/FlickrCI3D_Signatures_Transformer"

def merge_data(gt, pgt):
    """
    Read the ground-truth annotation from gt and merge it with pgt data
    """

    out = []
    for x in pgt:
        imgname = x['imgname'][:-4]
        contact_idx = x['contact_index']

        # get the ground truth annotation for this image
        current = gt[imgname]
        cisig = current['ci_sign'][contact_idx]
        region_id = cisig['smplx']['region_id']
        x['hhc_contacts_region_ids'] = region_id
        
        # create contact map
        contact_map = torch.zeros(75, 75).to(torch.bool)
        for rid in region_id:
            contact_map[rid[0], rid[1]] = True
        x['contact_map'] = contact_map
        
        out.append(x)
    
    return out

for fn in ['val', 'train']:
    # load interaction contact signatures form GT folder 
    if fn == 'val' or fn == 'train':
        csig_path = os.path.join(GT_FOLDER, 'train', 'interaction_contact_signature.json')
    else:
        csig_path = os.path.join(GT_FOLDER, 'test', 'interaction_contact_signature.json')
    gt_annotation = json.load(open(csig_path, 'r'))

    # load pgt_data_in
    pgt_data_path_in = os.path.join(PGT_FOLDER, fn + '_pgt.pkl')
    pgt_data = pickle.load(open(pgt_data_path_in, 'rb'))

    # merge data
    data = merge_data(gt_annotation, pgt_data)
    
    # overwrite pkl file
    pgt_data_path_out = os.path.join(PGT_FOLDER, fn + '.pkl')
    with open(pgt_data_path_out, 'wb') as f:
        pickle.dump(data, f)
