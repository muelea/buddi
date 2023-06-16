import torch 
import torch.nn as nn
import numpy as np
import pickle
import os.path as osp 
from typing import NewType, List, Union, Tuple, Optional
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.utils.threed.intersection import winding_numbers

class ContactMapDistError(nn.Module):
    def __init__(
        self, 
        name,
        region_to_vertex: str = '',
    ):
        super(ContactMapDistError, self).__init__()

        self.rid_to_vid = pickle.load(open(region_to_vertex, 'rb'))

    def forward(self, v1s, v2s, cmaps):
        batch_min_dist = []
        for i in range(v2s.shape[0]):
            min_dists = []
            v1, v2, cmap = v1s[[i]], v2s[[i]], cmaps[i]
            h1_map, h2_map = torch.where(cmap)
            for r1, r2 in zip(h1_map, h2_map):
                r1_vid, r2_vid = self.rid_to_vid[r1.item()], self.rid_to_vid[r2.item()]
                dists = pcl_pcl_pairwise_distance(v1[:,r1_vid,:], v2[:,r2_vid,:], squared=False)
                min_dists.append(dists.min())
            batch_min_dist.append(torch.stack(min_dists).mean())
        return torch.stack(batch_min_dist)


class ContactIOU(nn.Module):
    def __init__(
        self,
        name,
    ):
        super(ContactIOU, self).__init__()

        self.eps = 1e-6


    def forward(self, input_cmap, target_cmap):

        bs = input_cmap.shape[0]
        input_cmap = input_cmap.reshape(bs, -1)
        target_cmap = target_cmap.reshape(bs, -1) 
        intersection = (input_cmap * target_cmap).sum(1)
        union = (input_cmap | target_cmap).sum(1)
        iou = intersection / union

        true_positives = (input_cmap * target_cmap).sum(1)
        false_positives = (input_cmap * (~target_cmap)).sum(1)
        false_negatives = ((~input_cmap) * target_cmap).sum(1)
        precision = true_positives / (true_positives + false_positives + self.eps)
        recall = true_positives / (true_positives + false_negatives + self.eps)
        fsore = 2 * (precision * recall) / (precision + recall + self.eps)

        return iou, precision, recall, fsore


class MaxIntersection(nn.Module):
    def __init__(
        self,
        model_type: str = 'smplx',
        body_model_utils_folder: str = 'body_model_utils',
    ):
        super().__init__()
        """
        Compute intersection and contact between two meshes and resolves.
        """ 

        # create extra vertex and faces to close back of the mouth to maske
        # the smplx mesh watertight.
        self.model_type = model_type
        faces = torch.load(
            osp.join(body_model_utils_folder, f'{model_type}_faces.pt')
        )

        if self.model_type == 'smplx':
            max_face_id = faces.max().item() + 1
            inner_mouth_verts = pickle.load(open(
                f'{body_model_utils_folder}/smplx_inner_mouth_bounds.pkl', 'rb')
            )
            vert_ids_wt = torch.tensor(inner_mouth_verts[::-1]) # invert order 
            self.register_buffer('vert_ids_wt', vert_ids_wt)
            faces_mouth_closed = [] # faces that close the back of the mouth
            for i in range(len(vert_ids_wt)-1):
                faces_mouth_closed.append([vert_ids_wt[i], vert_ids_wt[i+1], max_face_id])
            faces_mouth_closed = torch.tensor(np.array(faces_mouth_closed).astype(np.int64), dtype=torch.long, device=faces.device)
            faces = torch.cat((faces, faces_mouth_closed), 0)
         
        self.register_buffer('faces', faces)

        # low resolution mesh 
        inner_mouth_verts_path = f'{body_model_utils_folder}/lowres_{model_type}.pkl'
        self.low_res_mesh = pickle.load(open(inner_mouth_verts_path, 'rb'))
        

    def triangles(self, vertices):
        # get triangles (close mouth for smplx)

        if self.model_type == 'smplx':
            mouth_vert = torch.mean(vertices[:,self.vert_ids_wt,:], 1,
                        keepdim=True)
            vertices = torch.cat((vertices, mouth_vert), 1)

        triangles = vertices[:,self.faces,:]

        return triangles

    def close_mouth(self, v):
            mv = torch.mean(v[:,self.vert_ids_wt,:], 1, keepdim=True)
            v = torch.cat((v, mv), 1)
            return v

    def to_lowres_close_mesh(self, v, n=100):
        lrm = self.low_res_mesh[n]
        v = self.close_mouth(v)
        v = v[:,lrm['smplx_vid'],:]
        t = v[:,lrm['faces'].astype(np.int32),:]
        return v, t
    
    def close_mesh(self, v, faces):
        vc = self.close_mouth(v)
        tc = vc[:,faces,:]
        return vc, tc

    def prep_mesh(self, v, n=None):
        if n is None:
            verts, triangles = self.close_mesh(v, self.faces)
        else:
            verts, triangles = self.to_lowres_close_mesh(v, n)        
        return verts, triangles
    
    def forward_loop(self, v1, v2, lowres):
        for bidx in range(v1.shape[0]):
            batch_out = self.forward_batch(v1[[bidx]], v2[[bidx]], lowres)
            if bidx == 0:
                out = batch_out
            else:
                for k in batch_out:
                    out[k] += batch_out[k]
        return out
       
    def forward_batch(self, v1, v2, lowres=None):
        out = {'min_dist': []}
        for aggr in ['max', 'mean']: #, 'median']:
            out[f'{aggr}_v1_in_v2'] =[]
            out[f'{aggr}_v2_in_v1'] =[]

        # get low resolution meshes for fast intersection computation
        _, t1l = self.prep_mesh(v1, lowres)
        _, t2l = self.prep_mesh(v2, lowres)

        # compute intersection between v1 and v2
        interior_v1 = winding_numbers(v1, t2l).ge(0.99)
        interior_v2 = winding_numbers(v2, t1l).ge(0.99)
         
        for bidx in range(v1.shape[0]):

            # compute distance between v1 and v2
            v1v2 = pcl_pcl_pairwise_distance(
                v1[[bidx]], v2[[bidx]], squared=False)
            try:
                # minimum distance between v1 and v2
                out['min_dist'].append(v1v2.min().item())

                # intersection distances from v1 vertices to v2
                max_val, mean_val, median_val = 0.0, 0.0, 0.0
                if interior_v1[bidx].any():
                    v1_to_v2 = v1v2[:,interior_v1[bidx],:].min(2)[0]
                    max_val = v1_to_v2.max().item()
                    mean_val = v1_to_v2.mean().item()
                    median_val = np.median(v1_to_v2.cpu()).item()
                out['max_v1_in_v2'].append(max_val)
                out['mean_v1_in_v2'].append(mean_val)
                #out['median_v1_in_v2'].append(median_val)

                max_val, mean_val, median_val = 0.0, 0.0, 0.0
                if interior_v2[bidx].any():
                    v2_to_v1 = v1v2[:,:,interior_v2[bidx]].min(1)[0]
                    max_val = v2_to_v1.max().item()
                    mean_val = v2_to_v1.mean().item()
                    median_val = np.median(v2_to_v1.cpu()).item()
                out['max_v2_in_v1'].append(max_val)
                out['mean_v2_in_v1'].append(mean_val)
                #out['median_v2_in_v1'].append(median_val)
            except:
                import ipdb; ipdb.set_trace()
        return out
     
    def forward(self, v1, v2, lowres=None, in_loop=False):
        return self.forward_batch(v1, v2, lowres) if not in_loop \
            else self.forward_loop(v1, v2, lowres)

# visu results 
'''
if True:
    import trimesh 
    mesh = trimesh.Trimesh(
        vertices=v2l[0].detach().cpu().numpy(), 
        faces=self.low_res_mesh[nn]['faces'].astype(np.int32),
    )
    col = 255 * np.ones((10476, 4))
    inside_idx = torch.where(interior_v2[0])[0].detach().cpu().numpy()
    col[inside_idx] = [0, 255, 0, 255]
    mesh.visual.vertex_colors = col
    _ = mesh.export('outdebug/interior_v2_lowres.obj')

    vmc = self.close_mouth(v1)
    mesh = trimesh.Trimesh(vertices=vmc[0].detach().cpu().numpy(), faces=self.faces.cpu().numpy())
    col = 255 * np.ones((10476, 4))
    inside_idx = torch.where(interior_v1[0])[0].detach().cpu().numpy()
    col[inside_idx] = [255, 0, 0, 255]
    mesh.visual.vertex_colors = col
    _ = mesh.export('outdebug/interior_v1_highres.obj')

    # Other direction
    mesh = trimesh.Trimesh(
        vertices=v1l[0].detach().cpu().numpy(), 
        faces=self.low_res_mesh[nn]['faces'].astype(np.int32),
    )
    col = 255 * np.ones((10476, 4))
    inside_idx = torch.where(interior_v1[0])[0].detach().cpu().numpy()
    col[inside_idx] = [0, 255, 0, 255]
    mesh.visual.vertex_colors = col
    _ = mesh.export('outdebug/interior_v1_lowres.obj')

    vmc = self.close_mouth(v2)
    mesh = trimesh.Trimesh(vertices=vmc[0].detach().cpu().numpy(), faces=self.faces.cpu().numpy())
    col = 255 * np.ones((10476, 4))
    inside_idx = torch.where(interior_v2[0])[0].detach().cpu().numpy()
    col[inside_idx] = [255, 0, 0, 255]
    mesh.visual.vertex_colors = col
    _ = mesh.export('outdebug/interior_v2_highres.obj')
'''



