import pickle
import torch 
import numpy as np
import torch.nn as nn
import os.path as osp
from llib.utils.threed.distance import pcl_pcl_pairwise_distance
from llib.utils.threed.intersection import winding_numbers


class MinDistLoss(nn.Module):
    def __init__(
        self,
        r2r_dist_type: str = 'min',
        squared_dist: bool = False,
        **kwargs
    ):
        super().__init__()
        """
        Minimize distance between two meshes.
        """
        self.squared=squared_dist
        assert r2r_dist_type in ['min'], f'Unknown distance type: {r2r_dist_type}'


    def forward(self, v1, v2, **args):

        squared_dist = pcl_pcl_pairwise_distance(
            v1, v2, squared=self.squared
        )

        loss = squared_dist.min()
        
        return loss

class ContactMapLoss(nn.Module):
    def __init__(
        self,
        r2r_dist_type: str = 'min',
        region_aggregation_type: str = 'sum',
        squared_dist: bool = False,
        region_to_vertex: str = '',
        **kwargs
    ):
        super().__init__()
        """
        Aggregated distance between multiple point clouds.
        """
        self.r2r_dist_type = r2r_dist_type
        self.region_aggregation_type = region_aggregation_type
        self.squared=squared_dist

        self.rid_to_vid = pickle.load(open(region_to_vertex, 'rb'))
        self.num_regions = len(self.rid_to_vid)
        self.rid_to_vid_lengths = np.array([len(v) for k, v in self.rid_to_vid.items()])

        self.downsample = True
        if self.downsample:
            np.random.seed(0)
            self.max_verts_per_region = 40

            random_sample = []
            for idx in range(len(self.rid_to_vid)):
                np.random.seed(0)
                sampled_index = np.random.choice(self.rid_to_vid[idx], self.max_verts_per_region, replace=False)
                random_sample.append(sampled_index)
            random_sample = torch.tensor(random_sample).to(torch.long)
            self.register_buffer('rid_to_vid_list', random_sample)

        self.criterion = self.init_loss()

    def aggregate_distance(self, x, dim=1):
        
        if self.r2r_dist_type == 'min':
            distances = x.min(dim)[0]
        elif self.r2r_dist_type == 'mean':
            distances = x.mean(dim)
        else:
            raise NotImplementedError(f'Unknown distance aggregation type: {self.dist_type}')
        return distances

    def region_to_region(self, v1, v2, r1, r2):


        v1r1 = v1[:,r1,:]
        v2r2 = v2[:,r2,:]
        
        squared_dist = pcl_pcl_pairwise_distance(
            v1r1, v2r2, squared=self.squared
        )

        v1r1_to_v2r2 = self.aggregate_distance(squared_dist, 1)
        v2r2_to_v1r1 = self.aggregate_distance(squared_dist, 2)

        return v1r1_to_v2r2, v2r2_to_v1r1      

    def get_full_heatmap(self, v1, v2, cmap=None, squared=False):
        
        batch_size = v1.shape[0]
        cmap = torch.ones(
            (batch_size, self.num_regions, self.num_regions)
        ).to(v1.device)

        batch_idx, h1_map, h2_map = torch.where(cmap)
        num_regions = batch_idx.shape[0]

        # select the regions of v1 and v2
        batch_idxs = batch_idx.unsqueeze(1).repeat((1, self.max_verts_per_region)).view(-1)
        r1_vid = self.rid_to_vid_list[h1_map]
        r2_vid = self.rid_to_vid_list[h2_map]
        v1r1 = v1[batch_idxs, r1_vid.view(-1),:].reshape(num_regions, -1, 3)
        v2r2 = v2[batch_idxs, r2_vid.view(-1),:].reshape(num_regions, -1, 3)

        squared_dist = pcl_pcl_pairwise_distance(
            v1r1, v2r2, squared=squared
        )
        squared_dist = squared_dist.view(
            (batch_size, self.num_regions, self.num_regions, self.max_verts_per_region, self.max_verts_per_region)
        ).view((batch_size, self.num_regions, self.num_regions, -1))

        heatmap = squared_dist.min(3)[0]

        return heatmap

    def init_loss(self):
        if self.downsample:
            def loss_func(v1, v2, cmap, factor=1):
                """
                Compute loss between region r1 on meshes v1 and 
                region r2 on mesh v2.
                """

                # in case old version is used
                if len(cmap.shape) == 2:
                    cmap = cmap.unsqueeze(0)

                batch_size = cmap.shape[0]

                distance = torch.zeros(batch_size).to(cmap.device)

                batch_idx, h1_map, h2_map = torch.where(cmap)
                num_regions = batch_idx.shape[0]

                if not num_regions == 0:
                    # select the regions of v1 and v2
                    batch_idxs = batch_idx.unsqueeze(1).repeat((1, self.max_verts_per_region)).view(-1)
                    r1_vid = self.rid_to_vid_list[h1_map]
                    r2_vid = self.rid_to_vid_list[h2_map]
                    v1r1 = v1[batch_idxs, r1_vid.view(-1),:].reshape(num_regions, -1, 3)
                    v2r2 = v2[batch_idxs, r2_vid.view(-1),:].reshape(num_regions, -1, 3)

                    squared_dist = pcl_pcl_pairwise_distance(
                        v1r1, v2r2, squared=self.squared
                    )
                    v1r1_to_v2r2 = self.aggregate_distance(squared_dist, 1)
                    v2r2_to_v1r1 = self.aggregate_distance(squared_dist, 2)

                    region_dist_loss = ((factor * v1r1_to_v2r2)**2).mean(1) + \
                                ((factor * v2r2_to_v1r1)**2).mean(1)

                    # aggregrate regions
                    # not deterministic that's why we're using a for loop instead
                    #distancenondet = torch.index_add(distance, 0, batch_idx, region_dist_loss)
                    for ii in range(batch_size):
                        if self.region_aggregation_type == 'sum':      
                            distance[ii] = region_dist_loss[batch_idx == ii].sum()
                        else:
                            raise NotImplementedError(f'Unknown region aggregation type: {self.region_aggregation_type}')
    
                return distance

        else:
            def loss_func(v1, v2, cmap, factor=1):
                """
                Compute loss between region r1 on meshes v1 and 
                region r2 on mesh v2.
                """
                h1_map, h2_map = torch.where(cmap)

                losses = []

                for r1, r2 in zip(h1_map, h2_map):
                    r1_vid = self.rid_to_vid[r1.item()]
                    r2_vid = self.rid_to_vid[r2.item()]

                    v1r1_to_v2r2, v2r2_to_v1r1 = self.region_to_region(v1, v2, r1_vid, r2_vid)

                    region_dist_loss = ((factor * v1r1_to_v2r2)**2).mean() + \
                            ((factor * v2r2_to_v1r1)**2).mean()
                    losses.append(region_dist_loss)
                
                return losses

        return loss_func 

    def forward(self, **args):
        loss = self.criterion(**args)
        #loss = self.aggregate_regions(losses)
        return loss


class GeneralContactLoss(nn.Module):
    def __init__(
        self,
        region_aggregation_type: str = 'sum',
        squared_dist: bool = False,
        model_type: str = 'smplx',
        body_model_utils_folder: str = 'body_model_utils',
        **kwargs
    ):
        super().__init__()
        """
        Compute intersection and contact between two meshes and resolves.
        """
    
        self.region_aggregation_type = region_aggregation_type
        self.squared=squared_dist

        self.criterion = self.init_loss()       

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

    def to_lowres(self, v, n=100):
        lrm = self.low_res_mesh[n]
        v = self.close_mouth(v)
        v = v[:,lrm['smplx_vid'],:]
        t = v[:,lrm['faces'].astype(np.int32),:]
        return v, t


    def init_loss(self):
        def loss_func(v1, v2, factor=1000, wn_batch=True):
            """
            Compute loss between region r1 on meshes v1 and 
            region r2 on mesh v2.
            """

            nn = 1000

            loss = torch.tensor(0.0, device=v1.device)

            if wn_batch:
                # close mouth for self-intersection test
                v1l, t1l = self.to_lowres(v1, nn)
                v2l, t2l = self.to_lowres(v2, nn)

                # compute intersection between v1 and v2
                interior_v1 = winding_numbers(v1, t2l).ge(0.99)
                interior_v2 = winding_numbers(v2, t1l).ge(0.99)

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
            
            batch_losses = []
            for bidx in range(v1.shape[0]):
                if not wn_batch:
                    # close mouth for self-intersection test
                    v1l, t1l = self.to_lowres(v1[[bidx]], nn)
                    v2l, t2l = self.to_lowres(v2[[bidx]], nn)

                    # compute intersection between v1 and v2
                    curr_interior_v1 = winding_numbers(v1[[bidx]], t2l.detach()).ge(0.99)[0]
                    curr_interior_v2 = winding_numbers(v2[[bidx]], t1l.detach()).ge(0.99)[0]
                    crit_v1, crit_v2 = torch.any(curr_interior_v1), torch.any(curr_interior_v2)
                else:
                    curr_interior_v1 = interior_v1[bidx]
                    curr_interior_v2 = interior_v2[bidx]
                    crit_v1, crit_v2 = torch.any(interior_v1[bidx]), torch.any(interior_v2[bidx])

                bloss = torch.tensor(0.0, device=v1.device)
                if crit_v1 and crit_v2:
                    # find vertices that are close to each other between v1 and v2
                    #squared_dist = pcl_pcl_pairwise_distance(
                    #    v1[:,interior_v1[bidx],:], v2[:, interior_v2[bidx], :], squared=self.squared
                    #)
                    squared_dist_v1v2 = pcl_pcl_pairwise_distance(
                        v1[[[bidx]],curr_interior_v1,:], v2[[bidx]], squared=self.squared)
                    squared_dist_v2v1 = pcl_pcl_pairwise_distance(
                        v2[[[bidx]], curr_interior_v2, :], v1[[bidx]], squared=self.squared)
 
                    v1_to_v2 = (squared_dist_v1v2[0].min(1)[0] * factor)**2
                    #v1_to_v2 = 10.0 * (torch.tanh(v1_to_v2 / 10.0)**2)
                    bloss += v1_to_v2.sum()
                    
                    v2_to_v1 = (squared_dist_v2v1[0].min(1)[0] * factor)**2
                    #v2_to_v1 = 10.0 * (torch.tanh(v2_to_v1 / 10.0)**2)
                    bloss += v2_to_v1.sum()

                batch_losses.append(bloss)

            # compute loss
            if len(batch_losses) > 0:
                loss = sum(batch_losses) / len(batch_losses)   

            #if loss > 1.0:
            #    import ipdb; ipdb.set_trace()

            return loss

        return loss_func 

    def forward(self, **args):
        losses = self.criterion(**args)
        return losses