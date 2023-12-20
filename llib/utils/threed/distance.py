import torch 
import pickle 
import numpy as np
import torch.nn as nn

def pcl_pcl_pairwise_distance(
    x: torch.Tensor, 
    y: torch.Tensor,
    use_cuda: bool = True,
    squared: bool = False
):
    """
    Calculate the pairse distance between two point clouds.
    """
    
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()

    dtype = torch.cuda.LongTensor if \
        use_cuda else torch.LongTensor

    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))

    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz

    if not squared:
        P = torch.clamp(P, min=0.0) # make sure we dont get nans
        P = torch.sqrt(P)
    
    return P


class ContactMap(nn.Module):
    def __init__(
        self,
        region_to_vertex: str = '',
    ):
        super().__init__()
        """
        Aggregated distance between multiple point clouds.
        """

        self.rid_to_vid = pickle.load(open(region_to_vertex, 'rb'))
        self.num_regions = len(self.rid_to_vid)
        self.rid_to_vid_lengths = np.array([len(v) for k, v in self.rid_to_vid.items()])

        self.downsample = True
        if self.downsample:
            np.random.seed(0)
            self.max_verts_per_region = 5

            random_sample = []
            for idx in range(len(self.rid_to_vid)):
                np.random.seed(0)
                sampled_index = np.random.choice(self.rid_to_vid[idx], self.max_verts_per_region, replace=False)
                random_sample.append(sampled_index)
            random_sample = torch.tensor(random_sample).to(torch.long)
            self.register_buffer('rid_to_vid_list', random_sample)

    def get_full_heatmap(self, v1, v2):
      
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
            v1r1, v2r2, squared=False
        )
        squared_dist = squared_dist.view(
            (batch_size, self.num_regions, self.num_regions, self.max_verts_per_region, self.max_verts_per_region)
        ).view((batch_size, self.num_regions, self.num_regions, -1))

        heatmap = squared_dist.min(3)[0]

        return heatmap