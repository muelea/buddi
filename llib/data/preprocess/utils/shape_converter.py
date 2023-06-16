import torch
import smplx
import os.path as osp
import pickle 

class ShapeConverter():
    def __init__(
        self, 
        essentials_folder='essentials',
        inbm_type='smpl',
        outbm_type='smplx'
    ):
        """
        Class for converting betas between body models (e.g. SMPL to SMPL-X)
        Parameters
        ----------
        essentials_folder: str
            path to essentials folder
        inbm_type: str
            type of input body model
        outbm_type: str
            type of output body model
        """
        super().__init__()

        self.inbm_type = inbm_type
        self.outbm_type = outbm_type
        self.essentials_folder = essentials_folder

        assert self.inbm_type in ['smil', 'smpl', 'smpla'], 'Only SMPL to SMPL-X conversion is supported'
        assert self.outbm_type in ['smplx', 'smplxa'], 'Only SMPL to SMPL-X conversion is supported'

        self.smpltosmplx = self.load_smpltosmplx()
        self.inbm,  self.inshapedirs = self.load_body_model(model_type=self.inbm_type)
        self.outbm, self.outshapedirs = self.load_body_model(model_type=self.outbm_type)

    def load_smpltosmplx(self):
        smpl_to_smplx_path = osp.join(self.essentials_folder, f'body_model_utils/smpl_to_smplx.pkl')
        smpltosmplx = pickle.load(open(smpl_to_smplx_path, 'rb'))
        matrix = torch.tensor(smpltosmplx['matrix']).float()   
        return matrix             

    def load_body_model(self, model_type):
        if model_type in ['smpl', 'smplx']:
            model_folder = osp.join(self.essentials_folder, 'body_models')
            bm = smplx.create(model_path=model_folder, model_type=model_type)
            shapedirs = bm.shapedirs
        elif model_type == 'smpla':
            model_path = osp.join(self.essentials_folder, 'body_models/smpla/SMPLA_NEUTRAL.pth')
            bm = torch.load(model_path) 
            shapedirs = bm['smpla_shapedirs']
        elif model_type == 'smil':
            model_path = osp.join(self.essentials_folder, 'body_models/smil/smil_packed_info.pth')
            bm = torch.load(model_path) 
            shapedirs = bm['shapedirs']
        elif model_type == 'smplxa':
            model_folder = osp.join(self.essentials_folder, 'body_models')
            kid_template = osp.join(model_folder, 'smil/smplx_kid_template.npy')
            bm = smplx.create(
                model_path=model_folder, model_type='smplx',
                kid_template_path=kid_template, age='kid'
            )
            shapedirs = bm.shapedirs
            
        else:
            raise ValueError(f'Unknown model type {model_type}')

        return bm, shapedirs

    def forward(self, in_betas):
        """ Convert betas from input to output body model. """
        bs = in_betas.shape[0]

        # get shape blend shapes of input
        in_shape_displacement = smplx.lbs.blend_shapes(in_betas, self.inshapedirs)

        # find the vertices common between the in- and output model and map them
        in_shape_displacement = torch.einsum('nm,bmv->bnv', self.smpltosmplx, in_shape_displacement)
        in_shape_displacement = in_shape_displacement.view(bs, -1)
        out_shapedirs = self.outshapedirs.reshape(-1, self.outshapedirs.shape[-1])

        # solve for betas in least-squares sense
        lsq_arr = torch.matmul(torch.inverse(torch.matmul(
            out_shapedirs.t(), out_shapedirs)), out_shapedirs.t())

        out_betas = torch.einsum('ij,bj->bi', [lsq_arr, in_shape_displacement])

        return out_betas
