import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from .losses import Losses 
from .optimizer import Optimizer
from typing import List, Optional
from dataclasses import field


@dataclass 
class HMR:
    load_pretrained: bool = True
    pretrained_type: str = 'resnet50'
    smpl_mean_params: str = 'essentials/spin/smpl_mean_params.npz'

@dataclass
class ResNet:
    depth: int = 50
    pretrained: bool = True

@dataclass
class MLP:
    in_channels: int = 10
    hidden_channels: List[int] = field(default_factory=lambda: [8, 6, 4, 2])
    bias: bool = True
    dropout: float = 0.0

@dataclass 
class AE:
    in_channels: int = 5625
    num_neurons: int = 4000
    hidden_channels: int = 1000
    dropout: float = 0.1

@dataclass
class AutoEncoder:
    in_channels: int = 5625
    hidden_channels: int = 1000
    depth: int = 4

@dataclass
class ViT:
    image_size: int = 224
    patch_size: int = 16 
    num_classes: int = 1000
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.1
    dim_head: int = 64

@dataclass
class Transformer:
    dim: int = 146
    depth: int = 6
    heads: int = 1
    mlp_dim: int = 2048
    dropout: float = 0.1
    

@dataclass
class DiffusionTransformer:
    dim: int = 146
    depth: int = 6
    heads: int = 1
    mlp_dim: int = 2048
    dropout: float = 0.1
    use_positional_encoding: bool = False # sinoid positional encoding (not learnable)
    use_positional_embedding: bool = False # learnable positional embedding
    use_human_embedding: bool = False # concatenate human embedding to transformer input
    use_param_embedding: bool = False # concate parameter embedding to transformer input
    max_tokens: int = 100
    use_cross_attention: bool = False
    share_linear_layers: bool = False
    encode_target: bool = True


@dataclass
class VAE_MLP:
    d_model: Optional[int] = None
    d_latent: int = 64
    d_features: int = 256
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dropout: float = 0.1
    embed_features: bool = False
    embed_id: bool = False
    enc_pos: bool = False

@dataclass
class AE_MLP:
    d_model: Optional[int] = None
    d_latent: int = 64
    d_features: int = 256
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dropout: float = 0.1
    embed_features: bool = False
    embed_id: bool = False
    enc_pos: bool = False

@dataclass 
class BEV:
    bv_with_fv_condition: bool = True
    add_offsetmap: bool = True
    add_depth_encoding: bool = True

    # backbone
    backbone_type: str = 'hrnet_32'
    hrnet_pretrain_path: str = 'hrbet_32.pth'
    # build bev
    params_num: int = 146
    outmap_size: int = 128
    cam_dim: int = 3
    num_center_maps: int = 1
    coord_maps_size: int = 128
    centermap_size: int = 64

    #head
    head_block_num: int = 2
    head_num_channels: int = 128
    #bv_center
    bv_center_num_block: int = 2
    bv_center_momentum: float = 0.1
    #transformer
    transformer_dropout_ratio: float = 0.2
    transformer_num_channels: int = 512

    # result parser
    max_person: int = 12
    conf_thresh: float = 0.1

@dataclass
class HHCC:
    pretrain_path: str = 'essentials/bev/BEV.pth'

@dataclass
class ExperimentSetup:
    contact_rep: str = 'contact_map' # contact representation (contact_map, contact_heat)
    rotrep: str = 'sixd' # pose and global orient rotation representation
    in_data: str = 'bev' # the transformer input are bev params
    token_setup: str = '' # how to build the transformer input
    num_contact_tokens: int = 25 # in how many tokens should the contact map be split
    # list which smpl params to mask out in smpl_mask
    # options: orient_h0, orient_h1, pose_h0, pose_h1, shape_h0, shape_h1, transl_h0, transl_h1
    smpl_mask: List[str] = field(default_factory=lambda: [])
    relative_transl: bool = True # use relative translation instead of absolute
    relative_orient: bool = True # use relative global orientation instead of absolute
    guidance_params:List[str] = field(default_factory=lambda: []) # e.g. 'orient_h1'
    # with some probability, set the guidance params to random noise
    # guidance_all_nc + guidance_no_nc + guidance_param_nc = 1.0
    guidance_param_nc: float = 0.1 # change for a single guidance parameter to be set to random noise
    guidance_all_nc: float = 0.1 # change for all guidance parameters to be set to random noise
    guidance_no_nc: float = 0.8 # change that no guidance parameter is set to random noise

@dataclass 
class Regressor():
    type: str = 'transformer'
    
    optimizer: Optimizer = Optimizer()
    losses: Losses = Losses()
    experiment: ExperimentSetup = ExperimentSetup()

    # list all regressors here
    hmr: HMR = HMR()
    resnet: ResNet = ResNet()
    mlp: MLP = MLP()
    vit: ViT = ViT()
    bev: BEV = BEV()
    ae: AE = AE()
    transformer: Transformer = Transformer()
    diffusion_transformer: DiffusionTransformer = DiffusionTransformer()
    hhcc: HHCC = HHCC()
    vae_mlp: VAE_MLP = VAE_MLP()
    autoencoder: AutoEncoder = AutoEncoder()
    ae_mlp: AE_MLP = AE_MLP()
