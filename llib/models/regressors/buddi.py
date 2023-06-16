# transformer model that takes smpl parameters as input and outputs smpl parameters
# Like ViT, but without patches and their linear projection 

import torch
from torch import nn
import numpy as np
import math

class TimestepEncoding(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        """
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".

        # from Diffusion original code 
        # https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py
        """

        if latent_dim % 2 == 1:  # only allow even latent_dim
            raise NotImplementedError

        half_dim = latent_dim // 2
        pos = torch.arange(half_dim)
        x = math.log(10000) / (half_dim - 1)
        emb = torch.exp(pos * -x)

        self.register_buffer('emb', emb)

    def forward(self, timesteps):
        encoding = timesteps[:, None] * self.emb[None, :]
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], axis=1)
        return encoding


class PositionalEncoding(nn.Module):
    """
    Positional encoding form attention is all you need.
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    """

    def __init__(self, dim, max_positions=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_positions, dim))

    def _get_sinusoid_encoding_table(self, max_positions, dim):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_positions)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.shape[1]].clone().detach()

class PositionalEmbedding(nn.Module):
    """ Learned parameter positional embedding."""
    def __init__(self, dim, max_positions=200):
        super(PositionalEmbedding, self).__init__()
        self.embed = nn.Parameter(torch.randn(1, max_positions, dim))
    def forward(self, x):
        return x + self.embed[:, :x.shape[1]]

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encode = TimestepEncoding(latent_dim)

        self.embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        #assert len(timesteps.shape) == 2, "timesteps must be 2D tensor"
        # encode timesteps 
        tenc = self.encode(timesteps).unsqueeze(1)
        temb = self.embed(tenc)
        return temb


class GuidanceEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x):
        return self.embed(x)

class BUDDI(nn.Module):
    def __init__(self, 
        dim, 
        depth, 
        heads, 
        mlp_dim,
        dropout=0.0, 
        embed_guidance_config = {
            'orient_h0': 6, 'pose_h0': 126, 'shape_h0': 11, 'transl_h0': 3,
            'orient_h1': 6, 'pose_h1': 126, 'shape_h1': 11, 'transl_h1': 3,
            'orient_bev_h0': 6, 'pose_bev_h0': 126, 'shape_bev_h0': 11, 'transl_bev_h0': 3,
            'orient_bev_h1': 6, 'pose_bev_h1': 126, 'shape_bev_h1': 11, 'transl_bev_h1': 3,
            'human_h0': 146, 'human_h1': 146, 'action': 1, 'contact': 5625, # 75 * 75
        },
        embed_target_config = {
            'orient_h0': 6, 'pose_h0': 126, 'shape_h0': 11, 'transl_h0': 3,
            'orient_h1': 6, 'pose_h1': 126, 'shape_h1': 11, 'transl_h1': 3,
            'human_h0': 146, 'human_h1': 146,
        },
        use_human_embedding=False,
        use_param_embedding=False,
        use_positional_encoding=False,
        use_positional_embedding=False,
        use_cross_attention=False,
        share_linear_layers=False,
        encode_target=False,
        max_tokens=100,
    ):
        super().__init__()
        self.heads = heads
        self.depth = depth

        self.use_human_embedding = use_human_embedding
        self.use_param_embedding = use_param_embedding
        self.use_positional_encoding = use_positional_encoding
        self.use_positional_embedding = use_positional_embedding
        self.use_cross_attention = use_cross_attention
        self.encode_target = encode_target
        self.share_linear_layers = share_linear_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, 
            dim_feedforward=mlp_dim, 
            dropout=dropout, 
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)

        if self.use_cross_attention:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, 
                dim_feedforward=mlp_dim, 
                dropout=dropout,
                batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=depth)
        
        # timestep embedding
        self.embed_timestep = TimestepEmbedder(dim)

        # guidance embedding
        for embed_name, embed_dim in embed_guidance_config.items():
            setattr(self, f'embed_guidance_{embed_name}', GuidanceEmbedder(embed_dim, dim))

        # target parameter embedding and mlp for final params
        for embed_name, embed_dim in embed_target_config.items():
            setattr(self, f'embed_input_{embed_name}', GuidanceEmbedder(embed_dim, dim))
            setattr(self, f'unembed_input_{embed_name}', GuidanceEmbedder(dim, embed_dim))

        # positional embedding 
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(dim, max_tokens) # sinoid positional encoding (no learnable params)
    
        if self.use_positional_embedding:
            self.pos_embedding = PositionalEmbedding(dim, max_tokens) # learnable positional embedding

        if self.use_human_embedding:
            self.human_embedding_h0 = nn.Parameter(torch.randn(dim))
            self.human_embedding_h1 = nn.Parameter(torch.randn(dim))

        if self.use_param_embedding:
            self.param_embedding_orient = nn.Parameter(torch.randn(dim))
            self.param_embedding_pose = nn.Parameter(torch.randn(dim))
            self.param_embedding_shape = nn.Parameter(torch.randn(dim))
            self.param_embedding_transl = nn.Parameter(torch.randn(dim))


    def mask_guidance_param(self, x, noise_chance=0.1):
        """Mask guidance parameters with noise."""
        if torch.rand(1) <= noise_chance:
            x = torch.randn_like(x)
        return x
    
    def add_human_param_embedding(self, x, param_name):
        if self.use_human_embedding:
            if 'h0' in param_name:
                x = x + self.human_embedding_h0
            if 'h1' in param_name:
                x = x + self.human_embedding_h1
        if self.use_param_embedding:
            if 'orient' in param_name:
                x = x + self.param_embedding_orient
            if 'pose' in param_name:
                x = x + self.param_embedding_pose
            if 'shape' in param_name:
                x = x + self.param_embedding_shape
            if 'transl' in param_name:
                x = x + self.param_embedding_transl
        return x

    def unembed_input(self, keys, latent):
        # get SMPL params form the transformer output
        preditions = {}
        for idx, k in enumerate(keys):
            preditions[k] = getattr(self, f'unembed_input_{k}')(latent[:, idx])
        return preditions

    def forward(self, x, timesteps, guidance={}, return_latent_vec=False):
        self.x_keys = list(x.keys())
        
        # embed input parameters
        emb_x = []
        for k, x_param in x.items():
            xemb = getattr(self, f'embed_input_{k}')(x_param).unsqueeze(1)
            xemb = self.add_human_param_embedding(xemb, k)
            emb_x.append(xemb)

        # embed time step
        if timesteps is not None:
            emb_timesteps = [self.embed_timestep(timesteps)]
        else:
            emb_timesteps = []
            
        # embed guidance parameters
        emb_guidance = []
        for k, guidance_param in guidance.items():
            if k in x.keys() and self.share_linear_layers:
                gemb = getattr(self, f'embed_input_{k}')(guidance_param).unsqueeze(1)
            else:
                gemb = getattr(self, f'embed_guidance_{k}')(guidance_param).unsqueeze(1)
            gemb = self.add_human_param_embedding(gemb, k)
            emb_guidance.append(gemb)
        
        # concat all the embeddings
        if self.use_cross_attention:
            xx = torch.cat((emb_x), dim=1)
        else:
            xx = torch.cat((emb_x + emb_timesteps + emb_guidance), dim=1)

        # add positional encoding to the input (not learnable / sinoid)
        if self.use_positional_encoding:
            xx = self.pos_encoding(xx)

        # add positional embedding to the input (learnable)
        if self.use_positional_embedding:
            xx = self.pos_embedding(xx)

        # forward transformer
        if self.use_cross_attention:
            cc = torch.cat((emb_timesteps + emb_guidance), dim=1)
            if self.encode_target: # doesn't make sense, but helpful to understand cross-attention
                xx = self.transformer_encoder(xx)
            else:
                cc = self.transformer_encoder(cc)
            
            xx = self.transformer_decoder(
                tgt = xx, memory = cc
            )
        else:
            xx = self.transformer_encoder(xx)

        # get SMPL params form the transformer output
        #predictions = {}
        #for idx, (k, target_param) in enumerate(x.items()):
        #    predictions[k] = getattr(self, f'unembed_input_{k}')(xx[:, idx])
        predictions = self.unembed_input(self.x_keys, xx)

        if return_latent_vec:
            return predictions, xx
        else:
            return predictions


def build_diffusion_transformer(transformer_config):
    model = BUDDI(**transformer_config)
    return model