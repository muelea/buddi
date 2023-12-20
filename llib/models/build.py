from .regressors.mlp import build_mlp
from .regressors.transformer import build_transformer
from .regressors.buddi import build_diffusion_transformer as build_buddi
#from .regressors.vae_mlp import build_vae_mlp
#from .regressors.autoencoder_mlp import build_ae_mlp



def build_model(model_cfg):
    model_type = model_cfg.type

    if model_type == "mlp":
        model = build_mlp(model_cfg.mlp)
    elif model_type == "transformer":
        model = build_transformer(model_cfg.transformer)
    elif model_type == "diffusion_transformer":
        model = build_buddi(model_cfg.diffusion_transformer)
    # elif model_type == "vae_mlp":
    #     model = build_vae_mlp(model_cfg.vae_mlp)
    # elif model_type == "ae_mlp":
    #     model = build_ae_mlp(model_cfg.ae_mlp)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model