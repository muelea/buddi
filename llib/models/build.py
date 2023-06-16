from .regressors.mlp import build_mlp
from .regressors.transformer import build_transformer
from .regressors.buddi import build_buddi



def build_model(model_cfg):
    model_type = model_cfg.type

    if model_type == "mlp":
        model = build_mlp(model_cfg.mlp)
    elif model_type == "transformer":
        model = build_transformer(model_cfg.transformer)
    elif model_type == "diffusion_transformer":
        model = build_buddi(model_cfg.buddi)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model