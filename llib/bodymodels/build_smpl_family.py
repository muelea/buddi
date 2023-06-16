def param_to_smpl_args(param_name, param_cfg):
    """
    Convert the config settings to SMPL input format.
    Parameters
    ----------
    param_name: str
        name of the parameter (see options in defaults)
    param_cfg: cfg
        config file of the parameter (see options in defaults)
    """
    
    model_cfg = {
        f'{param_name}': param_cfg.get('value', None),
        f'create_{param_name}': param_cfg.get('create', True),
    }

    if param_name == 'betas':
        model_cfg['num_betas'] = param_cfg.dim
    elif param_name in ['left_hand_pose', 'right_hand_pose']:
        model_cfg['use_pca'] = param_cfg.use_pca
        model_cfg['num_pca_comps'] = param_cfg.num_pca_comps
        model_cfg['flat_hand_mean'] = param_cfg.flat_hand_mean
    elif param_name == 'expression':
        model_cfg['num_expression_coeffs'] = param_cfg.dim
    
    return model_cfg

def smpl_cfg_to_args(cfg, batch_size=1):
    """
    Convert the config settings to SMPL input format.
    Parameters
    ----------
    cfg: 
        config file of body_model
    """

    init_args = dict(eval(f'cfg.{cfg.type}.init'))
    # update with input params
    init_args['model_path'] = cfg.smpl_family_folder
    init_args['model_type'] = cfg.type
    init_args['batch_size'] = batch_size

    smpl_param_names = \
        ['betas', 'body_pose', 'global_orient', 'transl']
    smplh_param_names = smpl_param_names + \
        ['left_hand_pose', 'right_hand_pose']
    smplx_param_names = smplh_param_names + \
        ['expression', 'jaw_pose', 'leye_pose', 'reye_pose']
    param_names = {
        'smpl': smpl_param_names,
        'smplh': smplh_param_names,
        'smplx': smplx_param_names
    }
    for param_name in param_names[cfg.type]:
        param_cfg = eval(f'cfg.{cfg.type}.init.{param_name}')
        new_args = param_to_smpl_args(param_name, param_cfg)
        init_args.update(new_args)

    return init_args


    
