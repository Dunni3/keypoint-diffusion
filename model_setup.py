from models.ligand_diffuser import KeypointDiffusion
from pathlib import Path

def model_from_config(config: dict):
    # create test dataset object
    dataset_path = Path(config['dataset']['location']) 
    # test_dataset_path = str(dataset_path / f'{args.split}.pkl')
    # test_dataset = CrossDockedDataset(name=args.split, processed_data_file=test_dataset_path, **config['graph'], **config['dataset'])

    # get the model architecture
    try:
        architecture = config['diffusion']['architecture']
    except KeyError:
        architecture = 'egnn'

    try:
        rec_encoder_type = config['diffusion']['rec_encoder_type']
    except KeyError:
        rec_encoder_type = 'learned'

    # determine if we're using fake atoms
    try:
        use_fake_atoms = config['dataset']['max_fake_atom_frac'] > 0
    except KeyError:
        use_fake_atoms = False

    # get the number of receptor atom features, ligand atom features, and keypoint features
    n_rec_feat = len(config['dataset']['rec_elements'])

    n_lig_feat = len(config['dataset']['lig_elements'])
    if use_fake_atoms:
        n_lig_feat += 1

    if rec_encoder_type == 'learned':
        if architecture == 'egnn':
            n_kp_feat = config["rec_encoder"]["out_n_node_feat"]
        elif architecture == 'gvp':
            n_kp_feat = config["rec_encoder_gvp"]["out_scalar_size"]
    else:
        n_kp_feat = n_rec_feat


    # get rec encoder config and dynamics config
    if architecture == 'gvp':
        rec_encoder_config = config["rec_encoder_gvp"]
        rec_encoder_config['in_scalar_size'] = n_rec_feat
        dynamics_config = config['dynamics_gvp']
    elif architecture == 'egnn':
        rec_encoder_config = config["rec_encoder"]
        rec_encoder_config["in_n_node_feat"] = n_rec_feat
        dynamics_config = config['dynamics']

    # create diffusion model
    model = KeypointDiffusion(
        n_lig_feat, 
        n_kp_feat,
        processed_dataset_dir=Path(config['dataset']['location']),
        graph_config=config['graph'],
        dynamics_config=dynamics_config, 
        rec_encoder_config=rec_encoder_config, 
        rec_encoder_loss_config=config['rec_encoder_loss'],
        use_fake_atoms=use_fake_atoms,
        **config['diffusion'])
    return model