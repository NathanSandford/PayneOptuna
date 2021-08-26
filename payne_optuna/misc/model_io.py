from pathlib import Path
import yaml
import numpy as np
from payne_optuna.model import LightningPaynePerceptron

def load_model(config_file, verbose=True):
    # Load Configs & Set Paths
    with open(config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = configs["name"]
    if verbose:
        print(f'Loading Model {model_name}')
    output_dir = Path(configs["paths"]["output_dir"])
    model_dir = output_dir.joinpath(model_name)
    meta_file = model_dir.joinpath("training_meta.yml")
    ckpt_dir = model_dir.joinpath("ckpts")
    ckpt_file = sorted(list(ckpt_dir.glob('*.ckpt')))[-1]
    # Load Meta
    with open(meta_file) as file:
        meta = yaml.load(file, Loader=yaml.UnsafeLoader)
    # Load the Payne
    nn_model = LightningPaynePerceptron.load_from_checkpoint(
        str(ckpt_file),
        input_dim=meta['input_dim'],
        output_dim=meta['output_dim']
    )
    nn_model.load_meta(meta)
    # Load Model Error from Validation
    try:
        validation_file = model_dir.joinpath('validation_results.npz')
        with np.load(validation_file) as tmp:
            try:
                nn_model.mod_errs = tmp['median_approx_err_wave_valid']
            except KeyError:
                nn_model.mod_errs = tmp['median_approx_err_wave']
    except FileNotFoundError:
        if verbose:
            print('validation_results.npz does not exist; assuming zero model error.')
        nn_model.mod_errs = np.zeros_like(nn_model.wavelength)
    return nn_model


def find_model_breaks(models, obs):
    model_bounds = np.zeros((len(models), 2))
    for i, mod in enumerate(models):
        model_coverage = mod.wavelength[[0, -1]]
        ord_wave_bounds_in_model = obs['wave'][:, [0, -1]][
            (obs['wave'][:, 0] > model_coverage[0])
            & (obs['wave'][:, -1] < model_coverage[-1])
            ]
        wave_min = model_coverage[0] if i == 0 else ord_wave_bounds_in_model[0, 0]
        wave_max = model_coverage[-1] if i == len(models) - 1 else ord_wave_bounds_in_model[-1, -1]
        model_bounds[i] = [wave_min, wave_max]
    breaks = (model_bounds.flatten()[2::2] + model_bounds.flatten()[1:-1:2]) / 2
    model_bounds[:-1, -1] = breaks
    model_bounds[1:, 0] = breaks
    model_bounds = [model_bounds[i] for i in range(len(models))]
    return model_bounds


def mask_lines(model, line_masks, mask_value=1.0):
    for line, mask in line_masks.items():
        model.mod_errs[
            (model.wavelength > mask['center'] - mask['width'])
            & (model.wavelength < mask['center'] + mask['width'])
        ] = mask_value