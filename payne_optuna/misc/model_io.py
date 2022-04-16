from pathlib import Path
import yaml
import numpy as np
import torch
from payne_optuna.model import LightningPaynePerceptron
from payne_optuna.fitting import UniformLogPrior, GaussianLogPrior, FlatLogPrior

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
        output_dim=meta['output_dim'],
        n_layers=configs['architecture']['n_layers'],
        activation=configs['architecture']['activation'],
        n_neurons=configs['architecture']['n_neurons'],
        dropout=configs['architecture']['dropout'],
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
    if (line_masks is None) or len(line_masks) == 0:
        pass
    else:
        for line, mask in line_masks.items():
            model.mod_errs[
                (model.wavelength > mask['center'] - mask['width'])
                & (model.wavelength < mask['center'] + mask['width'])
            ] = mask_value


def load_minimal_emulator(configs, model_class):
    model_res = configs['observation']['model_res']
    cont_deg = configs['fitting']['cont_deg']
    model_config_dir = Path(configs['paths']['model_config_dir'])
    model_config_files = sorted(list(model_config_dir.glob('*')))
    models = []
    for i, model_config_file in enumerate(model_config_files):
        model = load_model(model_config_file)
        models.append(model)
    models = [models[i] for i in np.argsort([model.wavelength.min() for model in models])]
    payne = model_class(
        models=models,
        cont_deg=cont_deg,
        cont_wave_norm_range=(-1, 1),
        obs_wave=None,
        obs_blaz=None,
        include_model_errs=True,
        model_res=model_res,
        vmacro_method='iso_fft',
    )
    return payne


def get_priors(payne, configs):
    stellar_label_priors = []
    for i, label in enumerate(payne.labels):
        if label in configs['fitting']['priors']:
            if (label in ['Teff', 'logg']) and configs['fitting']['use_gaia_phot']:
                stellar_label_priors.append(
                    UniformLogPrior(
                        label,
                        payne.unscale_stellar_labels(-0.55 * torch.ones(payne.n_stellar_labels))[i],
                        payne.unscale_stellar_labels(0.55 * torch.ones(payne.n_stellar_labels))[i],
                    )
                )
            elif configs['fitting']['priors'][label][0] == 'N':
                stellar_label_priors.append(
                    GaussianLogPrior(
                        label,
                        configs['fitting']['priors'][label][1],
                        configs['fitting']['priors'][label][2]
                    )
                )
            elif configs['fitting']['priors'][label][0] == 'U':
                stellar_label_priors.append(
                    UniformLogPrior(
                        label,
                        configs['fitting']['priors'][label][1],
                        configs['fitting']['priors'][label][2]
                    )
                )
            else:
                raise KeyError(f"Cannot parse prior info for {label}")
        else:
            stellar_label_priors.append(
                UniformLogPrior(
                    label,
                    payne.unscale_stellar_labels(-0.55 * torch.ones(payne.n_stellar_labels))[i],
                    payne.unscale_stellar_labels(0.55 * torch.ones(payne.n_stellar_labels))[i],
                )
            )
    priors = {
        "stellar_labels": stellar_label_priors,
        'log_vmacro': UniformLogPrior('log_vmacro', -1, 1.3),
        'log_vsini': FlatLogPrior('log_vsini'),
        'inst_res': FlatLogPrior('inst_res')
    }
    return priors