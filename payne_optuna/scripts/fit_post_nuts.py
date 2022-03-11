from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pymc as pm


payne_wrkdir = Path('/global/scratch/users/nathan_sandford/payne_wrkdir')
data_dir = payne_wrkdir.joinpath('data')
mcmc_summary_file = data_dir.joinpath('mcmc_summary.h5')

mcmc_df = pd.read_hdf(mcmc_summary_file, 'individual')
bounds = pd.read_hdf(mcmc_summary_file, 'bounds')
lower_bounds = bounds.loc['lower_bounds']
upper_bounds = bounds.loc['upper_bounds']
label_names = list(lower_bounds.index)

unscaled_samples_dict = np.load(data_dir.joinpath('mcmc_stacked_samples.npz'))

stack_df = pd.DataFrame(index=mcmc_df['obs_tag'].unique())

for obs_tag in tqdm(mcmc_df['obs_tag'].unique()):
    print(obs_tag)
    for i, label in enumerate(tqdm(label_names)):
        if label in ['Teff', 'logg']:
            continue
        print(i, label)

        data = unscaled_samples_dict[obs_tag][:, :, i]
        n_exp = data.shape[0]

        model = pm.Model()

        with model:
            # Priors
            mu_exp = pm.TruncatedNormal(
                "mu_exp",
                mu=np.mean(data, axis=(0, 1)),
                sigma=upper_bounds[label] - lower_bounds[label],
                lower=lower_bounds[label],
                upper=upper_bounds[label],
                shape=(n_exp,)
            )
            mu_stack = pm.TruncatedNormal(
                "mu_stack",
                mu=np.mean(data, axis=(0, 1)),
                sigma=upper_bounds[label] - lower_bounds[label],
                lower=lower_bounds[label],
                upper=upper_bounds[label]
            )
            sigma_exp = pm.HalfNormal("sigma_exp", 0.2, shape=(n_exp,))
            sigma_stack = pm.HalfNormal("sigma_stack", 0.2)
            # Likelihood
            for j in range(n_exp):
                theta_exp = pm.TruncatedNormal(
                    f"theta_exp_{j}",
                    mu=mu_exp[j],
                    sigma=sigma_exp[j],
                    lower=lower_bounds[label],
                    upper=upper_bounds[label],
                    observed=data[j, :],
                )
            theta_stack = pm.TruncatedNormal(
                f"theta_stack",
                mu=mu_stack,
                sigma=sigma_stack,
                lower=lower_bounds[label],
                upper=upper_bounds[label],
                observed=data.reshape(-1, 1),
            )
            # Sample
            trace = pm.sample(
                draws=2000,
                tune=2000,
                initvals={
                    'mu_exp': np.mean(data, axis=1),
                    'sigma_exp': np.std(data, axis=1),
                    'mu_stack': np.mean(data, axis=(0, 1)),
                    'sigma_stack': np.std(data, axis=(0, 1)),
                },
                init='adapt_diag',
                target_accept=0.9,
            )
            # Save
            mu_exp_med = trace.posterior['mu_exp'].median(axis=(0, 1))
            sigma_exp_med = trace.posterior['sigma_exp'].median(axis=(0, 1))
            mu_stack_med = trace.posterior['mu_stack'].median()
            sigma_stack_med = trace.posterior['sigma_stack'].median()

            mcmc_df.loc[mcmc_df.index[mcmc_df['obs_tag'] == obs_tag], label] = mu_exp_med
            mcmc_df.loc[mcmc_df.index[mcmc_df['obs_tag'] == obs_tag], label + '_err'] = sigma_exp_med
            stack_df.loc[obs_tag, label] = mu_stack_med
            stack_df.loc[obs_tag, label + '_err'] = sigma_stack_med

mcmc_df.to_hdf(mcmc_summary_file, 'individual')
stack_df.to_hdf(mcmc_summary_file, 'stacked')
