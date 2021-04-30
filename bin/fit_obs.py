#!/usr/bin/env python

"""
Train the Payne
"""
from payne_optuna.scripts import fit_obs


if __name__ == '__main__':
    fit_obs.main(fit_obs.parse_args())