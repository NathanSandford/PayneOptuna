import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def validation_plots(
    wavelength,
    valid_spec,
    approx_err,
    median_approx_err_star,
    median_approx_err_wave,
    twosigma_approx_err_wave,

):

    median_approx_err_wave_sorted = np.sort(median_approx_err_wave)
    twosigma_approx_err_wave_sorted = np.sort(twosigma_approx_err_wave)

    cumulative_n_pixels_median = (np.array(range(len(median_approx_err_wave_sorted))) + 1) / \
        median_approx_err_wave_sorted.shape[0]
    cumulative_n_pixels_twosigma = (np.array(range(len(twosigma_approx_err_wave_sorted))) + 1) / \
        twosigma_approx_err_wave_sorted.shape[0]

    fig = plt.figure(figsize=(20, 30))
    gs = GridSpec(3, 2)

    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(median_approx_err_star, bins=10.0 ** (np.arange(-4, 0, 0.1)), histtype='step')
    ax1.set_xlim(1e-4, 0.2)
    ax1.set_ylabel("# Cross validation models")
    ax1.set_xlabel("(Wavelength) Median approximation error")
    ax1.set_xscale('log')

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(median_approx_err_wave_sorted, cumulative_n_pixels_median, label='Median')
    ax2.plot(twosigma_approx_err_wave_sorted, cumulative_n_pixels_twosigma, label=r'$2\sigma$')
    ax2.set_xlim(1e-4, 0.2)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('(Spectrum) Median approximation error')
    ax2.set_ylabel('Cumulative # wavelength pixels [%]')
    ax2.set_xscale('log')
    ax2.legend()

    ax3 = plt.subplot(gs[1, :])
    ax3.scatter(wavelength, median_approx_err_wave, marker='.', s=1, alpha=0.5)
    ax3.set_xlabel('Wavelength [A]')
    ax3.set_ylabel('Median approximation error')
    ax3.set_yscale('log')

    ax4 = plt.subplot(gs[2, :])
    ax4.hist2d(
        approx_err.ravel(),
        valid_spec,
        bins=[10.0 ** (np.arange(-4, -1, 0.1)), np.arange(0, 1, 0.05)],
    )
    ax4.set_xscale('log')
    ax4.set_ylabel('Normalized Flux')
    ax4.set_xlabel('Approximation Error')

    return fig