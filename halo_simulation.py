"""
halo_simulation.py

Toy model simulating the relationship between Assembly Bias and
Splashback Radius in dark matter halos.

Developed with the assistance of Claude (Anthropic).

Based on published physical relations:
- Diemer & Kravtsov (2014), ApJ 789, 1
  -> empirical relation for the splashback radius as a function of the
     mass accretion rate Gamma.
- Gao, Springel & White (2005); Wechsler et al. (2006); Gao & White (2007)
  -> assembly bias: at fixed mass, halo clustering depends on formation
     history.
- McBride, Fakhouri & Ma (2009)
  -> distribution of the mass accretion rate Gamma = dlnM/dlna.
- Weinberger et al. (2017), MNRAS 465, 3291 (AGN feedback model in
  IllustrisTNG); Kormendy & Ho (2013) (approximate M_BH - M_halo relation)
  -> two-mode AGN feedback (radiative/quasar vs kinetic/radio), selected
     by the Eddington-normalized accretion rate lambda_Edd = Mdot / Mdot_Edd,
     with a threshold of ~0.01.

METHODOLOGICAL NOTE
--------------------
This is NOT an arbitrary fit to observational data. It is a reduced-parameter
model in which every term has:
  - a defined physical unit (masses in Msun/h, R in units of R_200m,
    Gamma dimensionless by definition)
  - a stated provenance (see references above)
  - a minimal number of free coefficients, all motivated by a known
    physical trend and not tuned to "match" specific numbers.

It serves as a qualitative/educational starting point. For a genuine
scientific analysis, the natural next step is to replace the synthetic
halo population with a real catalog (e.g. Bolshoi-Planck via Halotools,
or IllustrisTNG-Dark) and repeat the correlation analysis.

Usage:
    python halo_simulation.py
    python halo_simulation.py --n-halos 10000 --seed 1 --output out.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------
# Physical parameters of the model (see docstring for sources)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ModelParams:
    # Mass function: dN/dM ~ M^-(1+alpha)
    m_min: float = 1e12       # Msun/h
    m_max: float = 1e15       # Msun/h
    alpha: float = 1.0

    # Gamma_med(M) = gamma0 + gamma_slope * log10(M / 1e12)
    gamma0: float = 0.7
    gamma_slope: float = 0.15
    gamma_sigma_dex: float = 0.35  # log-normal scatter

    # Clustering bias b(M, Gamma)
    b0: float = 0.9
    b_slope: float = 0.35
    ab_amplitude: float = -0.25
    ab_mass_pivot: float = 5e12   # Msun/h

    # Splashback radius: R_sp/R_200m = A - B * tanh(C*(Gamma - D))
    sp_A: float = 1.2
    sp_B: float = 0.28
    sp_C: float = 1.1
    sp_D: float = 1.0

    # --- AGN feedback module --------------------------------------
    # M_BH - M_halo relation, approximate form (log-log linear,
    # calibrated to give M_BH ~ 1e8 Msun at M_halo ~ 1e13 Msun/h,
    # cf. qualitative trend of Kormendy & Ho 2013)
    mbh_norm: float = 1e8          # Msun, normalization at M_halo = 1e13
    mbh_slope: float = 1.15        # log-log slope
    mbh_scatter_dex: float = 0.3   # intrinsic scatter (dex)

    # Distribution of lambda_Edd = Mdot/Mdot_Edd (log-normal)
    lambda_edd_median: float = 0.02
    lambda_edd_sigma_dex: float = 0.8
    lambda_edd_threshold: float = 0.01  # radiative/kinetic threshold (Weinberger+17)

    # Amplitude of the feedback effect on the R_sp/R_200m ratio
    # (positive: kinetic mode slightly "inflates" the boundary due to
    # gas expulsion; negative: radiative mode compresses it)
    feedback_kinetic_boost: float = 0.05
    feedback_radiative_damp: float = -0.02


# ---------------------------------------------------------------------
# Physical components of the model
# ---------------------------------------------------------------------
def sample_halo_masses(n: int, p: ModelParams, rng: np.random.Generator) -> np.ndarray:
    """Sample masses from a power-law mass function dN/dM ~ M^-(1+alpha)
    via CDF inversion."""
    u = rng.uniform(0, 1, n)
    a = p.alpha
    return (p.m_min ** (-a) - u * (p.m_min ** (-a) - p.m_max ** (-a))) ** (-1 / a)


def gamma_median(mass: np.ndarray, p: ModelParams) -> np.ndarray:
    """Median value of the accretion rate Gamma = dlnM/dlna as a function
    of mass (approximate trend from McBride et al. 2009)."""
    return p.gamma0 + p.gamma_slope * np.log10(mass / 1e12)


def sample_accretion_rate(mass: np.ndarray, p: ModelParams,
                           rng: np.random.Generator) -> np.ndarray:
    """Sample Gamma from a log-normal distribution around the median value."""
    g_med = gamma_median(mass, p)
    gamma = rng.lognormal(mean=np.log(g_med), sigma=p.gamma_sigma_dex)
    return np.clip(gamma, 0.05, 5.0)


def clustering_bias(mass: np.ndarray, gamma: np.ndarray, p: ModelParams) -> np.ndarray:
    """Clustering bias b(M, Gamma): mass-dependent base term plus an
    assembly-bias term dependent on formation history at fixed mass
    (exponentially damped at high masses)."""
    b_base = p.b0 + p.b_slope * np.log10(mass / 1e12)
    g_med = gamma_median(mass, p)
    ab_strength = p.ab_amplitude * np.exp(-mass / (3 * p.ab_mass_pivot))
    delta_b = ab_strength * (np.log(g_med) - np.log(gamma))
    return b_base + delta_b


def splashback_ratio(gamma: np.ndarray, p: ModelParams) -> np.ndarray:
    """R_sp / R_200m following the approximate functional form of
    Diemer & Kravtsov (2014), calibrated on their Fig. 6."""
    return p.sp_A - p.sp_B * np.tanh(p.sp_C * (gamma - p.sp_D))


def sample_black_hole_mass(mass_halo: np.ndarray, p: ModelParams,
                            rng: np.random.Generator) -> np.ndarray:
    """M_BH as a function of halo mass, approximate log-log relation
    (qualitative trend of Kormendy & Ho 2013) with intrinsic log-normal
    scatter."""
    log_mbh_mean = (np.log10(p.mbh_norm)
                    + p.mbh_slope * np.log10(mass_halo / 1e13))
    log_mbh = rng.normal(loc=log_mbh_mean, scale=p.mbh_scatter_dex)
    return 10 ** log_mbh  # Msun


def sample_eddington_ratio(n: int, p: ModelParams,
                           rng: np.random.Generator) -> np.ndarray:
    """lambda_Edd = Mdot / Mdot_Edd, log-normally distributed (typical
    order of magnitude for AGN populations, e.g. Aird et al. 2018)."""
    lam = rng.lognormal(mean=np.log(p.lambda_edd_median), sigma=p.lambda_edd_sigma_dex, size=n)
    return np.clip(lam, 1e-5, 1.0)


def agn_feedback_mode(lambda_edd: np.ndarray, p: ModelParams) -> np.ndarray:
    """Assign the AGN feedback mode based on the Eddington-ratio threshold
    (Weinberger et al. 2017): 'radiative' above the threshold (efficient
    accretion, thermal/quasar feedback), 'kinetic' below the threshold
    (inefficient accretion, mechanical/radio feedback)."""
    return np.where(lambda_edd >= p.lambda_edd_threshold, "radiative", "kinetic")


def agn_feedback_correction(mbh: np.ndarray, lambda_edd: np.ndarray,
                             mode: np.ndarray, p: ModelParams) -> np.ndarray:
    """Multiplicative correction to R_sp/R_200m due to AGN feedback. The
    amplitude scales with M_BH (feedback power) and the sign depends on
    the mode:
      - kinetic mode: jets expel gas, slightly "inflating" the effective
        halo boundary (positive correction)
      - radiative mode: more concentrated thermal feedback, mild
        compression effect (negative correction)
    This term is EXPLICITLY illustrative/qualitative: the real physics
    linking feedback to the splashback radius is not established in the
    literature with this simplicity, and is our own extension rather
    than a published relation like the other three.
    """
    mbh_scale = np.log10(mbh / 1e8)  # dimensionless, grows with M_BH
    sign = np.where(mode == "kinetic", p.feedback_kinetic_boost,
                    p.feedback_radiative_damp)
    return 1.0 + sign * np.clip(mbh_scale, -2, 2) / 2


# ---------------------------------------------------------------------
# End-to-end simulation
# ---------------------------------------------------------------------
def run_simulation(n_halos: int, seed: int, p: ModelParams | None = None):
    p = p or ModelParams()
    rng = np.random.default_rng(seed)

    mass = sample_halo_masses(n_halos, p, rng)
    gamma = sample_accretion_rate(mass, p, rng)
    bias = clustering_bias(mass, gamma, p)
    r_ratio_dm = splashback_ratio(gamma, p)  # dark matter only, as before

    # AGN feedback module
    mbh = sample_black_hole_mass(mass, p, rng)
    lambda_edd = sample_eddington_ratio(n_halos, p, rng)
    mode = agn_feedback_mode(lambda_edd, p)
    agn_correction = agn_feedback_correction(mbh, lambda_edd, mode, p)
    r_ratio = r_ratio_dm * agn_correction

    return {
        "mass": mass, "gamma": gamma, "bias": bias,
        "r_ratio_dm": r_ratio_dm, "r_ratio": r_ratio,
        "mbh": mbh, "lambda_edd": lambda_edd, "mode": mode,
        "agn_correction": agn_correction,
    }


def print_summary(results: dict) -> None:
    mass, gamma, bias, r_ratio, r_ratio_dm, mbh, lambda_edd, mode = (
        results[k] for k in
        ("mass", "gamma", "bias", "r_ratio", "r_ratio_dm", "mbh", "lambda_edd", "mode")
    )
    r_pearson, p_pearson = stats.pearsonr(gamma, r_ratio_dm)
    r_spearman, p_spearman = stats.spearmanr(bias, r_ratio_dm)
    frac_kinetic = np.mean(mode == "kinetic")

    print("=== Simulation results ===")
    print(f"Number of simulated halos: {len(mass)}")
    print(f"Mass range: {mass.min():.2e} - {mass.max():.2e} Msun/h")
    print(f"M_BH range:  {mbh.min():.2e} - {mbh.max():.2e} Msun")
    print(f"Fraction in kinetic mode (lambda_Edd < threshold): {frac_kinetic:.1%}")
    print()
    print("--- Assembly bias / splashback correlations (dark matter only) ---")
    print(f"Correlation Gamma vs R_sp/R_200m (Pearson):  r   = {r_pearson:.3f} (p={p_pearson:.1e})")
    print(f"Correlation bias  vs R_sp/R_200m (Spearman): rho = {r_spearman:.3f} (p={p_spearman:.1e})")
    print()
    print(f"Mean effect of the AGN correction on R_sp/R_200m: "
          f"{np.mean(r_ratio / r_ratio_dm - 1):+.2%} (scatter: "
          f"{np.std(r_ratio / r_ratio_dm - 1):.2%})")
    print()
    print("WARNING: the DM-only correlation here is built in by construction,")
    print("since both bias and R_sp depend on the same Gamma. It is not a")
    print("discovery, it is an internal consistency check of the model. The")
    print("added AGN feedback term is EXPLICITLY illustrative/qualitative: the")
    print("M_BH-M_halo and lambda_Edd relations used have a basis in the")
    print("literature, but the direct feedback->splashback radius link with")
    print("this simple functional form is not a published relation like the")
    print("other three. It is meant to show HOW physically motivated variety")
    print("could be introduced, not to quantify it precisely.")


def make_plots(results: dict, p: ModelParams, output_path: str) -> None:
    mass, gamma, bias, r_ratio, r_ratio_dm, mbh, lambda_edd, mode = (
        results[k] for k in
        ("mass", "gamma", "bias", "r_ratio", "r_ratio_dm", "mbh", "lambda_edd", "mode")
    )
    is_kinetic = mode == "kinetic"

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    axes[0, 0].hist(np.log10(mass), bins=40, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_xlabel(r"$\log_{10}(M / M_\odot h^{-1})$")
    axes[0, 0].set_ylabel("N halos")
    axes[0, 0].set_title("Synthetic halo population")

    sc = axes[0, 1].scatter(gamma, bias, c=np.log10(mass), cmap="viridis", s=8, alpha=0.6)
    axes[0, 1].set_xlabel(r"$\Gamma$ (accretion rate)")
    axes[0, 1].set_ylabel("clustering bias b(M, Γ)")
    axes[0, 1].set_title("Assembly bias at fixed mass")
    plt.colorbar(sc, ax=axes[0, 1], label=r"$\log_{10} M$")

    axes[0, 2].scatter(np.log10(mass), np.log10(mbh), s=8, alpha=0.4, color="#55A868")
    axes[0, 2].set_xlabel(r"$\log_{10}(M_{\rm halo})$")
    axes[0, 2].set_ylabel(r"$\log_{10}(M_{\rm BH})$")
    axes[0, 2].set_title("M_BH - M_halo relation (with scatter)")

    axes[1, 0].scatter(gamma, r_ratio_dm, s=8, alpha=0.4, color="#DD8452")
    gamma_grid = np.linspace(0.05, 5, 200)
    axes[1, 0].plot(gamma_grid, splashback_ratio(gamma_grid, p), color="black", lw=2,
                     label="Diemer & Kravtsov (2014)")
    axes[1, 0].set_xlabel(r"$\Gamma$ (accretion rate)")
    axes[1, 0].set_ylabel(r"$R_{sp}/R_{200m}$ (DM only)")
    axes[1, 0].set_title("Splashback radius, dark matter only")
    axes[1, 0].legend()

    axes[1, 1].scatter(np.log10(mbh)[is_kinetic], (r_ratio / r_ratio_dm)[is_kinetic],
                        s=8, alpha=0.5, color="#C44E52", label="kinetic mode")
    axes[1, 1].scatter(np.log10(mbh)[~is_kinetic], (r_ratio / r_ratio_dm)[~is_kinetic],
                        s=8, alpha=0.5, color="#4C72B0", label="radiative mode")
    axes[1, 1].axhline(1.0, color="black", lw=1, ls="--")
    axes[1, 1].set_xlabel(r"$\log_{10}(M_{\rm BH})$")
    axes[1, 1].set_ylabel("AGN correction to R_sp/R_200m")
    axes[1, 1].set_title("Effect of AGN feedback (illustrative)")
    axes[1, 1].legend(fontsize=8)

    sc2 = axes[1, 2].scatter(r_ratio, bias, c=np.log10(mbh), cmap="magma", s=8, alpha=0.6)
    axes[1, 2].set_xlabel(r"$R_{sp}/R_{200m}$ (with AGN correction)")
    axes[1, 2].set_ylabel("clustering bias")
    axes[1, 2].set_title("Assembly bias vs total splashback\n(color: log M_BH)")
    plt.colorbar(sc2, ax=axes[1, 2], label=r"$\log_{10} M_{\rm BH}$")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Toy model: assembly bias vs splashback radius in dark matter "
                     "halos, with an AGN feedback module (M_BH + lambda_Edd).")
    )
    parser.add_argument("--n-halos", type=int, default=5000,
                         help="Number of synthetic halos to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random generator seed (default: 42)")
    parser.add_argument("--output", type=str, default="assembly_bias_splashback.png",
                         help="Output PNG file path (default: assembly_bias_splashback.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ModelParams()
    results = run_simulation(args.n_halos, args.seed, params)
    print_summary(results)
    make_plots(results, params, args.output)


if __name__ == "__main__":
    main()
