# Assembly Bias, Splashback Radius and AGN Feedback
## A Toy Cosmological Simulation Framework

This repository contains a reduced-parameter simulation designed to explore
possible relationships between **assembly bias**, **splashback radius**,
and **AGN feedback** in dark matter halos.

The framework combines several physically motivated relations drawn from
the astrophysical literature and applies them to a synthetic halo population.
Its purpose is exploratory rather than predictive.

## What the Model Does

The simulation generates a synthetic population of dark matter halos and computes:

1. **Halo Mass Function** — simplified power-law distribution,
   `dN/dM ∝ M^-(1+α)`
2. **Accretion Rate Γ** (`Γ = d ln M / d ln a`) — a proxy for halo formation
   history, sampled from a log-normal distribution around a mass-dependent median
3. **Clustering Bias `b(M, Γ)`** — a mass-dependent bias term combined with
   an assembly-bias contribution related to formation history
4. **Splashback Radius `R_sp/R_200m`** — based on an approximate functional form
   inspired by Diemer & Kravtsov (2014)

The model then investigates the resulting statistical correlations.

In addition, it includes an **AGN feedback module**:

5. **Black Hole Mass `M_BH`**
6. **Eddington Ratio `λ_Edd = Ṁ/Ṁ_Edd`**
7. **Feedback Mode** (radiative/quasar versus kinetic/radio)

## Important Methodological Note

This project is a **toy model**, not a full cosmological simulation.

Each component is included because it has a clear physical interpretation
and, where possible, is motivated by previously published work.

The correlation between assembly bias and splashback radius is partly
built into the model because both quantities depend on the same underlying
accretion parameter Γ.

Consequently, the appearance of a correlation should not be interpreted
as a discovery. Instead, it should be viewed as an internal consistency
test and a framework for generating hypotheses.

## AGN Extension

⚠️ **Important**

The direct connection between AGN feedback and splashback radius implemented
in this code is an exploratory extension.

The underlying ingredients:

- black-hole mass scaling relations,
- Eddington-ratio distributions,
- radiative versus kinetic feedback modes,

are motivated by the literature.

However, the specific correction applied to `R_sp/R_200m` is not intended
to represent a published or validated physical law. It is included solely
to explore how such a mechanism might qualitatively influence halo boundaries.

## Natural Next Step

Replace the synthetic halo population with real halo catalogs from:

- Halotools / Bolshoi-Planck
- IllustrisTNG
- MultiDark

and repeat the analysis using simulation-derived data.

## Usage

```bash
pip install -r requirements.txt
python halo_simulation.py
python halo_simulation.py --n-halos 10000 --seed 1 --output out.png
```

## Disclaimer

This repository presents an exploratory computational framework.

The results should not be interpreted as evidence for new cosmological
physics without validation against observational data, cosmological
surveys, and large-scale numerical simulations.

The purpose of this work is hypothesis generation, methodological
exploration, and educational investigation of possible relationships
between assembly bias, splashback radius, and AGN feedback.

## References

- Diemer & Kravtsov (2014)
- Gao, Springel & White (2005)
- Gao & White (2007)
- Wechsler et al. (2006)
- McBride, Fakhouri & Ma (2009)
- Weinberger et al. (2017)
- Kormendy & Ho (2013)
