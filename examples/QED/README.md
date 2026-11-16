# Photon emission (strong-field QED) examples

Tests of the photon emission (nonlinear Compton scattering with spin and photon polarization,
`<beam>.qed = 1`): single particles in constant external fields (`fields.external_B`) without plasma,
betatron oscillations in an ion channel, and a PWFA with a 10 TeV witness. Theory: the rates in the
spin- and polarization-resolved form of Y.-F. Li et al., PRL 122, 154801 (2019) and PRL 124, 014801 (2020)
(main README §1 and `src/QEDRates.hpp`).

Units of the single-particle decks: `units.n0_cm3 = 1.1e21`, the critical density of a 1 µm laser, so
ω_p = ω₀ and lengths are in c/ω₀ of that laser; η = ħω₀/mc² = 2.41·10⁻⁶ and χ = η γ B⊥ (code units). The
fixed-χ decks keep the momentum fixed (`e.momentum_push = 0`, `qed.no_recoil = 1`). Betatron and PWFA
decks: `units.n0_cm3 = 2e18` (k_p⁻¹ = 3.76 µm), η = 1.03·10⁻⁷.

| deck | what | result (theory) | run time (2 cores) |
|---|---|---|---|
| `larmor.in` | gyration in B = 10 e_z, p⊥ = 40, χ₀ = 0.00096 | γ 10⁵ → 99 814 (synchrotron loss with g(χ): 99 814) | 6 s |
| `larmor_spiral.in` | p⊥ = 200, B = 50, χ₀ = 0.024 | γ → 4.8·10⁴ (1 % below the deterministic loss: energy spread), the orbit spirals inwards (radius 4 → 1.9) | 33 s |
| `spectrum.in` | γ = 10⁷ in B_x, momentum fixed; χ = 0.1, 1, 10 via `fields.external_B` | dN/du = C₀ w(u, χ) | 1 s |
| `spin_dependence.in` | initial-spin dependence of the emission: beams with s₂ = +1, −1, 0 (spin fixed) in one run, χ = 1 (χ = 0.1 by override) | photon numbers within 1.4σ of N_eTC₀(W + Gs₂), ratio +1/−1 0.9155 ± 0.0037 (0.9128; χ = 0.1: 0.9630 ± 0.0038 vs 0.9623), dN₋/dE − dN₊/dE = −2C₀g | 1.4 s (χ = 0.1: 11 s) |
| `sokolov_ternov.in` | spin relaxation at fixed χ = 1 (χ = 0.1 by override) | ⟨s₂⟩ 1 → −0.836 (mean spin equation: −0.837); χ = 0.1: −0.805 at t = 2·10⁷ (−0.805, limit −0.905) | 5 s |
| `photon_polarization_spectrum.in` | photon polarization spectrum ξ₃(E_γ) at γ = 10⁷, χ = 0.1 for the initial spins s₂ = 0, +1, −1 (one run, six beams: all photons + the hard ones of 25 × more electrons via `photon_min_energy`) | ξ₃ = p_ξ/(w + g s₂) in all 24 bins up to 6·10⁶ mc² (largest deviation 0.013 = the u-spread within the steepest bin); at 6·10⁶: 0.53, −0.30, 0.97 | 18 s |
| `photon_polarization_mean.in` | mean Stokes vector of all photons vs the electron energy (χ = γ/10⁶) and the spin | ∫p_ξ du / ∫(w + g s₂) du | 0.4 s per point |
| `betatron.in` | γ₀ = 10⁵ from x₀ = 1 in an ion channel, χ₀ = 0.005, 30 betatron periods | ⟨γ⟩ = 67 222 vs the orbit-averaged loss with A ∝ γ^{1/8}: 67 148 (+0.11 %; constant A: +1.9 %); amplitude ⟨A⟩ = 0.9514 vs (γ/γ₀)^{1/8} = 0.9516 | 30 s |
| `pwfa_benchmark.in` | 500 pC driver + 35 pC 10 TeV witness at 2·10¹⁸ cm⁻³, t = 2700; three runs: no radiation / synchrotron (`qed.model = continuous`) / QED | see below | 1 min each |

Betatron oscillation with radiation losses: averaged over an oscillation of amplitude A the radiated power is
that of an electron at r_eff = A/√2. The emission happens mostly near the turning points (power ∝ x²), where
it takes energy but little transverse momentum: averaging the change of A² = x² + 2p_x²/γ with the weight
sin²φ of the phase gives d ln A = (1/8) d ln γ, i.e. A ∝ γ^{1/8}.

PWFA with a 10 TeV witness (`analyze.py benchmark` draws the energy evolution, the spectra and the wake):

| | QUARZ |
|---|---|
| witness energy gain without radiation at t = 2700 | 1.34 GeV |
| mean witness energy at t = 2700: synchrotron / QED | 7986 / 7970 GeV |
| witness energies (QED) | 1.06 … 10.1 TeV, spike at 10 TeV (particles near the axis, χ ≈ 0) |
| witness energies (synchrotron) | 4.9 … 10.1 TeV |
| highest photon energy | 8.5 TeV |
| χ of the emitters | mean 0.31, max 0.76 |
| bubble at t = 1400 | radius ≈ 6, closure at ξ ≈ 13, witness near the rear |

The mean energies of the stochastic and the deterministic (quantum-corrected) radiation agree; the stochastic
emission spreads the witness energies over an order of magnitude.

Run from this directory, then compare with the theory (scipy; writes a plot into the output directory):

```bash
../../build/quarz larmor.in
python3 analyze.py larmor larmor.in out_larmor
../../build/quarz spectrum.in "fields.external_B=0.41492 0 0" output.dir=out_spectrum_chi10
python3 analyze.py spectrum spectrum.in out_spectrum_chi10 "fields.external_B=0.41492 0 0"
../../build/quarz spin_dependence.in
python3 analyze.py spindiff spin_dependence.in out_spin_dependence
../../build/quarz photon_polarization_spectrum.in
python3 analyze.py polspec photon_polarization_spectrum.in out_polspec
../../build/quarz sokolov_ternov.in
python3 analyze.py spin sokolov_ternov.in out_sokolov_ternov
../../build/quarz photon_polarization_mean.in "e.spin=1 0 0" output.dir=out_pol_sp
python3 analyze.py polarization photon_polarization_mean.in out_pol_sp "e.spin=1 0 0"
../../build/quarz betatron.in
python3 analyze.py betatron betatron.in out_betatron
../../build/quarz pwfa_benchmark.in witness.qed=0 output.dir=out_pwfa_base
../../build/quarz pwfa_benchmark.in qed.model=continuous output.dir=out_pwfa_sync
../../build/quarz pwfa_benchmark.in
python3 analyze.py benchmark pwfa_benchmark.in out_pwfa_base out_pwfa_sync out_pwfa_qed
```

Options for such tests (all off by default; main README §5): `<beam>.photon_min_energy` (write only the
photons above an energy in m c²; all are emitted and counted in `qed.txt`), `qed.no_recoil` (emission without momentum
change), `qed.fixed_spin`, `<beam>.momentum_push = 0` (the fields do not change the momentum),
`<beam>.spin_precession = 0` (no Thomas–BMT rotation), `qed.model = continuous | classical`
(deterministic radiation reaction: synchrotron loss with or without g(χ), no photons).

(`analyze.py` takes the same `key=value` overrides as the run.) The samplers and spin models are chosen
with `qed.sampler = inversion | rejection` and `qed.spin_model = mixed | pure` (see the main README, §1
and §5). Outputs: `photons_e.bin` (one record per photon), `qed.txt`, `beams.txt`, `spin.txt`.
The validated versions of these tests are section 27 of `validation/run_validation.sh`.
