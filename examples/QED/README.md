# Photon emission (strong-field QED) examples

Tests of the photon emission (nonlinear Compton scattering with spin and photon polarization,
`<beam>.qed = 1`), after the simulations of the QED module paper (O. Mathiak, A. Pukhov et al.,
sections V and VI): single particles in constant external fields (`fields.external_B`) without plasma,
betatron oscillations in an ion channel, and the PWFA benchmark with a 10 TeV witness.

Units of the section V decks: `units.n0_cm3 = 1.1e21`, the critical density of a 1 µm laser, so ω_p = ω₀
and lengths are in c/ω₀ as in the paper; η = ħω₀/mc² = 2.41·10⁻⁶ and χ = η γ B⊥ (code units). The fixed-χ
decks keep the momentum fixed as in the paper (`e.momentum_push = 0`, `qed.no_recoil = 1`). Section VI:
`units.n0_cm3 = 2e18` (k_p⁻¹ = 3.76 µm), η = 1.03·10⁻⁷; the paper's ω₀t is ω_p t there.

| deck | paper | what | run time (2 cores) |
|---|---|---|---|
| `larmor.in` | Fig. 1 left | gyration in B = 10 e_z, p⊥ = 40, χ₀ = 0.00096: γ 10⁵ → 99 814 (synchrotron loss with g(χ): 99 814) | 6 s |
| `larmor_spiral.in` | Fig. 1 right | p⊥ = 200, B = 50, χ₀ = 0.024: γ → 4.8·10⁴, the orbit spirals inwards (radius 4 → 1.9) | 33 s |
| `spectrum.in` | Fig. 2 | γ = 10⁷ in B_x, momentum fixed (`qed.no_recoil`): dN/du = C₀ w(u, χ); χ = 0.1, 1, 10 via `fields.external_B` | 1 s |
| `spin_dependence.in` | Fig. 3 | as `spectrum.in` with the spin fixed along ±B: dN₊/du − dN₋/du = 2C₀g | 2 × 1 s |
| `sokolov_ternov.in` | Fig. 4 | spin relaxation at fixed χ = 1: ⟨s₂⟩ 1 → −0.836 (eq. 14: −0.837); χ = 0.1: −0.805 at t = 2·10⁷ (eq. 14: −0.805, limit −0.905) | 5 s |
| `photon_polarization.in` | Fig. 5 | mean Stokes vector of the photons vs γ (χ = γ/10⁶) and spin, eq. 22 | 0.4 s per point |
| `betatron.in` | §VI.A | γ₀ = 10⁵ from x₀ = 1 in an ion channel, χ₀ = 0.005, 30 betatron periods: ⟨γ⟩ = 67 222 vs the orbit-averaged loss (eq. 26) with A ∝ γ^{1/8}: 67 148 (+0.11 %; constant A: +1.9 %); amplitude ⟨A⟩ = 0.9514 vs (γ/γ₀)^{1/8} = 0.9516 (eq. 27) | 30 s |
| `pwfa_benchmark.in` | §VI, Figs. 6–8 | 500 pC driver + 35 pC 10 TeV witness at 2·10¹⁸ cm⁻³, t = 2700; three runs: base / synchrotron (`qed.model = continuous`) / QED | 1 min each |

PWFA benchmark, QUARZ vs the paper:

| | QUARZ | paper |
|---|---|---|
| witness energy gain without radiation at t = 2700 | 1.34 GeV | ≈ 1.2–1.3 (Fig. 8 left; the "MeV" axis labels of Fig. 8 must be GeV) |
| mean witness energy at t = 2700: synchrotron / QED | 7986 / 7970 GeV | ≈ 7950 / 7900 GeV (Fig. 8 right) |
| witness energies (QED) | 1.06 … 10.1 TeV, spike at 10 TeV | ≈ 1 … 10 TeV (Fig. 7 right) |
| witness energies (synchrotron) | 4.9 … 10.1 TeV | ≈ 3.5 … 10 TeV |
| highest photon energy | 8.5 TeV | ≈ 8 TeV (Fig. 7 left) |
| χ of the emitters | mean 0.31, max 0.76 | "χ_max ~ 2" |
| bubble at t = 1400 | radius ≈ 6, closure at ξ ≈ 13, witness near the rear | the same shape (Fig. 6) |

(The larger χ_max of the paper probably comes from its 3D witness, which can have particles further off axis;
the energy loss is dominated by χ ~ 0.3.)

Run from this directory, then compare with the theory (scipy; writes a plot into the output directory):

```bash
../../build/quarz larmor.in
python3 analyze.py larmor larmor.in out_larmor
../../build/quarz spectrum.in "fields.external_B=0.41492 0 0" output.dir=out_spectrum_chi10
python3 analyze.py spectrum spectrum.in out_spectrum_chi10 "fields.external_B=0.41492 0 0"
../../build/quarz spin_dependence.in "e.spin=1 0 0" output.dir=out_spin_plus
../../build/quarz spin_dependence.in "e.spin=-1 0 0" output.dir=out_spin_minus
python3 analyze.py spindiff spin_dependence.in out_spin_plus out_spin_minus
../../build/quarz sokolov_ternov.in
python3 analyze.py spin sokolov_ternov.in out_sokolov_ternov
../../build/quarz photon_polarization.in "e.spin=1 0 0" output.dir=out_pol_sp
python3 analyze.py polarization photon_polarization.in out_pol_sp "e.spin=1 0 0"
../../build/quarz betatron.in
python3 analyze.py betatron betatron.in out_betatron
../../build/quarz pwfa_benchmark.in witness.qed=0 output.dir=out_pwfa_base
../../build/quarz pwfa_benchmark.in qed.model=continuous output.dir=out_pwfa_sync
../../build/quarz pwfa_benchmark.in
python3 analyze.py benchmark pwfa_benchmark.in out_pwfa_base out_pwfa_sync out_pwfa_qed
```

Options for such tests (all off by default; main README §5): `qed.no_recoil` (emission without momentum
change), `qed.fixed_spin`, `<beam>.momentum_push = 0` (the fields do not change the momentum),
`<beam>.spin_precession = 0` (no Thomas–BMT rotation), `qed.model = continuous | classical`
(deterministic radiation reaction: synchrotron loss with or without g(χ), no photons).

(`analyze.py` takes the same `key=value` overrides as the run.) The samplers and spin models are chosen
with `qed.sampler = inversion | rejection` and `qed.spin_model = mixed | pure` (see the main README, §1
and §5). Outputs: `photons_e.bin` (one record per photon), `qed.txt`, `beams.txt`, `spin.txt`.
The validated versions of these tests are section 26 of `validation/run_validation.sh`.
