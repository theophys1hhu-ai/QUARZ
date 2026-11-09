# Photon emission (strong-field QED) examples

Single-particle tests of the photon emission (nonlinear Compton scattering with spin and photon
polarization, `<beam>.qed = 1`), after the simulations of the QED module paper (O. Mathiak, A. Pukhov
et al., section V). The electrons move in constant external fields (`fields.external_B`) without plasma.

Units: `units.n0_cm3 = 1.1e21`, the critical density of a 1 µm laser, so ω_p = ω₀ and lengths are in
c/ω₀ as in the paper; η = ħω₀/mc² = 2.41·10⁻⁶ and χ = η γ B⊥ (code units).

| deck | paper | what | run time (2 cores) |
|---|---|---|---|
| `larmor.in` | Fig. 1 left | gyration in B = 10 e_z, p⊥ = 40, χ₀ = 0.00096: γ 10⁵ → 99 814 (synchrotron loss with g(χ): 99 814) | 6 s |
| `larmor_spiral.in` | Fig. 1 right | p⊥ = 200, B = 50, χ₀ = 0.024: γ → 4.8·10⁴, the orbit spirals inwards (radius 4 → 1.9) | 33 s |
| `spectrum.in` | Fig. 2 | γ = 10⁷ in B_x, momentum fixed (`qed.no_recoil`): dN/du = C₀ w(u, χ); χ = 0.1, 1, 10 via `fields.external_B` | 1 s |
| `spin_dependence.in` | Fig. 3 | as `spectrum.in` with the spin fixed along ±B: dN₊/du − dN₋/du = 2C₀g | 2 × 1 s |
| `sokolov_ternov.in` | Fig. 4 | spin relaxation at fixed χ = 1: ⟨s₂⟩ 1 → −0.838 (eq. 14; −8√3/15 for χ → 0) | 5 s |
| `photon_polarization.in` | Fig. 5 | mean Stokes vector of the photons vs γ (χ = γ/10⁶) and spin, eq. 22 | 0.4 s per point |

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
```

(`analyze.py` takes the same `key=value` overrides as the run.) The samplers and spin models are chosen
with `qed.sampler = inversion | rejection` and `qed.spin_model = mixed | pure` (see the main README, §1
and §5). Outputs: `photons_e.bin` (one record per photon), `qed.txt`, `beams.txt`, `spin.txt`.
The validated versions of these tests are section 26 of `validation/run_validation.sh`.
