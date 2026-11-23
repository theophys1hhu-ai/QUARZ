# QUARZ: radiation of beam electrons and positrons (radiation reaction, photon emission, spin)

High-energy leptons that oscillate in the focusing field of a wake radiate (betatron radiation). QUARZ has
three models of increasing completeness for this radiation. Plasma electrons do not radiate in any of them, and
neither do the beams in a laser envelope (it has no oscillating field).

## When radiation matters: the quantum parameter chi

chi = eta sqrt((gamma E + u x B)^2 - (u.E)^2) in code units, with eta = hbar omega_p / (m c^2)
(1.03e-7 at 2e18 cm^-3, scales as sqrt(n0)). A longitudinal field along the momentum does not count. In a
full ion channel of density n0 the focusing field is r/2, so chi = eta gamma r / 2 for a particle at radius r.

`python3 scripts/quarz_numbers.py --density ... --energy-gev ... --sigma-r-um ... --length-m ...` prints eta,
chi at sigma_r and 3 sigma_r, the mean energy loss over the length (synchrotron with the quantum correction
g(chi), fixed beam size, full channel) and roughly how many photons each particle emits.

| chi (at sigma_r) | regime | what to use |
|---|---|---|
| < 1e-4, or a loss well below the energy spread you care about | negligible | nothing (the default) |
| 1e-4 ... 1e-2 | classical: the mean loss is the Landau-Lifshitz drag; the quantum noise of the emission adds energy spread | `pusher.rr_model = stochastic` (cheap), or `<beam>.qed = 1` |
| > 1e-2 | quantum: fewer and harder photons than classically, recoil of single photons, strong straggling | `<beam>.qed = 1` |

The emitted power grows as gamma^2 r^2: TeV witnesses, large betatron amplitudes (mismatched or offset beams,
hosing) and dense plasmas radiate most. Examples at 2e18 cm^-3: chi ~ 0.004 for 50 GeV electrons at
r = 0.7 c/omega_p; chi ~ 0.25 for a 10 TeV witness with sigma_r = 0.25 (`examples/pwfa_qed_witness.in`).

## Which model

| model | input | what it gives | limits |
|---|---|---|---|
| classical drag | `pusher.beam = imp_rr`, `pusher.rr_n0_cm3 = n0` | deterministic Landau-Lifshitz loss | no energy spread from the emission, chi << 1 |
| stochastic recoil | `pusher.rr_model = stochastic`, `pusher.rr_n0_cm3 = n0` (any pusher) | discrete-photon recoil with the classical spectrum (zero-inflated Gamma per step): mean loss and quantum excitation of the energy spread | chi <~ 1e-2 (no quantum reduction of the power); no photons, no spin effects. `gaussian` (Fokker-Planck) only for comparison: particles can gain energy |
| photon emission (strong-field QED) | `<beam>.qed = 1`, `units.n0_cm3 = n0` | nonlinear Compton scattering in the locally-constant-field approximation (LCFA): photons with their energy, direction and polarization; recoil of each photon; spin-dependent rates and radiative polarization with `<beam>.spin` | electrons and positrons only (charge +-1, mass 1), not rigid beams; no pair creation; the photons are records, they do not propagate, convert or act back |
| deterministic QED | `<beam>.qed = 1` and `qed.model = continuous` | the mean power of the emission (synchrotron loss x g(chi)), the lepton keeps its direction; no photons; with `<beam>.spin` deterministic radiative polarization | no energy spread from the emission. `qed.model = classical`: g = 1 (identical to the classical drag for chi << 1) |

`<beam>.qed` cannot be combined with `pusher.beam = imp_rr` or with `pusher.rr_model = stochastic | gaussian`
(the code refuses: radiation reaction twice). `pusher.rr_model` and `<beam>.qed` act per beam; a driver that
radiates little can stay without (`<beam>.qed = 0`, the default).

A good pair of runs for a result: `<beam>.qed = 1` and the same with `qed.model = continuous`. The mean
energies must agree closely (the deterministic model has the mean power of the stochastic one; with a large
loss the stochastic run loses slightly more, because the power grows as gamma^2 and the spread raises the mean
of gamma^2: 0.3 % of the energy at 20 % loss in `examples/pwfa_qed_witness.in`); the difference in the spread is
the quantum straggling. A third run without radiation gives the wake-only reference.

## Input keys of the photon emission

| key | default | use |
|---|---|---|
| `<beam>.qed` | 0 | 1: photon emission for this beam; needs `units.n0_cm3` (it sets eta) |
| `qed.sampler` | `inversion` | `inversion` (optical depth, inverse cumulative spectrum; can emit several photons per step) or `rejection` (one trial per sub-step); the results agree, keep the default |
| `qed.model` | `stochastic` | `continuous`, `classical`: deterministic, see above |
| `qed.max_prob` | 0.1 | largest emission probability per sub-step; sets the sub-cycling of the push |
| `qed.max_substeps` | 1000 | cap of the sub-steps per particle and step |
| `qed.photon_output` | 1 | 0: no photon file, only the per-step sums in `qed.txt` |
| `<beam>.photon_min_energy` (or `qed.photon_min_energy`) | 0 | write only the photons above this energy (m c^2); all are emitted and counted in `qed.txt` |
| `qed.seed` | 1 | random seed (two seeds show the statistical spread) |
| `qed.table_dir` | none | keep the look-up tables in this directory (they take 0.2 s to build otherwise) |
| `<beam>.spin` | none | `s_x s_y s_z`: spin of every particle (Thomas-BMT); with `<beam>.qed` spin-dependent emission and radiative polarization (Sokolov-Ternov) |
| `qed.spin_model` | `mixed` | `mixed` (spin of a macro-particle as an average, the normal choice) or `pure` (unit spins that collapse) |
| `fields.external_E`, `fields.external_B` | 0 0 0 | constant fields acting on the beams only, in E0; for single-particle tests |

Testing switches (not for physics runs): `qed.no_recoil`, `qed.fixed_spin`, `<beam>.momentum_push = 0`,
`<beam>.spin_precession = 0`.

**Watch out:** `units.n0_cm3` also switches on multiple Coulomb scattering of every non-rigid beam. For a clean
comparison of radiation models set `<beam>.scattering = 0`, or keep it on in all runs.

## Numerics and cost

- **Time step:** keep `time.dt` <= betatron period / 20 as without radiation. The emission itself needs no
  smaller step: each emitting particle is sub-cycled automatically so that its emission probability per
  sub-step stays below `qed.max_prob` (the whole push is sub-cycled in the fields of the step). Convergence
  check: halve `time.dt`, and `qed.max_prob = 0.05`; the mean and the spread of the witness energy must not move
  beyond their statistical errors.
- At very large chi with a large `time.dt` the cap `qed.max_substeps` can be reached: the sub-steps then hold a
  larger probability than `qed.max_prob` (the rejection sampler loses accuracy, the inversion sampler much less).
  Raise the cap or lower `time.dt`.
- **Cost:** only the push of the emitting beam grows (by the number of sub-steps); the plasma sweep usually
  dominates. The photon file grows by 104 bytes per photon: about photons per particle x particles x 104 bytes
  over the run (`quarz_numbers.py` gives the photons per particle). Use `<beam>.photon_min_energy` or
  `qed.photon_output = 0` for long runs with many particles.
- **Statistics:** the energy spread caused by the emission is a statistical result. With N witness particles the
  mean energy is known to the spread / sqrt(N); a spectrum tail of rare hard photons needs more particles or
  several seeds (`qed.seed`).
- Runs are reproducible: the random numbers come from a hash of the particle state, so restarts, threads and MPI
  ranks give bit-identical photons and particles.
- **LCFA validity:** the emission is local, which holds when the betatron strength parameter
  K = gamma k_beta r_beta = r_beta sqrt(gamma/2) (full channel) is >> 1, the wiggler regime of any multi-GeV beam with
  an amplitude above a few 0.01 c/omega_p. For K <~ 1 the soft end of the photon spectrum is overestimated.

## Output

| file | content |
|---|---|
| start-up log | `photon emission (nonlinear Compton, LCFA): sampler ..., eta = ...` per beam, and `QED tables: ...` |
| `qed.txt` | per beam and step: macro-photons emitted, sum of weights, sum of weight x photon energy (m c^2), sum of weight x chi, records lost (normally 0) |
| `photons_<beam>.bin` | one record per photon for the whole run: header char[8] "QZPHOTON", int32 version (1), int32 ncomp (13), then 13 float64 per photon: step, t, x, y, xi (emitter), k_x, k_y, k_z (photon momentum in m c; energy = abs(k) m c^2), w (weight of the emitter), xi_1, xi_2, xi_3 (Stokes vector), chi |
| `spin.txt` | with `<beam>.spin`: mean spin and polarization of each beam per step |
| `beams.txt` | gamma_mean and gamma_rms of the radiating beam: loss and spread |

`scripts/quick_look.py` sums `qed.txt`: photons per beam particle, radiated energy per particle and relative to the
initial energy, mean chi of the emitters. The weights are in the same unit as the beam `charge` column of
`beams.txt` (abs(charge) = sum of the weights of a beam with charge +-1), so photons per particle = sum of w over abs(charge).

Photon spectrum with numpy only:

```python
import numpy as np
with open("out/photons_witness.bin", "rb") as f:
    assert f.read(8) == b"QZPHOTON"; ver, nc = np.fromfile(f, np.int32, 2)
    p = np.fromfile(f, np.float64).reshape(-1, nc)
E = np.sqrt((p[:, 5:8] ** 2).sum(1)) * 0.511e-3          # GeV
dNdE, edges = np.histogram(E, bins=np.logspace(np.log10(E.min()), np.log10(E.max()), 60), weights=p[:, 8])
```

(The repository's `tools/quarz_read.py` has `read_photons` and `read_qedlog`.) Photons go along the emitter's
momentum: their angles k_x/k_z, k_y/k_z show the betatron cone of half-angle about K/gamma.

## What to check in a radiation run

1. The log lists the photon emission for the intended beams, eta matches `quarz_numbers.py`.
2. After the first steps: `quick_look.py` gives chi of the emitters near the estimate and a radiated energy per
   step consistent with the estimated loss over the length.
3. Photon energy (from `qed.txt`) = energy lost by the beam beyond what the wake changes: compare with a run
   without radiation.
4. Mean energy of the stochastic run against `qed.model = continuous` (above: close, slightly lower); spread
   against two seeds.

## A single electron in a constant field (to check chi and the photon rate)

Without plasma (immobile ions of zero density) and with the momentum fixed, chi is set by the field alone:
here gamma = 1e5 in B = 137.6 E0 at 1e18 cm^-3, chi = eta gamma B = 1. The emitted photons per particle and
time are C0 W(chi) = alpha W/(sqrt(3) pi eta gamma) = 1.04 here (W(1) = 5.63); `qed.txt` gives about 2080
macro-photons per step for the 2000 particles, and `quick_look.py` shows <chi> = 1.

```quarz
units.n0_cm3          = 1e18           # sets eta = hbar omega_p / (m c^2) = 7.27e-8
grid.type             = uniform
grid.rmax             = 4
grid.dr               = 0.2
xi.min                = 0
xi.max                = 1
xi.step               = 0.05
time.dt               = 1
time.steps            = 100
plasma.species        = ions           # no plasma: an empty, immobile species
ions.charge           = 1
ions.mass             = 1836
ions.mobile           = 0
plasma.density(x,y,z) = "0"
fields.external_B     = 137.614 0 0    # chi = eta gamma B = 1
beams                 = e
e.n0                  = 1e-9           # test particles: no wake
e.gamma               = 1e5
e.sigma_r             = 0.01
e.sigma_xi            = 0.01
e.xi0                 = 0.5
e.emit_n              = 0
e.nparticles          = 2000
e.passive             = 1              # the beam makes no fields
e.scattering          = 0
e.qed                 = 1
e.momentum_push       = 0              # testing: the field only sets chi
qed.no_recoil         = 1              # testing: chi stays fixed
output.dir            = out_qed_test
output.every          = 100            # axis files at the start and the end (for quick_look.py)
```

Further single-particle and betatron tests with their theory are in the repository's `examples/QED/`
(Larmor motion with losses, spectra, spin dependence, Sokolov-Ternov polarization, photon polarization,
betatron oscillation with A ~ gamma^(1/8), PWFA with a 10 TeV witness).
