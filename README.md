# SIH26052 — AI/ML-Enabled Adaptive Noise Cancellation (ANC)

DRDO problem statement: suppress stationary, non-stationary, and impulsive 
defence noises while preserving speech intelligibility, real-time on embedded hardware.

## Approach
Hybrid: NLMS adaptive filter (real-time backbone) + lightweight RNNoise-style 
neural model trained on defence-specific noise categories, targeting Cortex-M 
class embedded hardware (emulated via QEMU/Renode).

## Status
- [x] Milestone 1: Signal foundations & dataset (baseline STOI established)
- [x] Milestone 2: NLMS adaptive filter core
- [x] Milestone 3: AI/ML noise-type-robust model
- [ ] Milestone 4: Embedded porting & emulation
- [ ] Milestone 5: Digital twin & system testing
- [ ] Milestone 6: Optimization & demo packaging