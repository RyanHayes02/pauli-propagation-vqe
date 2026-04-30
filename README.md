# VQE via Pauli Propagation

Classical simulation of Variational Quantum Eigensolver (VQE) using Pauli propagation for frustrated quantum spin systems. Benchmarked against DMRG.

## Results summary

| System | Qubits | Best error | Notes |
|--------|--------|-----------|-------|
| 1D Transverse Field Ising (OBC) | 50 | <0.5% (away from critical point) | δc=1e-5, L=5&10 |
| 1D J1-J2 Heisenberg (OBC) | 20 | ~0% at J2/J1=0.5 (MG point) | δc=1e-4, MG dimer ref, L=5&10 |
| 2D J1-J2 Heisenberg 4×4 (PBC) | 16 | **0.64% at J2/J1=0.535** | δc=3e-4, L=5 |
| 2D J1-J2 Heisenberg 6×6 (PBC) | 36 | 6-17% (partial, re-eval at δc=1e-5) | δc=3e-4, L=5 |

## Method

**Pauli propagation:** The Hamiltonian H is expressed as a weighted sum of Pauli strings and propagated backwards through the circuit in the Heisenberg picture. Terms with |coefficient| < δc are truncated to control cost. Overlap with a reference state gives ⟨E⟩.

**Optimizer:** SPSA + Adam. SPSA estimates gradients from two forward passes only — compatible with truncated Pauli sums where exact gradients are unavailable.

**Benchmark:** All results compared to DMRG (ITensors.jl). Metric: ΔE/|E_DMRG|.

## Phase diagram (2D J1-J2, square lattice PBC)

```
J2/J1:   0 ────── 0.535 ──── 0.610 ──────→
Phase:   Néel AFM │  VBS (debated)  │  Stripe AFM
```

## Reference states

| System | Reference state | Why |
|--------|----------------|-----|
| TFI | \|+⟩^⊗n | Ground state of pure transverse field limit |
| 1D J1-J2 | MG dimer state | Exact ground state at J2/J1=0.5 |
| 2D J1-J2 (J2/J1 ≤ 0.535) | Néel state | Checkerboard ↑↓ pattern matches dominant order |
| 2D J1-J2 (J2/J1 > 0.535) | Stripe state | Alternating rows ↑↑↑/↓↓↓ matches stripe order |

## Circuit ansätze

**TFI (HVA-inspired):** `RZ · RX · RZZ` per layer. RZ breaks Z₂ symmetry of the Hamiltonian.

**1D J1-J2:** `RZ · RX` single-qubit layers + `RXX · RYY · RZZ` on NN then NNN bonds per layer.

**2D J1-J2:** `RXX · RYY · RZZ` on NN bonds then NNN bonds per layer. No single-qubit layers (preserves SU(2) symmetry). 6 shared parameters per layer.

## Repository structure

```
pauli-propagation-vqe/
├── README.md
├── report.pdf                        # full project report
├── .gitignore
├── code/
│   ├── 01_1D_tfi_n50_1e-5.jl        # 1D TFI sweep (n=50, OBC)
│   ├── 02_1D_j1j2_n20_1e-4.jl       # 1D J1-J2 sweep (n=20, OBC, MG init)
│   ├── 03_2D_j1j2_3e-4_4x4.jl       # 2D 4x4 PBC sweep
│   ├── 04_2D_j1j2_3e-4_6x6.jl       # 2D 6x6 PBC sweep
│   └── 05_reeval_6x6_pbc.jl          # 6x6 re-evaluation at finer truncation
└── figures/
    ├── 01_fig_1D_tfi_n50_1e-5.pdf
    ├── 02_fig_1D_j1j2_n20_1e-4.pdf
    ├── 03_fig_2D_j1j2_3e-4_4x4.pdf
    └── 04_fig_2D_j1j2_3e-4_6x6.pdf
```

## Requirements

```
Julia 1.12+
PauliPropagation.jl   (https://github.com/MSRudolph/PauliPropagation.jl)
ITensors.jl / ITensorMPS.jl
Plots.jl
Serialization (stdlib)
```

Activate the project environment:
```bash
julia --project=. --threads=30
```

## Running

All simulations were run on the ETH Zürich Euler HPC cluster using Julia 1.12, with threading ranging from 10 to 30 threads depending on the system size. Jobs were submitted via SLURM with memory allocations up to 125GB for the larger 6×6 runs. Checkpoints are saved periodically so jobs can be killed and resumed without losing progress.

## Notes

- Checkpoint `.jls` files and DMRG cache files are not tracked by git — they are large binary Julia serialized objects
- 2D 6×6 optimization reached 1000-2700 of 3000 target iterations before wall-clock limit. Parameters were re-evaluated at δc=1e-5 giving 6-17% errors
- Snake ordering is used for 2D lattice site labelling: odd rows go left→right, even rows right→left

## Author

Ryan Hayes — ETH Zürich  
Supervisor: Dr. Juan Carrasquilla Alvarez  
PhD Student: Matteo D'Anna
