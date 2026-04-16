using Pkg
Pkg.activate(@__DIR__)

using PauliPropagation
using Random
using LinearAlgebra
using Base.Threads
using Plots
using ITensors, ITensorMPS
using Serialization

println("Julia threads available: ", Threads.nthreads())
flush(stdout)

############################################################
# ---- PauliSum helpers ----
############################################################

function copy_paulisum(H::PauliSum)
    Hcopy = PauliSum(H.nqubits)
    for (ps, coeff) in H
        set!(Hcopy, ps, coeff)
    end
    return Hcopy
end

# H = J1 Σ_{<i,j>}(XX+YY+ZZ)  +  J2 Σ_{<<i,j>>}(XX+YY+ZZ)
# Uses standard (full) Pauli matrices — consistent with ITensors "Qubit" site type.
function j1j2_Hamiltonian(n::Int; J1::Real=1.0, J2::Real=0.0, periodic::Bool=false)
    H = PauliSum(n)
    # NN (J1) bonds
    for i in 1:n-1
        PauliPropagation.add!(H, [:X, :X], [i, i+1], J1)
        PauliPropagation.add!(H, [:Y, :Y], [i, i+1], J1)
        PauliPropagation.add!(H, [:Z, :Z], [i, i+1], J1)
    end
    if periodic
        PauliPropagation.add!(H, [:X, :X], [n, 1], J1)
        PauliPropagation.add!(H, [:Y, :Y], [n, 1], J1)
        PauliPropagation.add!(H, [:Z, :Z], [n, 1], J1)
    end
    # NNN (J2) bonds
    for i in 1:n-2
        PauliPropagation.add!(H, [:X, :X], [i, i+2], J2)
        PauliPropagation.add!(H, [:Y, :Y], [i, i+2], J2)
        PauliPropagation.add!(H, [:Z, :Z], [i, i+2], J2)
    end
    if periodic
        # The two "wrap-around" NNN pairs: (n-1,1) and (n,2)
        PauliPropagation.add!(H, [:X, :X], [n-1, 1], J2)
        PauliPropagation.add!(H, [:Y, :Y], [n-1, 1], J2)
        PauliPropagation.add!(H, [:Z, :Z], [n-1, 1], J2)
        PauliPropagation.add!(H, [:X, :X], [n, 2],   J2)
        PauliPropagation.add!(H, [:Y, :Y], [n, 2],   J2)
        PauliPropagation.add!(H, [:Z, :Z], [n, 2],   J2)
    end
    return H
end

############################################################
# ---- Ansatz circuit  (5 shared params / layer)         ----
#   per layer:  Rz(α) × n  |  Rx(β) × n  |              ----
#               RZZ(γ) × n_nn  |  RXX(δ) × n_nn  |      ----
#               RYY(ε) × n_nn                            ----
############################################################

const N_SHARED = 5   # (α, β, γ, δ, ε) per layer

function j1j2_circuit(n::Int; nlayers::Int=5, periodic::Bool=false)
    topology = bricklayertopology(n; periodic=periodic)
    circuit  = PauliRotation[]
    for _ in 1:nlayers
        for i in 1:n
            push!(circuit, PauliRotation(:Z, i))
        end
        for i in 1:n
            push!(circuit, PauliRotation(:X, i))
        end
        for (i, j) in topology
            push!(circuit, PauliRotation([:Z, :Z], [i, j]))
        end
        for (i, j) in topology
            push!(circuit, PauliRotation([:X, :X], [i, j]))
        end
        for (i, j) in topology
            push!(circuit, PauliRotation([:Y, :Y], [i, j]))
        end
    end
    return circuit
end

function expand_shared_thetas(thetas_shared, n::Int, nlayers::Int, n_nn::Int)
    thetas_full = Vector{Float64}(undef, nlayers * (2n + 3n_nn))
    idx = 1
    for k in 1:nlayers
        α = thetas_shared[N_SHARED*(k-1) + 1]   # Rz
        β = thetas_shared[N_SHARED*(k-1) + 2]   # Rx
        γ = thetas_shared[N_SHARED*(k-1) + 3]   # RZZ
        δ = thetas_shared[N_SHARED*(k-1) + 4]   # RXX
        ε = thetas_shared[N_SHARED*(k-1) + 5]   # RYY
        for _ in 1:n;    thetas_full[idx] = α; idx += 1; end
        for _ in 1:n;    thetas_full[idx] = β; idx += 1; end
        for _ in 1:n_nn; thetas_full[idx] = γ; idx += 1; end
        for _ in 1:n_nn; thetas_full[idx] = δ; idx += 1; end
        for _ in 1:n_nn; thetas_full[idx] = ε; idx += 1; end
    end
    return thetas_full
end

function propagate_obs(circuit, H, thetas_shared, n, nlayers, n_nn; min_abs_coeff=1e-4)
    thetas_full = expand_shared_thetas(thetas_shared, n, nlayers, n_nn)
    obs = copy_paulisum(H)
    propagate!(circuit, obs, thetas_full; min_abs_coeff=min_abs_coeff)
    return obs
end

############################################################
# ---- Reference-state overlaps via Clifford circuits   ----
#                                                        ----
#  Both functions conjugate the (already-VQE-propagated) ----
#  observable O with a Clifford circuit V that prepares  ----
#  the reference state from |0⟩, then call              ----
#  overlapwithzero.  This gives ⟨ref|O|ref⟩.           ----
############################################################

# Neel state  |↑↓↑↓…⟩ = |1010…⟩  — apply X on odd sites (1,3,5,…)
function overlap_with_neel(psum::PauliSum)
    n = psum.nqubits
    circ = CliffordGate[]
    for j in 1:2:n
        push!(circ, CliffordGate(:X, j))
    end
    rotated = propagate(circ, psum, []; min_abs_coeff=1e-12)
    return overlapwithzero(rotated)
end

# Majumdar–Ghosh dimer state — singlets on pairs (1,2),(3,4),…
# Clifford preparation from |0⟩: X then H on odd site, X on even site, CNOT(odd→even)
function overlap_with_mg(psum::PauliSum)
    n = psum.nqubits
    circ = CliffordGate[]
    for j in 1:2:n
        j + 1 > n && continue
        push!(circ, CliffordGate(:X,    j))
        push!(circ, CliffordGate(:H,    j))
        push!(circ, CliffordGate(:X,    j+1))
        push!(circ, CliffordGate(:CNOT, [j, j+1]))
    end
    rotated = propagate(circ, psum, []; min_abs_coeff=1e-12)
    return overlapwithzero(rotated)
end

# Generic energy wrapper — swap overlapwith to change reference state
function expect_fast(circuit, H, thetas_shared, n, nlayers, n_nn;
                     min_abs_coeff=1e-4, overlapwith=overlap_with_neel)
    psum = propagate_obs(circuit, H, thetas_shared, n, nlayers, n_nn;
                         min_abs_coeff=min_abs_coeff)
    return overlapwith(psum)
end

############################################################
# ---- Entropy & norm ----
############################################################

function normalize_paulisum(psum)
    frob = norm(psum, 2)
    out  = copy_paulisum(psum)
    for (ps, coeff) in out
        set!(out, ps, coeff / frob)
    end
    return out
end

function stabilizer_renyi_entropy(psum; alpha::Float64=2.0)
    @assert alpha != 1.0
    psum_norm = normalize_paulisum(psum)
    inside    = norm(psum_norm, 2*alpha)^2
    return (alpha / (1 - alpha)) * log(inside)
end

############################################################
# ---- DMRG  (Qubit site types → standard Paulis)       ----
#  Fixing: (a) S=1/2 → Qubit (avoids ×4 energy factor)  ----
#          (b) periodic NNN wrap-around bonds            ----
############################################################

function dmrg_j1j2_energy(n::Int; J1::Real=1.0, J2::Real=0.0, periodic::Bool=false,
                           nsweeps::Int=50,
                           maxdim::Vector{Int}=[10,20,50,100,200,400,600,800],
                           cutoff::Real=1e-10,
                           linkdims0::Int=10,
                           seed::Int=0)
    Random.seed!(seed)
    sites = siteinds("Qubit", n)          # standard Pauli X/Y/Z, not S=½ operators
    os    = OpSum()
    # NN
    for i in 1:n-1
        os += float(J1), "X", i, "X", i+1
        os += float(J1), "Y", i, "Y", i+1
        os += float(J1), "Z", i, "Z", i+1
    end
    if periodic
        os += float(J1), "X", n, "X", 1
        os += float(J1), "Y", n, "Y", 1
        os += float(J1), "Z", n, "Z", 1
    end
    # NNN
    for i in 1:n-2
        os += float(J2), "X", i, "X", i+2
        os += float(J2), "Y", i, "Y", i+2
        os += float(J2), "Z", i, "Z", i+2
    end
    if periodic
        os += float(J2), "X", n-1, "X", 1   # wrap-around NNN pair 1
        os += float(J2), "Y", n-1, "Y", 1
        os += float(J2), "Z", n-1, "Z", 1
        os += float(J2), "X", n, "X", 2     # wrap-around NNN pair 2
        os += float(J2), "Y", n, "Y", 2
        os += float(J2), "Z", n, "Z", 2
    end
    H_mpo = MPO(os, sites)
    psi0  = random_mps(sites; linkdims=linkdims0)
    energy, _ = dmrg(H_mpo, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=[cutoff])
    return energy
end

############################################################
# ---- SPSA + ADAM ----
############################################################

function optimize_spsa_adam(f, thetas0;
                             maxiter=7500,
                             eta=0.001,
                             delta=0.005,
                             beta1=0.9,
                             beta2=0.999,
                             eps=1e-5,
                             show_every=500)
    p      = length(thetas0)
    theta  = copy(thetas0)
    m      = zeros(p)
    v      = zeros(p)
    f_last = NaN
    for i in 1:maxiter
        local chi, Delta, C_plus, C_minus, G, m_hat, v_hat
        chi     = rand(p) .* 2 .- 1
        Delta   = delta .* chi ./ norm(chi)
        C_plus  = f(theta .+ Delta)
        C_minus = f(theta .- Delta)
        G       = (C_plus - C_minus) / (2*delta) .* Delta
        m       = beta1 .* m .+ (1 - beta1) .* G
        v       = beta2 .* v .+ (1 - beta2) .* G.^2
        m_hat   = m ./ (1 - beta1^i)
        v_hat   = v ./ (1 - beta2^i)
        theta   = theta .- eta .* m_hat ./ (sqrt.(v_hat) .+ eps)
        f_last  = (C_plus + C_minus) / 2
        if i % show_every == 0
            println("      [tid=$(Threads.threadid())] iter $i  f≈$(round(f_last, digits=6))")
            flush(stdout)
        end
    end
    return theta, f_last
end

############################################################
# ---- Main sweep ----
############################################################

relerr(Evqe, Edmrg) = abs(Evqe - Edmrg) / abs(Edmrg)

function run_j1j2_sweep_threaded(;
        n              = 20,
        j2j1_list      = [0.1,0.2,0.25,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1.0,1.1,1.2],
        nlayers_list   = [5, 10],
        J1             = 1.0,
        scale          = 0.05,
        min_abs_coeff  = 1e-4,
        maxiter_neel   = 5000,
        maxiter_mg     = 5000,
        eta            = 0.001,
        delta_spsa     = 0.005,
        periodic       = false,
        outdir         = ".")

    println("\n=== J1-J2 sweep  n=$n  threads=$(Threads.nthreads()) ===")
    flush(stdout)

    topology = bricklayertopology(n; periodic=periodic)
    n_nn     = length(topology)

    circuits = Dict(L => j1j2_circuit(n; nlayers=L, periodic=periodic)
                    for L in nlayers_list)

    ratios  = collect(j2j1_list)
    ng      = length(ratios)

    # result containers keyed by (init_sym, L)
    inits       = [:mg]
    result_keys = [(init, L) for init in inits for L in nlayers_list]
    results     = Dict(k => Vector{Union{Nothing,NamedTuple}}(nothing, ng) for k in result_keys)

    # ---- DMRG (sequential -- with serialization cache) ----
    dmrg_cache_path = joinpath(outdir, "dmrg_n$(n)_j1j2.jls")
    E_dmrg_vec = Vector{Float64}(undef, ng)

    cached = isfile(dmrg_cache_path) ? deserialize(dmrg_cache_path) : Dict{Float64,Float64}()

    any_new = false
    for (gi, r) in enumerate(ratios)
        if haskey(cached, float(r))
            println("  DMRG  J2/J1=$r  [cached]")
            E_dmrg_vec[gi] = cached[float(r)]
        else
            println("  DMRG  J2/J1=$r  [computing...]")
            flush(stdout)
            E_dmrg_vec[gi]   = dmrg_j1j2_energy(n; J1=J1, J2=J1*r, periodic=periodic)
            cached[float(r)] = E_dmrg_vec[gi]
            any_new = true
        end
        flush(stdout)
    end

    if any_new
        serialize(dmrg_cache_path, cached)
        println("DMRG results serialized -> $dmrg_cache_path")
    else
        println("All DMRG values loaded from cache.")
    end
    flush(stdout)

    # ---- VQE (threaded over all (ratio, L, init) combos) ----
    tasks = [(gi, L, init) for gi in 1:ng for L in nlayers_list for init in inits]
    println("Total VQE tasks: $(length(tasks))  " *
            "($(ng) ratios × $(length(nlayers_list)) layer counts × 2 init states)")
    flush(stdout)

    Threads.@threads for ti in eachindex(tasks)
        gi, L, init = tasks[ti]
        ratio   = ratios[gi]
        J2      = J1 * ratio
        E_dmrg  = E_dmrg_vec[gi]
        circuit = circuits[L]
        H       = j1j2_Hamiltonian(n; J1=J1, J2=J2, periodic=periodic)

        overlapfn = (init == :neel) ? overlap_with_neel : overlap_with_mg
        iters     = (init == :mg)   ? maxiter_mg        : maxiter_neel

        println("  [tid=$(Threads.threadid())] J₂/J₁=$ratio  L=$L  init=$init  → VQE")
        flush(stdout)

        rng     = MersenneTwister(0)
        thetas0 = randn(rng, N_SHARED * L) .* scale

        f = θ -> expect_fast(circuit, H, θ, n, L, n_nn;
                              min_abs_coeff=min_abs_coeff, overlapwith=overlapfn)

        thetas_opt, _ = optimize_spsa_adam(f, thetas0;
                                           maxiter=iters, eta=eta,
                                           delta=delta_spsa, show_every=500)

        psum_opt = propagate_obs(circuit, H, thetas_opt, n, L, n_nn;
                                 min_abs_coeff=min_abs_coeff)
        E_opt  = overlapfn(psum_opt)
        S_opt  = stabilizer_renyi_entropy(psum_opt; alpha=2.0)
        err    = relerr(E_opt, E_dmrg)

        println("  [tid=$(Threads.threadid())] J₂/J₁=$ratio  L=$L  init=$init" *
                "  E=$(round(E_opt,digits=5))  err=$(round(err,sigdigits=3))")
        flush(stdout)

        results[(init, L)][gi] = (
            ratio  = float(ratio),
            E_dmrg = E_dmrg,
            E_opt  = E_opt,
            S_opt  = S_opt,
            err    = err,
        )

        serialize(
            joinpath(outdir, "ckpt_n$(n)_L$(L)_$(init)_r$(ratio).jls"),
            results[(init, L)][gi]
        )
    end

    return Dict(k => filter(!isnothing, results[k]) for k in result_keys)
end

############################################################
# ---- Plotting ----
############################################################

function make_plots(results, nlayers_list, n, outdir)
    mkpath(outdir)

    # color = layer depth, linestyle/marker = init state
    colors     = Dict(5 => :blue,   10 => :red)
    lstyles    = Dict(:neel => :solid, :mg => :dash)
    mshapes    = Dict(:neel => :circle, :mg => :diamond)

    p1 = plot(xlabel="J₂/J₁", ylabel="ΔE / |E_DMRG|",
              title="Relative energy error  (n=$n, δc=1e-4)",
              legend=:topleft, left_margin=10Plots.mm, bottom_margin=10Plots.mm)

    p2 = plot(xlabel="J₂/J₁", ylabel="OSE  S₂",
              title="Operator Stabilizer Entropy  (n=$n, δc=1e-4)",
              legend=:topleft, left_margin=10Plots.mm, bottom_margin=10Plots.mm)

    for L in sort(nlayers_list)
        for init in [:neel, :mg]
            key = (init, L)
            haskey(results, key)  || continue
            rs  = sort(results[key]; by=r->r.ratio)
            isempty(rs)           && continue

            xs  = [r.ratio for r in rs]
            err = [r.err   for r in rs]
            Ss  = [r.S_opt for r in rs]
            lbl = "ℓ=$L, $(init)"

            plot!(p1, xs, err;
                  color=colors[L], linestyle=lstyles[init],
                  marker=mshapes[init], label=lbl, markersize=5)
            plot!(p2, xs, Ss;
                  color=colors[L], linestyle=lstyles[init],
                  marker=mshapes[init], label=lbl, markersize=5)
        end
    end

    # Mark known critical / special points
    for p in (p1, p2)
        vline!(p, [0.25]; color=:gray,     linestyle=:dot,  label="J₂/J₁=0.25")
        vline!(p, [0.5];  color=:darkgray, linestyle=:dash, label="J₂/J₁=0.5 (MG pt)")
    end

    panel = plot(p1, p2;
                 layout=(1,2),
                 size=(1200, 500),
                 plot_title="J1-J2 Heisenberg  n=$n OBC — MG init, L=10, δc=1e-4",
                 top_margin=10Plots.mm)

    outpath = joinpath(outdir, "fig_panel_n$(n)_j1j2_1e4.pdf")
    savefig(panel, outpath)
    println("Saved → $outpath")
    return panel
end

############################################################
# ---- Entry point ----
############################################################

n      = 20
outdir = joinpath(@__DIR__, "fig_j1j2_out", "n$(n)_L10_mg_1e4_new")
mkpath(outdir)

results = run_j1j2_sweep_threaded(;
    n             = n,
    j2j1_list     = [0.1, 0.2, 0.25, 0.3, 0.4, 0.45,
                     0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    nlayers_list  = [10],
    J1            = 1.0,
    scale         = 0.05,
    min_abs_coeff = 1e-4,
    maxiter_neel  = 5000,
    maxiter_mg    = 5000,
    eta           = 0.001,
    delta_spsa    = 0.005,
    periodic      = false,
    outdir        = outdir,
)

serialize(joinpath(outdir, "results_n$(n)_j1j2_1e4.jls"), results)
println("Results serialized → $outdir")

make_plots(results, [10], n, outdir)
println("\nAll done.")
