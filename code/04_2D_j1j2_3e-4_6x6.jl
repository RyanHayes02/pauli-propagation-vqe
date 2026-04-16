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
# ---- 2D lattice helpers ----
############################################################

function snake_index(row::Int, col::Int, Lx::Int, Ly::Int)
    if isodd(row)
        return (row - 1) * Ly + col
    else
        return (row - 1) * Ly + (Ly - col + 1)
    end
end

function nn_bonds_2d(Lx::Int, Ly::Int; periodic::Bool=false)
    bonds = Tuple{Int,Int}[]
    for r in 1:Lx, c in 1:Ly
        s = snake_index(r, c, Lx, Ly)
        if c < Ly
            push!(bonds, minmax(s, snake_index(r, c+1, Lx, Ly)))
        elseif periodic
            push!(bonds, minmax(s, snake_index(r, 1, Lx, Ly)))
        end
        if r < Lx
            push!(bonds, minmax(s, snake_index(r+1, c, Lx, Ly)))
        elseif periodic
            push!(bonds, minmax(s, snake_index(1, c, Lx, Ly)))
        end
    end
    return unique!(sort!(bonds))
end

function nnn_bonds_2d(Lx::Int, Ly::Int; periodic::Bool=false)
    bonds = Tuple{Int,Int}[]
    for r in 1:Lx, c in 1:Ly
        s = snake_index(r, c, Lx, Ly)
        for (dr, dc) in [(1,1), (1,-1)]
            nr, nc = r + dr, c + dc
            if periodic
                nr = mod1(nr, Lx); nc = mod1(nc, Ly)
            end
            if 1 <= nr <= Lx && 1 <= nc <= Ly
                push!(bonds, minmax(s, snake_index(nr, nc, Lx, Ly)))
            end
        end
    end
    return unique!(sort!(bonds))
end

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

function j1j2_2d_Hamiltonian(Lx::Int, Ly::Int; J1::Real=1.0, J2::Real=0.0,
                              periodic::Bool=false)
    n   = Lx * Ly
    H   = PauliSum(n)
    nn  = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    for (i, j) in nn
        PauliPropagation.add!(H, [:X, :X], [i, j], J1)
        PauliPropagation.add!(H, [:Y, :Y], [i, j], J1)
        PauliPropagation.add!(H, [:Z, :Z], [i, j], J1)
    end
    for (i, j) in nnn
        PauliPropagation.add!(H, [:X, :X], [i, j], J2)
        PauliPropagation.add!(H, [:Y, :Y], [i, j], J2)
        PauliPropagation.add!(H, [:Z, :Z], [i, j], J2)
    end
    return H
end

############################################################
# ---- Circuit: RXX·RYY·RZZ on NN then NNN per layer ----
############################################################

const N_SHARED_2D = 6

function j1j2_2d_circuit_supervisor(Lx::Int, Ly::Int; nlayers::Int=5,
                                     periodic::Bool=false)
    nn      = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn     = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    circuit = PauliRotation[]
    for _ in 1:nlayers
        for (i, j) in nn;  push!(circuit, PauliRotation([:X, :X], [i, j])); end
        for (i, j) in nn;  push!(circuit, PauliRotation([:Y, :Y], [i, j])); end
        for (i, j) in nn;  push!(circuit, PauliRotation([:Z, :Z], [i, j])); end
        for (i, j) in nnn; push!(circuit, PauliRotation([:X, :X], [i, j])); end
        for (i, j) in nnn; push!(circuit, PauliRotation([:Y, :Y], [i, j])); end
        for (i, j) in nnn; push!(circuit, PauliRotation([:Z, :Z], [i, j])); end
    end
    return circuit
end

function expand_shared_thetas_2d(thetas_shared, n::Int, nlayers::Int,
                                  n_nn::Int, n_nnn::Int)
    thetas_full = Vector{Float64}(undef, nlayers * 3 * (n_nn + n_nnn))
    idx = 1
    for k in 1:nlayers
        a_nn   = thetas_shared[N_SHARED_2D*(k-1) + 1]
        b_nn   = thetas_shared[N_SHARED_2D*(k-1) + 2]
        g_nn   = thetas_shared[N_SHARED_2D*(k-1) + 3]
        a_nnn  = thetas_shared[N_SHARED_2D*(k-1) + 4]
        b_nnn  = thetas_shared[N_SHARED_2D*(k-1) + 5]
        g_nnn  = thetas_shared[N_SHARED_2D*(k-1) + 6]
        for _ in 1:n_nn;  thetas_full[idx] = a_nn;  idx += 1; end
        for _ in 1:n_nn;  thetas_full[idx] = b_nn;  idx += 1; end
        for _ in 1:n_nn;  thetas_full[idx] = g_nn;  idx += 1; end
        for _ in 1:n_nnn; thetas_full[idx] = a_nnn; idx += 1; end
        for _ in 1:n_nnn; thetas_full[idx] = b_nnn; idx += 1; end
        for _ in 1:n_nnn; thetas_full[idx] = g_nnn; idx += 1; end
    end
    return thetas_full
end

function propagate_obs_2d(circuit, H, thetas_shared, n, nlayers, n_nn, n_nnn;
                           min_abs_coeff=3e-4)
    thetas_full = expand_shared_thetas_2d(thetas_shared, n, nlayers, n_nn, n_nnn)
    obs = copy_paulisum(H)
    propagate!(circuit, obs, thetas_full; min_abs_coeff=min_abs_coeff)
    return obs
end

function expect_fast_2d(circuit, H, thetas_shared, n, nlayers, n_nn, n_nnn;
                        min_abs_coeff=3e-4, overlapwith)
    psum = propagate_obs_2d(circuit, H, thetas_shared, n, nlayers, n_nn, n_nnn;
                            min_abs_coeff=min_abs_coeff)
    return overlapwith(psum)
end

############################################################
# ---- Reference-state overlaps ----
############################################################

function overlap_with_neel_2d(psum::PauliSum, Lx::Int, Ly::Int)
    circ = CliffordGate[]
    for r in 1:Lx, c in 1:Ly
        if iseven(r + c)
            push!(circ, CliffordGate(:X, snake_index(r, c, Lx, Ly)))
        end
    end
    rotated = propagate(circ, psum, []; min_abs_coeff=1e-12)
    return overlapwithzero(rotated)
end

function overlap_with_stripe_2d(psum::PauliSum, Lx::Int, Ly::Int)
    circ = CliffordGate[]
    for r in 1:Lx, c in 1:Ly
        if iseven(r)
            push!(circ, CliffordGate(:X, snake_index(r, c, Lx, Ly)))
        end
    end
    rotated = propagate(circ, psum, []; min_abs_coeff=1e-12)
    return overlapwithzero(rotated)
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
# ---- DMRG ----
############################################################

function dmrg_j1j2_2d_energy(Lx::Int, Ly::Int;
                              J1::Real=1.0, J2::Real=0.0,
                              periodic::Bool=false,
                              nsweeps::Int=50,
                              maxdim::Vector{Int}=[20,50,100,200,400,800,1600,3200],
                              cutoff::Real=1e-10,
                              linkdims0::Int=10,
                              seed::Int=0)
    Random.seed!(seed)
    n     = Lx * Ly
    sites = siteinds("Qubit", n)
    os    = OpSum()
    nn    = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn   = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    for (i, j) in nn
        os += float(J1), "X", i, "X", j
        os += float(J1), "Y", i, "Y", j
        os += float(J1), "Z", i, "Z", j
    end
    for (i, j) in nnn
        os += float(J2), "X", i, "X", j
        os += float(J2), "Y", i, "Y", j
        os += float(J2), "Z", i, "Z", j
    end
    H_mpo = MPO(os, sites)
    psi0  = random_mps(sites; linkdims=linkdims0)
    energy, psi = dmrg(H_mpo, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=[cutoff])
    max_bd = maximum(dim(linkind(psi, b)) for b in 1:n-1)
    println("  DMRG done: E=$(round(energy,digits=6))  max_bond_dim=$max_bd")
    flush(stdout)
    return energy, max_bd
end

############################################################
# ---- SPSA + Adam with convergence tracking ----
############################################################

function optimize_spsa_adam_tracked(f, thetas0;
                                     maxiter=3000,
                                     eta=0.001,
                                     delta=0.005,
                                     beta1=0.9,
                                     beta2=0.999,
                                     eps=1e-5,
                                     show_every=100,
                                     checkpoint_fn=nothing)
    p        = length(thetas0)
    theta    = copy(thetas0)
    m        = zeros(p)
    v        = zeros(p)
    iters    = Int[]
    energies = Float64[]

    for i in 1:maxiter
        chi      = rand(p) .* 2 .- 1
        Delta    = delta .* chi ./ norm(chi)
        C_plus   = f(theta .+ Delta)
        C_minus  = f(theta .- Delta)
        G        = (C_plus - C_minus) / (2*delta) .* Delta
        m        = beta1 .* m .+ (1 - beta1) .* G
        v        = beta2 .* v .+ (1 - beta2) .* G.^2
        m_hat    = m ./ (1 - beta1^i)
        v_hat    = v ./ (1 - beta2^i)
        theta    = theta .- eta .* m_hat ./ (sqrt.(v_hat) .+ eps)
        f_approx = (C_plus + C_minus) / 2

        if i % show_every == 0
            push!(iters, i)
            push!(energies, f_approx)
            println("      [tid=$(Threads.threadid())] iter $i  f~$(round(f_approx, digits=6))")
            flush(stdout)
            if !isnothing(checkpoint_fn)
                checkpoint_fn(theta, i, f_approx, iters, energies)
            end
        end
    end
    return theta, iters, energies
end

############################################################
# ---- Main sweep ----
############################################################

relerr(Evqe, Edmrg) = abs(Evqe - Edmrg) / abs(Edmrg)

function run_2d_sweep_threaded(;
        Lx             = 6,
        Ly             = 6,
        j2j1_list      = [0.1, 0.3, 0.5, 0.535,
                          0.55, 0.58, 0.61, 0.65,
                          0.7, 1.0],
        nlayers_list   = [5],
        J1             = 1.0,
        scale          = 0.05,
        min_abs_coeff  = 3e-4,
        maxiter        = 3000,
        eta            = 0.001,
        delta_spsa     = 0.005,
        periodic       = true,
        outdir         = ".")

    n     = Lx * Ly
    nn    = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn   = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    n_nn  = length(nn)
    n_nnn = length(nnn)

    println("\n=== 2D J1-J2 sweep  $(Lx)x$(Ly)=$(n) qubits  threads=$(Threads.nthreads()) ===")
    println("NN bonds: $n_nn   NNN bonds: $n_nnn")
    println("Circuit: RXX(NN)->RYY(NN)->RZZ(NN)->RXX(NNN)->RYY(NNN)->RZZ(NNN) per layer")
    println("Inits: neel for J2/J1 <= 0.535, stripe for J2/J1 > 0.535  (no columnar dimer)")
    flush(stdout)

    circuits = Dict(L => j1j2_2d_circuit_supervisor(Lx, Ly; nlayers=L, periodic=periodic)
                    for L in nlayers_list)

    ratios  = collect(j2j1_list)
    ng      = length(ratios)

    inits       = [:neel, :stripe]
    result_keys = [(init, L) for init in inits for L in nlayers_list]
    results     = Dict(k => Vector{Union{Nothing,NamedTuple}}(nothing, ng) for k in result_keys)

    # ---- DMRG with cache ----
    dmrg_cache_path = joinpath(outdir, "dmrg_$(Lx)x$(Ly)_j1j2.jls")
    E_dmrg_vec  = Vector{Float64}(undef, ng)
    BD_dmrg_vec = Vector{Int}(undef, ng)

    cached  = isfile(dmrg_cache_path) ? deserialize(dmrg_cache_path) : Dict{Float64,Tuple{Float64,Int}}()
    any_new = false

    for (gi, r) in enumerate(ratios)
        if haskey(cached, float(r))
            E_dmrg_vec[gi], BD_dmrg_vec[gi] = cached[float(r)]
            println("  DMRG  J2/J1=$r  [cached]  E=$(round(E_dmrg_vec[gi],digits=5))  max_bd=$(BD_dmrg_vec[gi])")
        else
            println("  DMRG  J2/J1=$r  [computing...]")
            flush(stdout)
            E, bd = dmrg_j1j2_2d_energy(Lx, Ly; J1=J1, J2=J1*r, periodic=periodic)
            E_dmrg_vec[gi]   = E
            BD_dmrg_vec[gi]  = bd
            cached[float(r)] = (E, bd)
            any_new = true
        end
        flush(stdout)
    end

    if any_new
        serialize(dmrg_cache_path, cached)
        println("DMRG serialized -> $dmrg_cache_path")
    else
        println("All DMRG values from cache.")
    end
    flush(stdout)

    # Build only the meaningful tasks: neel <= 0.535, stripe > 0.535
    tasks = [(gi, L, init) for gi in 1:ng for L in nlayers_list for init in inits
             if !((init == :neel && ratios[gi] > 0.535) || (init == :stripe && ratios[gi] <= 0.535))]

    println("\nTotal VQE tasks: $(length(tasks))  (neel: J2/J1 <= 0.535, stripe: J2/J1 > 0.535)")
    flush(stdout)

    Threads.@threads for ti in eachindex(tasks)
        gi, L, init = tasks[ti]
        ratio   = ratios[gi]
        J2      = J1 * ratio
        E_dmrg  = E_dmrg_vec[gi]
        circuit = circuits[L]
        H       = j1j2_2d_Hamiltonian(Lx, Ly; J1=J1, J2=J2, periodic=periodic)

        overlapfn = init == :neel ?
            psum -> overlap_with_neel_2d(psum, Lx, Ly) :
            psum -> overlap_with_stripe_2d(psum, Lx, Ly)

        println("  [tid=$(Threads.threadid())] J2/J1=$ratio  L=$L  init=$init  -> VQE")
        flush(stdout)

        rng     = MersenneTwister(hash((gi, L, init)))
        thetas0 = randn(rng, N_SHARED_2D * L) .* scale

        f = th -> expect_fast_2d(circuit, H, th, n, L, n_nn, n_nnn;
                                 min_abs_coeff=min_abs_coeff, overlapwith=overlapfn)

        ckpt_path = joinpath(outdir, "ckpt_$(Lx)x$(Ly)_L$(L)_$(init)_r$(ratio).jls")

        function mid_checkpoint(theta_cur, iter_cur, f_cur, iters_cur, energies_cur)
            serialize(ckpt_path, (
                ratio         = float(ratio),
                E_dmrg        = E_dmrg,
                E_opt         = NaN,       # not yet evaluated
                S_opt         = NaN,
                err           = NaN,
                psum_len      = 0,
                conv_iters    = copy(iters_cur),
                conv_energies = copy(energies_cur),
                thetas_opt    = copy(theta_cur),
                iter_saved    = iter_cur,
                f_approx      = f_cur,
                completed     = false,
            ))
        end

        thetas_opt, conv_iters, conv_energies = optimize_spsa_adam_tracked(
            f, thetas0; maxiter=maxiter, eta=eta, delta=delta_spsa, show_every=100,
            checkpoint_fn=mid_checkpoint)

        psum_opt = propagate_obs_2d(circuit, H, thetas_opt, n, L, n_nn, n_nnn;
                                    min_abs_coeff=min_abs_coeff)
        E_opt    = overlapfn(psum_opt)
        S_opt    = stabilizer_renyi_entropy(psum_opt; alpha=2.0)
        err      = relerr(E_opt, E_dmrg)
        psum_len = length(psum_opt)

        println("  [tid=$(Threads.threadid())] J2/J1=$ratio  L=$L  init=$init" *
                "  E=$(round(E_opt,digits=5))  err=$(round(err,sigdigits=3))  |psum|=$psum_len")
        flush(stdout)

        results[(init, L)][gi] = (
            ratio         = float(ratio),
            E_dmrg        = E_dmrg,
            E_opt         = E_opt,
            S_opt         = S_opt,
            err           = err,
            psum_len      = psum_len,
            conv_iters    = conv_iters,
            conv_energies = conv_energies,
            thetas_opt    = thetas_opt,
        )

        serialize(ckpt_path, merge(results[(init, L)][gi], (completed=true,)))
    end

    return Dict(k => filter(!isnothing, results[k]) for k in result_keys)
end

############################################################
# ---- Error & OSE plots ----
############################################################

function make_plots_2d(results, nlayers_list, Lx, Ly, outdir)
    mkpath(outdir)
    colors  = Dict(:neel => :blue, :stripe => :green)
    lstyles = Dict(:neel => :solid, :stripe => :dot)
    mshapes = Dict(:neel => :circle, :stripe => :square)

    p1 = plot(xlabel="J2/J1", ylabel="dE/|E_DMRG|",
              title="$(Lx)x$(Ly) PBC J1-J2  (L=5, dc=3e-4)",
              titlefontsize=12, yaxis=:log,
              left_margin=20Plots.mm, bottom_margin=15Plots.mm,
              right_margin=5Plots.mm, legend=:bottomright)
    p2 = plot(xlabel="J2/J1", ylabel="OSE S2",
              title="OSE -- $(Lx)x$(Ly) PBC  (L=5, dc=3e-4)",
              titlefontsize=12,
              left_margin=15Plots.mm, bottom_margin=15Plots.mm,
              right_margin=5Plots.mm, legend=:topright)

    for L in sort(nlayers_list)
        for init in [:neel, :stripe]
            key = (init, L)
            haskey(results, key) || continue
            rs  = sort(results[key]; by=r->r.ratio)
            isempty(rs) && continue
            xs  = [r.ratio for r in rs]
            lbl = "$(init) L=$L"
            plot!(p1, xs, [r.err   for r in rs];
                  color=colors[init], linestyle=lstyles[init],
                  marker=mshapes[init], label=lbl, markersize=5, linewidth=2)
            plot!(p2, xs, [r.S_opt for r in rs];
                  color=colors[init], linestyle=lstyles[init],
                  marker=mshapes[init], label=lbl, markersize=5, linewidth=2)
        end
    end

    for p in (p1, p2)
        vline!(p, [0.535]; color=:gray,     linestyle=:dash, label="Neel->VBS (0.535)")
        vline!(p, [0.610]; color=:darkgray, linestyle=:dash, label="VBS->Stripe (0.610)")
    end

    panel = plot(p1, p2; layout=(1,2), size=(1300, 550), top_margin=10Plots.mm)
    outpath = joinpath(outdir, "fig_2d_$(Lx)x$(Ly)_3e4_pbc_phased.pdf")
    savefig(panel, outpath)
    println("Saved -> $outpath")
    return panel
end

############################################################
# ---- Entry point ----
############################################################

Lx, Ly = 6, 6
outdir  = joinpath(@__DIR__, "fig_j1j2_out", "2d_$(Lx)x$(Ly)_3e4_pbc_phased")
mkpath(outdir)

results = run_2d_sweep_threaded(;
    Lx            = Lx,
    Ly            = Ly,
    j2j1_list     = [0.1, 0.3, 0.5, 0.535,
                     0.55, 0.58, 0.61, 0.65,
                     0.7, 1.0],
    nlayers_list  = [5],
    J1            = 1.0,
    scale         = 0.05,
    min_abs_coeff = 3e-4,
    maxiter       = 3000,
    eta           = 0.001,
    delta_spsa    = 0.005,
    periodic      = true,
    outdir        = outdir,
)

serialize(joinpath(outdir, "results_2d_$(Lx)x$(Ly)_3e4_pbc_phased.jls"), results)
println("Results serialized -> $outdir")

make_plots_2d(results, [5], Lx, Ly, outdir)
println("\nAll done.")
