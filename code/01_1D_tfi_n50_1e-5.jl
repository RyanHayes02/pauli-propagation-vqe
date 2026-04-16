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
# ---- helpers (PauliSum, NOT VectorPauliSum) ----
############################################################

function copy_paulisum(H::PauliSum)
    Hcopy = PauliSum(H.nqubits)
    for (ps, coeff) in H
        set!(Hcopy, ps, coeff)
    end
    return Hcopy
end

function tfi_Hamiltonian(n::Int, J::Real=1.0, h::Real=1.0; periodic::Bool=false)
    H = PauliSum(n)
    for i in 1:n-1
        PauliPropagation.add!(H, [:Z,:Z], [i, i+1], -J)
    end
    for i in 1:n
        PauliPropagation.add!(H, :X, i, -h)
    end
    return H
end

function tfi_circuit_with_z_layer(n::Int; nlayers::Int=5, periodic::Bool=false)
    topology = bricklayertopology(n; periodic=periodic)
    circuit = PauliRotation[]
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
    end
    return circuit
end

function expand_shared_thetas(thetas_shared, n::Int, nlayers::Int, n_zz::Int)
    thetas_full = Vector{Float64}(undef, nlayers*(2n + n_zz))
    idx = 1
    for k in 1:nlayers
        α = thetas_shared[3*(k-1) + 1]
        β = thetas_shared[3*(k-1) + 2]
        γ = thetas_shared[3*(k-1) + 3]
        for _ in 1:n;    thetas_full[idx] = α; idx += 1; end
        for _ in 1:n;    thetas_full[idx] = β; idx += 1; end
        for _ in 1:n_zz; thetas_full[idx] = γ; idx += 1; end
    end
    return thetas_full
end

function expect_on_plus(circuit, H, thetas_shared, n, nlayers, n_zz; min_abs_coeff=1e-5)
    thetas_full = expand_shared_thetas(thetas_shared, n, nlayers, n_zz)
    obs = copy_paulisum(H)
    propagate!(circuit, obs, thetas_full; min_abs_coeff=min_abs_coeff)
    return overlapwithplus(obs)
end

function propagate_obs(circuit, H, thetas_shared, n, nlayers, n_zz; min_abs_coeff=1e-5)
    thetas_full = expand_shared_thetas(thetas_shared, n, nlayers, n_zz)
    obs = copy_paulisum(H)
    propagate!(circuit, obs, thetas_full; min_abs_coeff=min_abs_coeff)
    return obs
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

function operator_norm_diff(psum_a, psum_b)
    ca = Dict(ps => coeff for (ps, coeff) in psum_a)
    cb = Dict(ps => coeff for (ps, coeff) in psum_b)
    all_keys = union(keys(ca), keys(cb))
    return sqrt(sum(abs2(get(ca, k, 0.0) - get(cb, k, 0.0)) for k in all_keys))
end

############################################################
# ---- DMRG ----
############################################################

function dmrg_tfi_energy(n::Int; J::Real=1.0, h::Real=1.0, periodic::Bool=false,
                          nsweeps::Int=50,
                          maxdim::Vector{Int}=[10,20,50,100,200,400,600,800],
                          cutoff::Real=1e-10,
                          linkdims0::Int=10,
                          seed::Int=0)
    Random.seed!(seed)
    sites = siteinds("Qubit", n)
    os = OpSum()
    for i in 1:(n-1)
        os += -float(J), "Z", i, "Z", i+1
    end
    if periodic
        os += -float(J), "Z", n, "Z", 1
    end
    for i in 1:n
        os += -float(h), "X", i
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
        G = (C_plus - C_minus) / (2 * delta) .* Delta
        m = beta1 .* m .+ (1 - beta1) .* G
        v = beta2 .* v .+ (1 - beta2) .* G.^2
        m_hat = m ./ (1 - beta1^i)
        v_hat = v ./ (1 - beta2^i)
        theta = theta .- eta .* m_hat ./ (sqrt.(v_hat) .+ eps)
        f_last = (C_plus + C_minus) / 2

        if i % show_every == 0
            println("      [tid=$(Threads.threadid())] iter $i  f_approx = $(round(f_last, digits=6))")
            flush(stdout)
        end
    end

    return theta, f_last
end

############################################################
# ---- Main sweep (one thread per gx per L) ----
############################################################

relerr(Epp, Edmrg) = abs(Epp - Edmrg) / abs(Edmrg)

function run_gx_sweep_threaded(; n=50,
                                 gx_list=collect(0.6:0.1:1.6),
                                 nlayers_list=[5, 10],
                                 scale=0.05,
                                 min_abs_coeff=1e-5,
                                 maxiter=7500,
                                 eta=0.001,
                                 delta_spsa=0.005,
                                 periodic=false,
                                 outdir=".")

    println("\n=== PauliSum sweep n=$n  threads=$(Threads.nthreads()) ===")
    flush(stdout)

    topology = bricklayertopology(n; periodic=periodic)
    n_zz     = length(topology)

    circuits = Dict(L => tfi_circuit_with_z_layer(n; nlayers=L, periodic=periodic)
                    for L in nlayers_list)

    gx_vec  = collect(gx_list)
    ng      = length(gx_vec)
    results = Dict(L => Vector{Union{Nothing,NamedTuple}}(nothing, ng) for L in nlayers_list)

    # ---- DMRG: sequential ----
    println("Running DMRG for all gx values...")
    flush(stdout)
    E_dmrg_vec = Vector{Float64}(undef, ng)
    for (gi, gx) in enumerate(gx_vec)
        println("  DMRG gx=$gx")
        flush(stdout)
        E_dmrg_vec[gi] = dmrg_tfi_energy(n; h=gx, periodic=periodic)
    end
    println("DMRG done.")
    flush(stdout)

    # ---- VQE: one thread per (gx, L) pair — 11 gx values x 2 layers = 22 tasks ----
    tasks = [(gi, L) for gi in 1:ng for L in nlayers_list]
    println("Total tasks: $(length(tasks))  ($(ng) gx values × $(length(nlayers_list)) layer counts)")
    flush(stdout)

    Threads.@threads for ti in eachindex(tasks)
        gi, L   = tasks[ti]
        gx      = gx_vec[gi]
        E_dmrg  = E_dmrg_vec[gi]
        circuit = circuits[L]
        H       = tfi_Hamiltonian(n, 1.0, gx; periodic=periodic)

        println("  [tid=$(Threads.threadid())] gx=$gx  L=$L  starting VQE")
        flush(stdout)

        rng     = MersenneTwister(0)
        thetas0 = randn(rng, 3*L) .* scale

        f = θ -> expect_on_plus(circuit, H, θ, n, L, n_zz; min_abs_coeff=min_abs_coeff)

        thetas_opt, _ = optimize_spsa_adam(f, thetas0;
                                           maxiter=maxiter, eta=eta,
                                           delta=delta_spsa, show_every=500)

        psum_opt = propagate_obs(circuit, H, thetas_opt, n, L, n_zz; min_abs_coeff=min_abs_coeff)
        E_opt    = overlapwithplus(psum_opt)
        S_opt    = stabilizer_renyi_entropy(psum_opt; alpha=2.0)
        err_opt  = relerr(E_opt, E_dmrg)

        println("  [tid=$(Threads.threadid())] gx=$gx  L=$L  E_opt=$(round(E_opt,digits=5))  err=$(round(err_opt,sigdigits=3))")
        flush(stdout)

        results[L][gi] = (
            gx     = float(gx),
            E_dmrg = E_dmrg,
            E_opt  = E_opt,
            S_opt  = S_opt,
            err    = err_opt,
        )

        serialize(joinpath(outdir, "checkpoint_n$(n)_L$(L)_gx$(gx).jls"), results[L][gi])
    end

    results_clean = Dict(L => filter(!isnothing, results[L]) for L in nlayers_list)
    return results_clean
end

############################################################
# ---- Plotting ----
############################################################

function make_plots(results, nlayers_list, n, outdir)
    mkpath(outdir)
    colors = Dict(5 => :blue, 10 => :red)

    p1 = plot(xlabel="gₓ", ylabel="ΔE/|E_DMRG|",
              title="Energy error vs gₓ  (n=$n, PauliSum, δc=1e-5)",
              legend=:topright, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
    for L in sort(nlayers_list)
        rs  = sort(results[L]; by=r->r.gx)
        gxs = [r.gx  for r in rs]
        err = [r.err  for r in rs]
        plot!(p1, gxs, err; color=colors[L], marker=:circle, label="ℓ=$L")
    end

    p2 = plot(xlabel="gₓ", ylabel="OSE S₂",
              title="OSE vs gₓ  (n=$n, PauliSum, δc=1e-5)",
              legend=:topright, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
    for L in sort(nlayers_list)
        rs  = sort(results[L]; by=r->r.gx)
        gxs = [r.gx   for r in rs]
        Ss  = [r.S_opt for r in rs]
        plot!(p2, gxs, Ss; color=colors[L], marker=:circle, label="ℓ=$L")
    end

    panel = plot(p1, p2;
                 layout=(1,2),
                 size=(1000, 500),
                 plot_title="TFI n=$n OBC — PauliSum, δc=1e-5, seed=0, scale=0.05",
                 top_margin=10Plots.mm)

    outpath = joinpath(outdir, "fig_panel_n$(n)_paulisum_1e5.pdf")
    savefig(panel, outpath)
    println("Saved figure to $outpath")
    return panel
end

############################################################
# ---- RUN ----
############################################################

n      = 50
outdir = joinpath(@__DIR__, "fig_gx_out", "paulisum_n$(n)_1e5")
mkpath(outdir)

results = run_gx_sweep_threaded(;
    n            = n,
    gx_list      = 0.6:0.1:1.6,
    nlayers_list = [5, 10],
    scale        = 0.05,
    min_abs_coeff = 1e-5,
    maxiter      = 7500,
    eta          = 0.001,
    delta_spsa   = 0.005,
    periodic     = false,
    outdir       = outdir,
)

serialize(joinpath(outdir, "results_n$(n)_paulisum_1e5.jls"), results)
println("Results serialized to $outdir")

make_plots(results, [5, 10], n, outdir)
println("\nAll done.")
