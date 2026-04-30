using Pkg
Pkg.activate(@__DIR__)

using PauliPropagation
using LinearAlgebra
using Base.Threads
using Serialization
using Plots

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
# ---- PauliSum helper ----
############################################################

function copy_paulisum(H::PauliSum)
    Hcopy = PauliSum(H.nqubits)
    for (ps, coeff) in H
        set!(Hcopy, ps, coeff)
    end
    return Hcopy
end

############################################################
# ---- Hamiltonian ----
############################################################

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
# ---- Circuit ----
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
        a_nn  = thetas_shared[N_SHARED_2D*(k-1) + 1]
        b_nn  = thetas_shared[N_SHARED_2D*(k-1) + 2]
        g_nn  = thetas_shared[N_SHARED_2D*(k-1) + 3]
        a_nnn = thetas_shared[N_SHARED_2D*(k-1) + 4]
        b_nnn = thetas_shared[N_SHARED_2D*(k-1) + 5]
        g_nnn = thetas_shared[N_SHARED_2D*(k-1) + 6]
        for _ in 1:n_nn;  thetas_full[idx] = a_nn;  idx += 1; end
        for _ in 1:n_nn;  thetas_full[idx] = b_nn;  idx += 1; end
        for _ in 1:n_nn;  thetas_full[idx] = g_nn;  idx += 1; end
        for _ in 1:n_nnn; thetas_full[idx] = a_nnn; idx += 1; end
        for _ in 1:n_nnn; thetas_full[idx] = b_nnn; idx += 1; end
        for _ in 1:n_nnn; thetas_full[idx] = g_nnn; idx += 1; end
    end
    return thetas_full
end

############################################################
# ---- Reference-state overlaps ----
############################################################

function overlap_with_neel_2d(psum, Lx::Int, Ly::Int)
    circ = CliffordGate[]
    for r in 1:Lx, c in 1:Ly
        if iseven(r + c)
            push!(circ, CliffordGate(:X, snake_index(r, c, Lx, Ly)))
        end
    end
    rotated = propagate(circ, psum, []; min_abs_coeff=1e-12)
    return overlapwithzero(rotated)
end

function overlap_with_stripe_2d(psum, Lx::Int, Ly::Int)
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
# ---- Entropy ----
############################################################

function normalize_paulisum(psum)
    frob = norm(psum, 2)
    out  = PauliSum(psum.nqubits)
    for (ps, coeff) in psum
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

relerr(Evqe, Edmrg) = abs(Evqe - Edmrg) / abs(Edmrg)

############################################################
# ---- Re-evaluation sweep ----
############################################################

function run_reeval(;
        Lx            = 6,
        Ly            = 6,
        nlayers       = 5,
        J1            = 1.0,
        periodic      = true,
        reeval_cutoff = 1e-5,
        indir         = ".",
        outdir        = ".")

    n     = Lx * Ly
    nn    = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn   = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    n_nn  = length(nn)
    n_nnn = length(nnn)

    circuit = j1j2_2d_circuit_supervisor(Lx, Ly; nlayers=nlayers, periodic=periodic)

    # Load DMRG cache
    dmrg_path  = joinpath(indir, "dmrg_$(Lx)x$(Ly)_j1j2.jls")
    dmrg_cache = deserialize(dmrg_path)
    println("Loaded DMRG cache: $(length(dmrg_cache)) entries")

    # Find checkpoint files
    ckpt_files = filter(f -> startswith(f, "ckpt_") && endswith(f, ".jls") &&
                             !occursin("reeval", f), readdir(indir))
    println("Found $(length(ckpt_files)) checkpoint files")
    flush(stdout)

    mkpath(outdir)

    all_results = []

    # Process sequentially — each uses all available threads internally
    for fname in sort(ckpt_files)
        fpath = joinpath(indir, fname)
        d = deserialize(fpath)

        if !haskey(d, :thetas_opt)
            println("  SKIP $fname — no thetas_opt saved")
            continue
        end

        ratio = d.ratio
        J2    = J1 * ratio

        init = if occursin("_neel_", fname)
            :neel
        elseif occursin("_stripe_", fname)
            :stripe
        else
            println("  SKIP $fname — unrecognised init")
            continue
        end

        E_dmrg = haskey(dmrg_cache, float(ratio)) ? dmrg_cache[float(ratio)][1] : d.E_dmrg

        iter_reached = get(d, :iter_saved, "?")
        completed    = get(d, :completed, false)
        println("  Reeval $fname  iter=$iter_reached  completed=$completed  cutoff=$reeval_cutoff")
        flush(stdout)

        H = j1j2_2d_Hamiltonian(Lx, Ly; J1=J1, J2=J2, periodic=periodic)

        thetas_full = expand_shared_thetas_2d(d.thetas_opt, n, nlayers, n_nn, n_nnn)

        # Propagate at fine cutoff using VectorPauliSum for threading
        obs_vec = VectorPauliSum(H)
        propagate!(circuit, obs_vec, thetas_full; min_abs_coeff=reeval_cutoff)
        obs = PauliSum(obs_vec)

        overlapfn = init == :neel ?
            p -> overlap_with_neel_2d(p, Lx, Ly) :
            p -> overlap_with_stripe_2d(p, Lx, Ly)

        E_reeval   = overlapfn(obs)
        S_reeval   = stabilizer_renyi_entropy(obs; alpha=2.0)
        err_reeval = relerr(E_reeval, E_dmrg)
        psum_len   = length(obs)

        println("    iter_reached=$iter_reached  E_DMRG=$(round(E_dmrg,digits=5))" *
                "  E_reeval=$(round(E_reeval,digits=5))" *
                "  err=$(round(err_reeval*100,digits=3))%  |psum|=$psum_len")
        flush(stdout)

        result = (
            ratio         = ratio,
            init          = init,
            E_dmrg        = E_dmrg,
            E_opt_orig    = get(d, :E_opt, NaN),
            err_orig      = get(d, :err, NaN),
            E_reeval      = E_reeval,
            S_reeval      = S_reeval,
            err_reeval    = err_reeval,
            psum_len      = psum_len,
            reeval_cutoff = reeval_cutoff,
            iter_reached  = iter_reached,
            completed     = completed,
        )

        # Also evaluate at original truncation 3e-4 using VectorPauliSum
        obs_orig_vec = VectorPauliSum(H)
        propagate!(circuit, obs_orig_vec, thetas_full; min_abs_coeff=3e-4)
        obs_orig = PauliSum(obs_orig_vec)
        E_orig    = overlapfn(obs_orig)
        err_orig_eval = relerr(E_orig, E_dmrg)

        println("    3e-4: E=$(round(E_orig,digits=5))  err=$(round(err_orig_eval*100,digits=3))%  " *
                "1e-5: E=$(round(E_reeval,digits=5))  err=$(round(err_reeval*100,digits=3))%")
        flush(stdout)

        result = (
            ratio          = ratio,
            init           = init,
            E_dmrg         = E_dmrg,
            E_orig         = E_orig,
            err_orig       = err_orig_eval,
            E_reeval       = E_reeval,
            S_reeval       = S_reeval,
            err_reeval     = err_reeval,
            psum_len       = psum_len,
            reeval_cutoff  = reeval_cutoff,
            iter_reached   = iter_reached,
            completed      = completed,
        )

        push!(all_results, result)

        outname = replace(fname, ".jls" => "_reeval1e-5.jls")
        serialize(joinpath(outdir, outname), result)
        println("    Saved -> $outname")
        flush(stdout)
    end

    println("\nRe-evaluation done. Results in $outdir")
    return all_results
end

############################################################
# ---- Entry point ----
############################################################

Lx, Ly = 6, 6
indir   = joinpath(@__DIR__, "fig_j1j2_out", "2d_$(Lx)x$(Ly)_3e4_pbc_phased")
outdir  = joinpath(@__DIR__, "fig_j1j2_out", "2d_$(Lx)x$(Ly)_reeval_1e5")
mkpath(outdir)

all_results = run_reeval(;
    Lx            = Lx,
    Ly            = Ly,
    nlayers       = 5,
    J1            = 1.0,
    periodic      = true,
    reeval_cutoff = 1e-5,
    indir         = indir,
    outdir        = outdir,
)

############################################################
# ---- Plot ----
############################################################

dmrg_vals = Dict(
    0.1   => -91.52547,
    0.3   => -80.53558,
    0.5   => -72.0283,
    0.535 => -71.16664,
    0.55  => -70.90622,
    0.58  => -70.67234,
    0.61  => -71.11494,
    0.65  => -72.91728,
    0.7   => -76.24289,
    1.0   => -101.89624,
)

# Separate into neel and stripe, sort by ratio
neel_res   = sort(filter(r -> r.init == :neel,   all_results), by=r->r.ratio)
stripe_res = sort(filter(r -> r.init == :stripe, all_results), by=r->r.ratio)

function signed_err(E, E_dmrg)
    # positive = above DMRG (physical), negative = below DMRG (SPSA noise)
    return (E - E_dmrg) / abs(E_dmrg)
end

p1 = plot(xlabel="J2/J1", ylabel="(E_VQE - E_DMRG) / |E_DMRG|",
          title="6x6 PBC J1-J2  (L=5) — 3e-4 vs 1e-5 truncation",
          titlefontsize=11,
          left_margin=20Plots.mm, bottom_margin=15Plots.mm,
          right_margin=10Plots.mm, top_margin=5Plots.mm,
          legend=:topright, size=(900, 550))

# Horizontal line at 0 (= DMRG energy)
hline!(p1, [0.0]; color=:black, linestyle=:dash, linewidth=1.5, label="DMRG (E_ref)")

# Neel — 3e-4
if !isempty(neel_res)
    plot!(p1, [r.ratio for r in neel_res], [signed_err(r.E_orig, dmrg_vals[r.ratio]) for r in neel_res];
          marker=:circle, color=:blue, linestyle=:dot, linewidth=1.5,
          markersize=5, alpha=0.7, label="Neel dc=3e-4")
    plot!(p1, [r.ratio for r in neel_res], [signed_err(r.E_reeval, dmrg_vals[r.ratio]) for r in neel_res];
          marker=:circle, color=:blue, linestyle=:solid, linewidth=2,
          markersize=6, label="Neel dc=1e-5")
end

# Stripe — 3e-4
if !isempty(stripe_res)
    plot!(p1, [r.ratio for r in stripe_res], [signed_err(r.E_orig, dmrg_vals[r.ratio]) for r in stripe_res];
          marker=:square, color=:green, linestyle=:dot, linewidth=1.5,
          markersize=5, alpha=0.7, label="Stripe dc=3e-4")
    plot!(p1, [r.ratio for r in stripe_res], [signed_err(r.E_reeval, dmrg_vals[r.ratio]) for r in stripe_res];
          marker=:square, color=:green, linestyle=:solid, linewidth=2,
          markersize=6, label="Stripe dc=1e-5")
end

vline!(p1, [0.535]; color=:gray,     linestyle=:dash, label="Neel->VBS (0.535)")
vline!(p1, [0.610]; color=:darkgray, linestyle=:dash, label="VBS->Stripe (0.610)")

outpath = joinpath(outdir, "fig_6x6_reeval_1e5_comparison.pdf")
savefig(p1, outpath)
println("Plot saved -> $outpath")

println("\nAll done.")
