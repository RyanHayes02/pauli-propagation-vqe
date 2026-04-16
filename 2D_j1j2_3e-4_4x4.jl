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

# Snake (boustrophedon) ordering for a Lx x Ly lattice.
# Minimises the DMRG entanglement cut by keeping rows local.
# Row 1: left->right,  Row 2: right->left,  Row 3: left->right, ...
#
#   (1,1) (1,2) (1,3) (1,4)      1   2   3   4
#   (2,1) (2,2) (2,3) (2,4)  ->  8   7   6   5
#   (3,1) (3,2) (3,3) (3,4)      9  10  11  12
#   (4,1) (4,2) (4,3) (4,4)     16  15  14  13
#
function snake_index(row::Int, col::Int, Lx::Int, Ly::Int)
    if isodd(row)
        return (row - 1) * Ly + col
    else
        return (row - 1) * Ly + (Ly - col + 1)
    end
end

# All NN bonds [(s1,s2),...] in snake-index space
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

# All NNN bonds (diagonals) in snake-index space
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

# 2D J1-J2 Heisenberg Hamiltonian (snake-ordered sites)
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
# ---- Ansatz circuit for 2D                             ----
# Per layer: Rz(a) x n | Rx(b) x n |                   ----
#            RZZ(g) x |NN| | RXX(d) x |NN| | RYY(e) x |NN| ----
############################################################

const N_SHARED_2D = 5   # (a, b, g, d, e) per layer

function j1j2_2d_circuit(Lx::Int, Ly::Int; nlayers::Int=5, periodic::Bool=false)
    n       = Lx * Ly
    nn      = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn     = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    all_bonds = vcat(nn, nnn)
    circuit = PauliRotation[]
    for _ in 1:nlayers
        for i in 1:n
            push!(circuit, PauliRotation(:Z, i))
        end
        for i in 1:n
            push!(circuit, PauliRotation(:X, i))
        end
        for (i, j) in all_bonds
            push!(circuit, PauliRotation([:Z, :Z], [i, j]))
        end
        for (i, j) in all_bonds
            push!(circuit, PauliRotation([:X, :X], [i, j]))
        end
        for (i, j) in all_bonds
            push!(circuit, PauliRotation([:Y, :Y], [i, j]))
        end
    end
    return circuit
end

function expand_shared_thetas_2d(thetas_shared, n::Int, nlayers::Int, n_bonds::Int)  # n_bonds = |NN| + |NNN|
    thetas_full = Vector{Float64}(undef, nlayers * (2n + 3n_bonds))
    idx = 1
    for k in 1:nlayers
        a = thetas_shared[N_SHARED_2D*(k-1) + 1]
        b = thetas_shared[N_SHARED_2D*(k-1) + 2]
        g = thetas_shared[N_SHARED_2D*(k-1) + 3]
        d = thetas_shared[N_SHARED_2D*(k-1) + 4]
        e = thetas_shared[N_SHARED_2D*(k-1) + 5]
        for _ in 1:n;      thetas_full[idx] = a; idx += 1; end
        for _ in 1:n;      thetas_full[idx] = b; idx += 1; end
        for _ in 1:n_bonds; thetas_full[idx] = g; idx += 1; end
        for _ in 1:n_bonds; thetas_full[idx] = d; idx += 1; end
        for _ in 1:n_bonds; thetas_full[idx] = e; idx += 1; end
    end
    return thetas_full
end

function propagate_obs_2d(circuit, H, thetas_shared, n, nlayers, n_bonds;
                           min_abs_coeff=1e-4)
    thetas_full = expand_shared_thetas_2d(thetas_shared, n, nlayers, n_bonds)
    obs = copy_paulisum(H)
    propagate!(circuit, obs, thetas_full; min_abs_coeff=min_abs_coeff)
    return obs
end

############################################################
# ---- Reference-state overlaps (Clifford circuits)     ----
############################################################

# Neel: checkerboard |up> on (r+c even), |down> on (r+c odd)
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

# Stripe AFM: alternating ferromagnetic rows (all-up / all-down)
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

# Columnar/horizontal dimer state: horizontal singlets tiled in 2x2 blocks
# Each block at top-left (r,c): pairs (r,c)-(r,c+1) and (r+1,c)-(r+1,c+1)
# Note: this is a product of horizontal dimers, NOT a symmetric plaquette resonance
# Singlet prep: X->H on site 1, X on site 2, CNOT(1->2)
function overlap_with_columnar_dimer_2d(psum::PauliSum, Lx::Int, Ly::Int)
    circ = CliffordGate[]
    for r in 1:2:Lx
        for c in 1:2:Ly
            # top pair
            if c + 1 <= Ly
                s1 = snake_index(r, c,   Lx, Ly)
                s2 = snake_index(r, c+1, Lx, Ly)
                push!(circ, CliffordGate(:X,    s1))
                push!(circ, CliffordGate(:H,    s1))
                push!(circ, CliffordGate(:X,    s2))
                push!(circ, CliffordGate(:CNOT, [s1, s2]))
            end
            # bottom pair
            if r + 1 <= Lx && c + 1 <= Ly
                s3 = snake_index(r+1, c,   Lx, Ly)
                s4 = snake_index(r+1, c+1, Lx, Ly)
                push!(circ, CliffordGate(:X,    s3))
                push!(circ, CliffordGate(:H,    s3))
                push!(circ, CliffordGate(:X,    s4))
                push!(circ, CliffordGate(:CNOT, [s3, s4]))
            end
        end
    end
    rotated = propagate(circ, psum, []; min_abs_coeff=1e-12)
    return overlapwithzero(rotated)
end

function expect_fast_2d(circuit, H, thetas_shared, n, nlayers, n_bonds;
                        min_abs_coeff=1e-4, overlapwith)
    psum = propagate_obs_2d(circuit, H, thetas_shared, n, nlayers, n_bonds;
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
# ---- DMRG with snake ordering                         ----
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
    sites = siteinds("Qubit", n)   # standard Pauli X/Y/Z, consistent with PauliPropagation
    os    = OpSum()

    nn  = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn = nnn_bonds_2d(Lx, Ly; periodic=periodic)

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
# ---- SPSA + ADAM ----
############################################################

function optimize_spsa_adam(f, thetas0;
                             maxiter=5000,
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
            println("      [tid=$(Threads.threadid())] iter $i  f~$(round(f_last, digits=6))")
            flush(stdout)
        end
    end
    return theta, f_last
end

############################################################
# ---- Main sweep ----
############################################################

relerr(Evqe, Edmrg) = abs(Evqe - Edmrg) / abs(Edmrg)

function run_2d_sweep_threaded(;
        Lx             = 4,
        Ly             = 4,
        j2j1_list      = [0.1, 0.3, 0.5, 0.535,
                          0.55, 0.58, 0.61, 0.65,
                          0.7, 1.0],
        nlayers_list   = [5],
        J1             = 1.0,
        scale          = 0.05,
        min_abs_coeff  = 1e-4,   # override at call site for different runs
        maxiter        = 5000,
        eta            = 0.001,
        delta_spsa     = 0.005,
        periodic       = false,
        outdir         = ".")

    n       = Lx * Ly
    nn      = nn_bonds_2d(Lx, Ly;  periodic=periodic)
    nnn     = nnn_bonds_2d(Lx, Ly; periodic=periodic)
    n_bonds = length(nn) + length(nnn)

    println("\n=== 2D J1-J2 sweep  $(Lx)x$(Ly)=$(n) qubits  threads=$(Threads.nthreads()) ===")
    println("NN bonds: $(length(nn))   NNN bonds: $(length(nnn))   total circuit bonds: $n_bonds")
    flush(stdout)

    circuits = Dict(L => j1j2_2d_circuit(Lx, Ly; nlayers=L, periodic=periodic)
                    for L in nlayers_list)

    ratios  = collect(j2j1_list)
    ng      = length(ratios)

    inits       = [:neel, :columnar_dimer, :stripe]
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

    println("\nDMRG max bond dimensions summary:")
    for (gi, r) in enumerate(ratios)
        println("  J2/J1=$r  max_bd=$(BD_dmrg_vec[gi])")
    end
    flush(stdout)

    # ---- VQE (threaded) ----
    tasks = [(gi, L, init) for gi in 1:ng for L in nlayers_list for init in inits]
    println("\nTotal VQE tasks: $(length(tasks))  " *
            "($(ng) ratios x $(length(nlayers_list)) layers x $(length(inits)) inits)")
    flush(stdout)

    Threads.@threads for ti in eachindex(tasks)
        gi, L, init = tasks[ti]
        ratio   = ratios[gi]
        J2      = J1 * ratio
        E_dmrg  = E_dmrg_vec[gi]
        circuit = circuits[L]
        H       = j1j2_2d_Hamiltonian(Lx, Ly; J1=J1, J2=J2, periodic=periodic)

        overlapfn = if init == :neel
            psum -> overlap_with_neel_2d(psum, Lx, Ly)
        elseif init == :columnar_dimer
            psum -> overlap_with_columnar_dimer_2d(psum, Lx, Ly)
        else
            psum -> overlap_with_stripe_2d(psum, Lx, Ly)
        end

        println("  [tid=$(Threads.threadid())] J2/J1=$ratio  L=$L  init=$init  -> VQE")
        flush(stdout)

        rng     = MersenneTwister(hash((gi, L, init)))
        thetas0 = randn(rng, N_SHARED_2D * L) .* scale

        f = th -> expect_fast_2d(circuit, H, th, n, L, n_bonds;
                                 min_abs_coeff=min_abs_coeff, overlapwith=overlapfn)

        thetas_opt, _ = optimize_spsa_adam(f, thetas0;
                                           maxiter=maxiter, eta=eta,
                                           delta=delta_spsa, show_every=500)

        psum_opt = propagate_obs_2d(circuit, H, thetas_opt, n, L, n_bonds;
                                    min_abs_coeff=min_abs_coeff)
        E_opt      = overlapfn(psum_opt)
        S_opt      = stabilizer_renyi_entropy(psum_opt; alpha=2.0)
        err        = relerr(E_opt, E_dmrg)
        psum_len   = length(psum_opt)

        println("  [tid=$(Threads.threadid())] J2/J1=$ratio  L=$L  init=$init" *
                "  E=$(round(E_opt,digits=5))  err=$(round(err,sigdigits=3))  |psum|=$psum_len")
        flush(stdout)

        results[(init, L)][gi] = (
            ratio    = float(ratio),
            E_dmrg   = E_dmrg,
            E_opt    = E_opt,
            S_opt    = S_opt,
            err      = err,
            psum_len = psum_len,
        )

        serialize(
            joinpath(outdir, "ckpt_$(Lx)x$(Ly)_L$(L)_$(init)_r$(ratio).jls"),
            results[(init, L)][gi]
        )
    end

    return Dict(k => filter(!isnothing, results[k]) for k in result_keys)
end

############################################################
# ---- Plotting ----
############################################################

function make_plots_2d(results, nlayers_list, Lx, Ly, outdir)
    mkpath(outdir)

    colors  = Dict(5 => :blue, 10 => :red)
    lstyles = Dict(:neel => :solid, :columnar_dimer => :dash, :stripe => :dot)
    mshapes = Dict(:neel => :circle, :columnar_dimer => :diamond, :stripe => :square)

    p1 = plot(xlabel="J2/J1", ylabel="dE/|E_DMRG|",
              title="Energy error -- $(Lx)x$(Ly) J1-J2, dc=1e-5",
              left_margin=20Plots.mm, bottom_margin=15Plots.mm, right_margin=5Plots.mm)

    p2 = plot(xlabel="J2/J1", ylabel="OSE S2",
              title="OSE -- $(Lx)x$(Ly) J1-J2, dc=1e-5",
              left_margin=15Plots.mm, bottom_margin=15Plots.mm, right_margin=5Plots.mm)

    for L in sort(nlayers_list)
        for init in [:neel, :columnar_dimer, :stripe]
            key = (init, L)
            haskey(results, key) || continue
            rs  = sort(results[key]; by=r->r.ratio)
            isempty(rs) && continue
            xs  = [r.ratio for r in rs]
            lbl = "$(init) L=$L"
            plot!(p1, xs, [r.err   for r in rs];
                  color=colors[L], linestyle=lstyles[init],
                  marker=mshapes[init], label=lbl, markersize=5)
            plot!(p2, xs, [r.S_opt for r in rs];
                  color=colors[L], linestyle=lstyles[init],
                  marker=mshapes[init], label=lbl, markersize=5)
        end
    end

    for p in (p1, p2)
        vline!(p, [0.535]; color=:blue, linestyle=:dash, label="Neel->VBS (0.535)")
        vline!(p, [0.610]; color=:red,  linestyle=:dash, label="VBS->Stripe (0.610)")
    end

    panel = plot(p1, p2; layout=(1,2), size=(1300,550),
                 plot_title="2D J1-J2 Heisenberg $(Lx)x$(Ly) OBC -- dc=1e-5",
                 top_margin=10Plots.mm)

    outpath = joinpath(outdir, "fig_2d_$(Lx)x$(Ly)_j1j2_1e5.pdf")
    savefig(panel, outpath)
    println("Saved -> $outpath")
    return panel
end

############################################################
# ---- Entry point ----
############################################################

Lx, Ly = 4, 4
outdir  = joinpath(@__DIR__, "fig_j1j2_out", "2d_$(Lx)x$(Ly)_1e5")
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
    min_abs_coeff = 1e-5,
    maxiter       = 5000,
    eta           = 0.001,
    delta_spsa    = 0.005,
    periodic      = false,
    outdir        = outdir,
)

serialize(joinpath(outdir, "results_2d_$(Lx)x$(Ly)_j1j2_1e5.jls"), results)
println("Results serialized -> $outdir")

make_plots_2d(results, [5], Lx, Ly, outdir)
println("\nAll done.")