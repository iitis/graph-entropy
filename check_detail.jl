using LightGraphs
using LinearAlgebra
using LambertW
using NPZ
using QuantumInformation

include("common_functions.jl")


function check_cl(nrange, m, samples, matrix=laplacian_matrix)
    ret = zeros(length(nrange), samples)
    τ = 0.1
    for s=1:samples
        for i=1:length(nrange)
            n = nrange[i]
            ω = vec(NPZ.npzread("data/BA_m0-$(m)_ε-0.05/$(n).npz"))
            g = expected_degree_graph(ω)
            ret[i, s] = real(entropy_from_ev(τ, g, matrix))
        end
    end
    Dict("nrange"=>nrange, "result"=>ret, "samples"=>samples)
end

function check_ba(nrange, m, k, samples, matrix=laplacian_matrix)
    ret = zeros(length(nrange),  samples)
    τ = 0.1
    for s=1:samples
        for i=1:length(nrange)
            n = nrange[i]
            g = barabasi_albert(n, m, k; is_directed=false, complete=true)
            ret[i, s] = real(entropy_from_ev(τ, g, matrix))
        end
    end
    Dict("nrange"=>nrange, "result"=>ret, "samples"=>samples)
end

function check_er(nrange, p, samples, matrix=laplacian_matrix)
    ret = zeros(length(nrange),  samples)
    τ = 0.1
    for s=1:samples
        for i=1:length(nrange)
            n = nrange[i]
            g = erdos_renyi(n, p; is_directed=false)
            ret[i, s] = real(entropy_from_ev(τ, g, matrix))
        end
    end
    Dict("nrange"=>nrange, "result"=>ret, "samples"=>samples)
end

function check_ws(nrange, k, β, samples, matrix=laplacian_matrix)
    ret = zeros(length(nrange),  samples)
    τ = 0.1
    for s=1:samples
        for i=1:length(nrange)
            n = nrange[i]
            g = watts_strogatz(n, k, β; is_directed=false)
            ret[i, s] = real(entropy_from_ev(τ, g, matrix))
        end
    end
    Dict("nrange"=>nrange, "result"=>ret, "samples"=>samples)
end


const nrange = 20:20:2000
const samples = 30

println("----- CHECK CL -----")
@time r = check_cl(nrange, 2, samples, adjacency_matrix)
NPZ.npzwrite("results/detail/detail_check_cl_m=2_adjacency.npz", r)

@time r = check_cl(nrange, 2, samples, normalized_laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_cl_m=2_normalized_laplacian.npz", r)

@time r = check_cl(nrange, 2, samples, laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_cl_m=2_laplacian.npz", r)

println("----- CHECK BA 2 -----")
@time r = check_ba(nrange, 2, 2, samples, adjacency_matrix)
NPZ.npzwrite("results/detail/detail_check_ba_m=2_k=2_adjacency.npz", r)

@time r = check_ba(nrange, 2, 2, samples, laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_ba_m=2_k=2_laplacian.npz", r)

@time r = check_ba(nrange, 2, 2, samples, normalized_laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_ba_m=2_k=2_normalized_laplacian.npz", r)

println("----- CHECK BA 3 -----")
@time r = check_ba(nrange, 3, 3, samples, normalized_laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_ba_m=3_k=3.npz", r)

println("----- CHECK ER -----")
@time r = check_er(nrange, 0.4, samples, adjacency_matrix)
NPZ.npzwrite("results/detail/detail_check_er_p=0.4_adjacency.npz", r)

@time r = check_er(nrange, 0.4, samples, laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_er_p=0.4_laplacian.npz", r)

@time r = check_er(nrange, 0.4, samples, normalized_laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_er_p=0.4_normalized_laplacian.npz", r)

println("----- CHECK WS -----")
@time r = check_ws(nrange, 4, 0.6, samples, adjacency_matrix)
NPZ.npzwrite("results/detail/detail_check_ws_k=4_p=0.6_adjacency.npz", r)

@time r = check_ws(nrange, 4, 0.6, samples, laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_ws_k=4_p=0.6_laplacian.npz", r)

@time r = check_ws(nrange, 4, 0.6, samples, normalized_laplacian_matrix)
NPZ.npzwrite("results/detail/detail_check_ws_k=4_p=0.6_normalized_laplacian.npz", r)

println("----- RECHECK ADJ ------")
@time r = check_ws(nrange, 4, 0.6, samples, x->-adjacency_matrix(x))
NPZ.npzwrite("results/detail/detail_check_ws_k=4_p=0.6_adjacency_2.npz", r)

@time r = check_er(nrange, 0.4, samples, x->-adjacency_matrix(x))
NPZ.npzwrite("results/detail/detail_check_er_p=0.4_adjacency_2.npz", r)

@time r = check_ba(nrange, 2, 2, samples, x->-adjacency_matrix(x))
NPZ.npzwrite("results/detail/detail_check_ba_m=2_k=2_adjacency_2.npz", r)

@time r = check_cl(nrange, 2, samples, x->-adjacency_matrix(x))
NPZ.npzwrite("results/detail/detail_check_cl_m=2_adjacency_2.npz", r)

println("----- RECHECK SHIFT ------")


for p=0.0:0.2:1.0
    println(p)

    @time r = check_ws(nrange, 4, p, samples, normalized_laplacian_matrix)
    NPZ.npzwrite("results/shift/ws_k=4_p=$(p)_normalized_laplacian.npz", r)

    @time r = check_er(nrange, 0.4, samples, normalized_laplacian_matrix)
    NPZ.npzwrite("results/shift/er_p=$(p)_normalized_laplacian.npz", r)

    @time r = check_ws(nrange, 4, p, samples, laplacian_matrix)
    NPZ.npzwrite("results/shift/ws_k=4_p=$(p)_laplacian.npz", r)

    @time r = check_er(nrange, 0.4, samples, laplacian_matrix)
    NPZ.npzwrite("results/shift/er_p=$(p)_laplacian.npz", r)

    @time r = check_ws(nrange, 4, p, samples, adjacency_matrix)
    NPZ.npzwrite("results/shift/ws_k=4_p=$(p)_adjacency.npz", r)

    @time r = check_er(nrange, 0.4, samples, adjacency_matrix)
    NPZ.npzwrite("results/shift/er_p=$(p)_adjacency.npz", r)
end

for m=1:3
    println(m)

    @time r = check_ba(nrange, m, m, samples, normalized_laplacian_matrix)
    NPZ.npzwrite("results/shift/ba_m=$(m)_k=$(m)_normalized_laplacian.npz", r)

    @time r = check_cl(nrange, 2, samples, normalized_laplacian_matrix)
    NPZ.npzwrite("results/shift/cl_m=$(m)_normalized_laplacian.npz", r)

    @time r = check_ba(nrange, m, m, samples, laplacian_matrix)
    NPZ.npzwrite("results/shift/ba_m=$(m)_k=$(m)_laplacian.npz", r)

    @time r = check_cl(nrange, 2, samples, laplacian_matrix)
    NPZ.npzwrite("results/shift/cl_m=$(m)_laplacian.npz", r)
    
    @time r = check_ba(nrange, m, m, samples, adjacency_matrix)
    NPZ.npzwrite("results/shift/ba_m=$(m)_k=$(m)_adjacency.npz", r)

    @time r = check_cl(nrange, 2, samples, adjacency_matrix)
    NPZ.npzwrite("results/shift/cl_m=$(m)_adjacency.npz", r)
end