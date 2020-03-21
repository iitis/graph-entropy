using LightGraphs
using LinearAlgebra
using LambertW
using NPZ
using QuantumInformation
import LightGraphs: expected_degree_graph, watts_strogatz

include("common_functions.jl")

expected_degree_graph(n, ω) = expected_degree_graph(ω)
watts_strogatz(n, k, p; kwargs...) = watts_strogatz(Int(n), Int(k), p; kwargs...)

function check_model(τrange, samples, n, model, params, matrix=laplacian_matrix; kwargs...)
    l = size(params, 1)
    ret = zeros(length(τrange), l, samples)
    for k=1:samples
        for j=1:l
            p = params[j, :]
            println(p)
            for (i, τ) in enumerate(τrange)
                g = model(n, p...; kwargs...)
                ret[i, j, k] = real(entropy_from_ev(τ, g, matrix))
            end
        end
    end
    Dict("result"=>ret, "samples"=>samples, "params"=>params, "taurange"=>τrange, "n"=>n)
end

τrange = 10 .^ (range(-2, stop=4, length=100))
samples = 20
n = 200

@time r = check_model(τrange, samples, n, barabasi_albert, [1 1; 2 2; 3 3]; is_directed=false, complete=true)
NPZ.npzwrite("results/phase_transition/ba_normalized_laplacian.npz", r)

@time r = check_model(τrange, samples, n, erdos_renyi, [0.3; 0.6; 0.9]; is_directed=false)
NPZ.npzwrite("results/phase_transition/er_normalized_laplacian.npz", r)

@time r = check_model(τrange, samples, n, watts_strogatz, [ceil(2log(n)) 0.0; ceil(2log(n)) 0.1; ceil(2log(n)) 1.0]; is_directed=false)
NPZ.npzwrite("results/phase_transition/ws_normalized_laplacian.npz", r)


@time r = check_model(τrange, samples, n, barabasi_albert, [1 1; 2 2; 3 3], laplacian_matrix; is_directed=false, complete=true)
NPZ.npzwrite("results/phase_transition/ba_laplacian.npz", r)

@time r = check_model(τrange, samples, n, erdos_renyi, [0.3; 0.6; 0.9], laplacian_matrix; is_directed=false)
NPZ.npzwrite("results/phase_transition/er_laplacian.npz", r)

@time r = check_model(τrange, samples, n, watts_strogatz, [ceil(2log(n)) 0.0; ceil(2log(n)) 0.1; ceil(2log(n)) 1.0], laplacian_matrix; is_directed=false)
NPZ.npzwrite("results/phase_transition/ws_laplacian.npz", r)


@time r = check_model(τrange, samples, n, barabasi_albert, [1 1; 2 2; 3 3], x->-adjacency_matrix(x); is_directed=false, complete=true)
NPZ.npzwrite("results/phase_transition/ba_adjacency.npz", r)

@time r = check_model(τrange, samples, n, erdos_renyi, [0.3; 0.6; 0.9], x->-adjacency_matrix(x); is_directed=false)
NPZ.npzwrite("results/phase_transition/er_adjacency.npz", r)

@time r = check_model(τrange, samples, n, watts_strogatz, [ceil(2log(n)) 0.0; ceil(2log(n)) 0.1; ceil(2log(n)) 1.0], x->-adjacency_matrix(x); is_directed=false)
NPZ.npzwrite("results/phase_transition/ws_adjacency.npz", r)