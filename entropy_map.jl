using LightGraphs
using LinearAlgebra
using LambertW
using NPZ
using QuantumInformation

include("common_functions.jl")

function entropy_map(dims, samples=1, p0=42, eps=0.1)

    ret = zeros(20, length(dims), samples)
    arange = zeros(length(dims))
    brange = zeros(length(dims))
    for (k, n) in enumerate(dims)
        p = p0 * log(n)/n
        a = (1-p0)/lambertw((1-p0)/(exp(1)*p0), -1)
        b = (1-p0)/lambertw((1-p0)/(exp(1)*p0), 0)
        arange[k] = a
        brange[k] = b
        for i=1:samples
            g = erdos_renyi(n, p; is_directed=false)
            for (j, τ) in enumerate(range(1/b - eps, 1/a + eps, length=20))
                ret[j, k, i] = real(graph_entropy(τ, g))/log(n)
            end
        end
    end
    to_save = Dict("samples"=>samples, "results"=>ret, "arange"=>arange, "brange"=>brange, "eps"=>eps)
    NPZ.npzwrite("results/entropy_map_p0=$(p0).npz", to_save)
end

const dims = 10:10:2000
entropy_map([10,])
@time entropy_map(dims, 100, 42)
@time entropy_map(dims, 100, 21)
@time entropy_map(dims, 100, 10.5)