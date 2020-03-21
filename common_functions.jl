function gibbs_state(τ, H)
    ρ = exp(-τ*Matrix{Float64}(H))
    ρ /= tr(ρ)
end

function normalized_laplacian_matrix(g)
    L = Matrix{Float64}(laplacian_matrix(g))
    d = diag(L)
    invsqrt = x-> x>0 ? 1/sqrt(x) : 0.
    Dinvsqrt = diagm(invsqrt.(d))
    Dinvsqrt * L * Dinvsqrt
end

function entropy(τ, H)
    ρ = gibbs_state(τ, H)
    vonneumann_entropy(ρ)
end

function graph_entropy(τ, g, matrix=laplacian_matrix)
    entropy(τ, matrix(g))
end

function graph_gibbs_state(τ, g, matrix=laplacian_matrix)
    gibbs_state(τ, matrix(g))
end

function entropy_from_ev(τ, g, matrix=laplacian_matrix)
    H = Matrix{Float64}(matrix(g))
    ev = eigvals(H)
    S1 = τ * sum(ev .* exp.(-τ .* ev)) / sum(exp.(-τ .* ev))
    S2 = log(sum(exp.(-τ .* ev)))
    S1 + S2
end