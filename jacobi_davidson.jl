function jacobi_davidson{T <: AbstractLinearMap, Algorithm<:CorrectionSolver}(
  A::T, 
  solver::Algorithm, 
  krylov_dim::Int; 
  num::Int = 2, 
  expansions::Int = 5, 
  ɛ::Float64 = 1e-6)

  dim = krylov_dim + expansions
  n = size(A, 1)

  # Initialize space for `krylov_dim` Krylov basis vectors
  # and `expansions` additional basis vectors
  V = zeros(Complex{Float64}, n, dim + 1)
  H = zeros(Complex{Float64}, dim + 1, dim)
  V[:, 1] = rand(n)
  V[:, 1] /= norm(V[:, 1])

  # First build a Krylov subspace
  for k = 1 : krylov_dim
    # Add the (k + 1)st basis vector
    expand!(V, H, A * V[:, k], k)
  end

  W = A.lmap * V[:, 1 : krylov_dim]

  θ = 0.0 + 0.0im
  u = zeros(Complex{Float64}, n)

  # Then expand the subspace with approximate solutions to the correction equation
  for k = krylov_dim : dim - 1

    # Factorize the Hessenberg matrix into Schur form
    F = schurfact(H[1 : k, 1 : k])

    # Reorder the Schur form with the `num` largest eigs in front
    p = sortperm(abs(F[:values]), rev = true)
    Π = falses(k)
    Π[p[1 : num]] = true
    ordschur!(F, Π)

    # Take a Ritz pair (θ, y)
    θ = F[:values][1]
    y = F[:vectors][:, 1]

    # Approximate eigenpair (θ, u)
    u .= V[:, 1 : k] * y

    solver(A, θ, u)

    # Define the residual mapping
    R = LinearMap(x -> A * x - θ * x, nothing, n)
    
    # Projection Cⁿ → Cⁿ ∖ span {u}: Px = (I - uu')x
    P = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)
    
    # Coefficient matrix A - θI : Cⁿ ∖ span {u} -> Cⁿ ∖ span {u}
    C = P * R * P

    # Residual
    r = R * u

    @show norm(r)

    if norm(r) < ɛ
      return θ, u
    end

    # Let's solve the correction eqn exactly for now.
    # Ã = [(A.lmap - θ * speye(n)) u; u' 0]
    # rhs = [-r; 0]
    # v = (Ã \ rhs)[1 : n]

    # Solve the correction equation approximately
    v = gmres(C, -r)
    # expand!(V, H, v, k)

    # Orthogonalize
    for j = 1 : k
      v -= dot(v, V[:, j]) * V[:, j]
    end
    v /= norm(v)

    # Expand
    V[:, k + 1] = v
    H[k + 1, 1 : k] = v' * W
    w = A * v
    W = [W w]
    H[1 : k + 1, k + 1] = V[:, 1 : k + 1]' * w

    @show H[1 : k + 1, 1 : k + 1]
  end

  θ, u
end

function expand!(V::AbstractMatrix, H::AbstractMatrix, w::AbstractVector, k::Int)

  # Orthogonalize using Gramm-Schmidt
  for j = 1 : k
    H[j, k] = dot(w, V[:, j])
    w -= H[j, k] * V[:, j]
  end

  H[k + 1, k] = norm(w)
  V[:, k + 1] = w / H[k + 1, k]
end
