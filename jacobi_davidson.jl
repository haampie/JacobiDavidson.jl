import LinearMaps: AbstractLinearMap, LinearMap

function jacobi_davidson{T <: AbstractLinearMap}(A::T, krylov_dim::Int; num::Int = 2, expansions::Int = 5)
  dim = krylov_dim + expansions
  n = size(A, 1)

  # Initialize space for `krylov_dim` Krylov basis vectors
  # and `expansions` additional basis vectors
  V = zeros(Complex{Float64}, n, dim + 1)
  H = zeros(Complex{Float64}, dim + 1, dim)
  V[:, 1] = rand(n)
  V[:, 1] /= norm(V[:, 1])

  # First build a Krylov subspace
  for k = 2 : krylov_dim
    # Add the (k + 1)st basis vector
    expand!(V, H, A * V[:, k - 1], k)
  end

  return V, H

  # @show norm(A * V[:, 1 : krylov_dim] * ones(krylov_dim) - V[:, 1 : krylov_dim + 1] * H[1 : krylov_dim + 1, 1 : krylov_dim] * ones(krylov_dim))

  # Then expand the subspace with approximate solutions to the correction equation
  for k = krylov_dim + 1 : dim

    # Factorize the Hessenberg matrix into Schur form
    F = schurfact(H[1 : k - 1, 1 : k - 1])

    # Reorder the Schur form with the `num` largest eigs in front
    p = sortperm(abs(F[:values]), rev = true)
    Π = falses(k)
    Π[p[1 : num]] = true
    ordschur!(F, Π)

    @show abs(F[:values])

    # Take a Ritz pair (θ, y)
    θ = F[:values][1]
    y = F[:vectors][:, 1]

    @show abs(θ)

    # Approximate eigenpair (θ, u)
    u = V[:, 1 : k - 1] * y

    # Define the residual mapping
    R = LinearMap(x -> A * x - θ * x, nothing, n)
    
    # Projection Cⁿ → Cⁿ ∖ span {u}: Px = (I - uu')x
    P = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)
    
    # Coefficient matrix (I - uu')(A - θI)(I - uu')
    C = P * R * P

    # Residual
    r = R * u

    # Solve the correction equation
    new_vec = gmres(C, -r)

    expand!(V, H, new_vec, k)
  end
end

function expand!(V::AbstractMatrix, H::AbstractMatrix, w::AbstractVector, k::Int)

  # Orthogonalize using Gramm-Schmidt
  for j = 1 : k - 1
    H[j, k - 1] = dot(w, V[:, j])
    w -= H[j, k - 1] * V[:, j]
  end

  H[k, k - 1] = norm(w)
  V[:, k] = w / H[k, k - 1]
end

function gmres{T <: AbstractLinearMap}(A::T, b::AbstractVector; max_iter::Int = 5)
  n = size(A, 1)
  β = norm(b)
  V = zeros(Complex{Float64}, n, max_iter + 1)
  H = zeros(Complex{Float64}, max_iter + 1, max_iter)
  V[:, 1] = b / complex(β)

  # history = Float64[]
  
  # Create a Krylov subspace of dimension max_iter
  for k = 2 : max_iter
    # Add the (k + 1)th basis vector
    expand!(V, H, A * V[:, k - 1], k)

    # Solve the system
    e₁ = zeros(k)
    e₁[1] = β
    y = H[1 : k, 1 : k - 1] \ e₁
    x = V[:, 1 : k - 1] * y
    # push!(history, norm(A * x - b))
  end

  # Solve the low-dimensional problem
  e₁ = zeros(max_iter + 1)
  e₁[1] = β
  y = H \ e₁

  # Project back to the large dimensional solution
  V[:, 1 : max_iter] * y
end

function some_well_conditioned_matrix(n::Int = 100)
  off_diag₁ = rand(n - 1)
  off_diag₂ = rand(n - 1)

  diags = (off_diag₁, linspace(3, 100, n), off_diag₂)
  spdiagm(diags, (-1, 0, 1))
end

function testing()
  n = 200
  A = some_well_conditioned_matrix(n)
  x = ones(n)
  b = A * x

  d = eigs(A)
  @show d[1]

  B = LinearMap(A)
  # gmres(B, b)  
  V, H = jacobi_davidson(B, 10)

  A, V, H
end