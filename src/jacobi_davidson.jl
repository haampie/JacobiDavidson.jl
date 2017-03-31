function jacobi_davidson{T <: AbstractLinearMap, Alg<:CorrectionSolver, Target<:Target}(
  A::T, 
  solver::Alg,
  krylov_dim::Int; 
  # num::Int = 2,  # For now we'll return 1 eigenpair
  expansions::Int = 5, 
  target::Target = LM(),
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
    # TODO: use Givens rotations and do this on the fly.
    F = schurfact(H[1 : k, 1 : k])

    @show abs(H[1 : k, 1 : k]) .> 0

    # Reorder the Schur form
    permutation = schur_permutation(target, F)
    Π = falses(k)
    Π[permutation[1]] = true
    ordschur!(F, Π)

    # Take a Ritz pair (θ, y)
    θ = F[:values][1]
    y = F[:vectors][:, 1]

    # Approximate eigenpair (θ, u)
    u .= V[:, 1 : k] * y

    # Have we converged yet? TODO: can do without MV product
    if norm(A * u - θ * u) < ɛ
      return θ, u
    end

    # Solve the correction equation
    v = solve_correction(solver, A, θ, u)

    # Orthogonalize
    for j = 1 : k
      v -= dot(v, V[:, j]) * V[:, j]
    end
    v /= norm(v)

    # Expand (TODO: make this efficient)
    V[:, k + 1] = v
    H[k + 1, 1 : k] = v' * W
    w = A * v
    W = [W w]
    H[1 : k + 1, k + 1] = V[:, 1 : k + 1]' * w
  end

  θ, u
end

