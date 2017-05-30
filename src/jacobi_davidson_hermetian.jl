function jacobi_davidson_hermetian{T <: AbstractLinearMap, Alg <: CorrectionSolver, Target <: Target}(
  A::T,                    # Some square Hermetian matrix
  solver::Alg;             # Solver for the correction equation
  pairs::Int = 2,          # Number of eigenpairs wanted
  max_dimension::Int = 50, # Maximal search space size
  min_dimension::Int = 10, # Minimal search space size
  target::Target = LM(),   # Search target
  ɛ::Float64 = 1e-9        # Maximum residual norm
)

  residuals::Vector{Float64} = []

  n = size(A, 1)

  # V's vectors span the search space
  V::Vector{Vector{Float64}} = [rand(n)]

  # W will just hold the matrix A * V
  W::Vector{Vector{Float64}} = []

  # And M will be the Galerkin approximation of A, i.e. M = V' * A * V = V' * W
  M = zeros(max_dimension, max_dimension)

  # k is the number of converged eigenpairs
  k = 0

  # m is the current dimension of the search subspace
  m = 0

  # X holds the converged / locked eigenvectors
  X = zeros(n, pairs)

  # D holds the converged / locked eigenvalues
  D = zeros(pairs)

  # Iterate until k eigenpairs are converged
  while k <= pairs

    # Orthogonalize V[m + 1] w.r.t. {V[1], …, V[m]} using modified Gramm-Schmidt
    for i = 1 : m
      V[m + 1] -= dot(V[m + 1], V[i]) * V[i]
    end

    # Normalize the new vector
    V[m + 1] /= norm(V[m + 1])

    # Increment the search subspace dimension
    m += 1

    # Add A * V[m]
    push!(W, A * V[m])

    # Update the upper triangular part of the Galerkin matrix
    for i = 1 : m
      M[i, m] = dot(V[i], W[m])
    end

    # Compute the Ritz values and pre-Ritz vectors; here we exploit symmetry
    # M S = S diagm(θs) with S orthonormal
    θs, S = eig(Symmetric(M[1 : m, 1 : m]))

    # TODO: Sort the decomposition via target.

    # Consider the first pre-Ritz vector
    # and lift it to Ritz vector (i.e. compute V * xs[:, 1])
    u = mv_product(V, S[:, 1], m)

    # Compute the residual
    # Two choices here: either compute A * u via a matrix-vector product
    # or use A * u = A * V * s1 = W * s1; whatever is least work.
    r = A * u - θs[1] * u

    push!(residuals, norm(r))

    # Find one (or more) converged eigenpairs
    while norm(r) ≤ ɛ

      # An eigenpair has converged
      k += 1

      # Store the approximate eigenpair
      D[k] = θs[1]
      X[:, k] = u

      # Have we found all pairs?
      if k == pairs
        return D, X, residuals
      end

      # Reduce dimension of search space as one vector is removed
      m -= 1

      # Remove directions of the converged eigenpair.
      # The idea is to define S_new = [s₂, …, sₘ], M_new = diagm([Θ₂, …, Θₘ])
      # and V_new = V*S_new (without the Ritz vector s₁), so that
      # V_new' * A * V_new = S_new' * V' * A * V * S_new = S_new' * M * S_new
      # = S_new' * S_new * M_new = M_new. Hence, M_new is the Galerkin projection
      # of A w.r.t. V_new and the converged eigenvector is not in the span of V_new

      V_new::Vector{Vector{Float64}} = []
      for i = 1 : m
        push!(V_new, mv_product(V, S[:, i + 1], m))
        W[i] = A * V_new[i]
      end

      shift!(θs)
      M[1 : m, 1 : m] = diagm(θs)
      V = V_new
      S = eye(m)

      u = V[1]
      r = A * u - θs[1] * u

      push!(residuals, norm(r))
    end

    @show k

    if m == max_dimension
      return D[1 : k], X[:, 1 : k], residuals
    end

    # Solve the correction equation
    push!(V, solve_deflated_correction(solver, A, θs[1], X[:, 1 : k], u))

    Q = [X[:, 1 : k] u]
    proj1 = V[end] - Q * (Q' * V[end])
    nxt = A * proj1 - θs[1] * proj1
    proj2 = nxt - Q * (Q' * nxt)

    @show norm(proj2 + r)
  end
end

function mv_product{T}(V::Vector{Vector{T}}, x::Vector{T}, m)
  u = V[1] * x[1]
  for i = 2 : m
    u += V[i] * x[i]
  end
  u
end
