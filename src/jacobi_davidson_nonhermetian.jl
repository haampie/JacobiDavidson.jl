function jacobi_davidson_nonhermetian{Alg <: CorrectionSolver}(
  A,                       # Some square Hermetian matrix
  solver::Alg;             # Solver for the correction equation
  pairs::Int = 5,          # Number of eigenpairs wanted
  max_dimension::Int = 80, # Maximal search space size
  min_dimension::Int = 10, # Minimal search space size
  max_iter::Int = 200,
  target::Float64 = 100.0,   # Search target
  ɛ::Float64 = 1e-7,       # Maximum residual norm
  v0::Vector{Float64} = rand(size(A, 1))
)

  residuals::Vector{Float64} = []

  n = size(A, 1)

  # V's vectors span the search space
  V::Vector{Vector{Float64}} = [v0]

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

  iter = 1

  # Iterate until k eigenpairs are converged
  while k <= pairs && iter <= max_iter

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

    # Update the Galerkin approximation
    for i = 1 : m - 1
      M[i, m] = dot(V[i], W[m])
      M[m, i] = dot(V[m], W[i])
    end
    M[m, m] = dot(V[m], W[m])

    # Compute the Ritz values and pre-Ritz vectors; here we exploit symmetry
    # M S = S diagm(θs) with S orthonormal
    θs, S = eig(Symmetric(M[1 : m, 1 : m]))

    # Sort the ritz values
    perm = sortperm(abs(θs - target))
    θs = θs[perm]
    S = S[:, perm]

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
      W_new::Vector{Vector{Float64}} = []

      for i = 1 : m
        push!(V_new, mv_product(V, S[:, i + 1], m + 1))
        push!(W_new, A * V_new[i])
      end

      shift!(θs)
      M = zeros(max_dimension, max_dimension)
      M[1 : m, 1 : m] = diagm(θs)
      V = V_new
      W = W_new
      S = eye(m)

      u = V[1]
      r = A * u - θs[1] * u

      push!(residuals, norm(r))
    end

    # Do a restart
    if m == max_dimension

      # We want to update V <- V * S[:, 1 : min_dimension]
      # without allocating new memory. So we compute LU = S
      # and then V *= L (column-wise from left -> right)
      # followed by V *= U (column-wise from right -> left)

      # Also, we already have u = V * S[:, 1] at our disposal

      my_V::Vector{Vector{Float64}} = []
      my_W::Vector{Vector{Float64}} = []

      push!(my_V, u)
      push!(my_W, W[1])

      for i = 2 : min_dimension
        push!(my_V, mv_product(V, S[:, i], m))
        push!(my_W, A * my_V[i])
      end

      m = min_dimension
      θs = θs[1 : m]
      M = zeros(max_dimension, max_dimension)
      M[1 : m, 1 : m] = diagm(θs)
      V = my_V
      W = my_W
      S = eye(m)
    end

    # Solve the correction equation
    if iter < 3
      push!(V, r)
    else
      push!(V, solve_deflated_correction(solver, A, θs[1], X[:, 1 : k], u, r))
    end

    iter += 1
  end

  return D[1 : k], X[:, 1 : k], residuals
end

function mv_product{T}(V::Vector{Vector{T}}, x::Vector{T}, m)
  u = V[1] * x[1]
  for i = 2 : m
    u += V[i] * x[i]
  end
  u
end
