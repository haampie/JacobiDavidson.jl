function jacobi_davidson_nonhermetian{Alg <: CorrectionSolver, T}(
  A,                       # Some square Hermetian matrix
  solver::Alg;             # Solver for the correction equation
  pairs::Int = 5,          # Number of eigenpairs wanted
  max_dimension::Int = 80, # Maximal search space size
  min_dimension::Int = 10, # Minimal search space size
  max_iter::Int = 200,
  target::Target = LM(),   # Search target
  ɛ::Float64 = 1e-7,       # Maximum residual norm
  v0::Vector{T} = rand(T, size(A, 1))
)

  residuals::Vector{Float64} = []

  n = size(A, 1)

  # V's vectors span the search space
  V::Vector{Vector{Complex{Float64}}} = [v0]

  # W will just hold the matrix A * V
  W::Vector{Vector{Complex{Float64}}} = []

  # And M will be the Galerkin approximation of A, i.e. M = V' * A * V = V' * W
  M = zeros(Complex{Float64}, max_dimension, max_dimension)

  # k is the number of converged eigenpairs
  k = 0

  # m is the current dimension of the search subspace
  m = 0

  # Q holds the converged eigenvectors as Schur vectors
  Q = zeros(Complex{Float64}, n, pairs)

  # R is upper triangular and has the converged eigenvalues on the diagonal
  R = zeros(Complex{Float64}, pairs, pairs)

  iter = 1

  # Iterate until k eigenpairs are converged
  while k <= pairs && iter <= max_iter

    # Orthogonalize V[m + 1] w.r.t. {V[1], …, V[m]} using modified Gramm-Schmidt
    for i = 1 : m
      @blas! V[m + 1] -= dot(V[m + 1], V[i]) * V[i]
    end

    # Normalize the new vector
    @blas! V[m + 1] *= 1.0 / norm(V[m + 1])

    # Increment the search subspace dimension
    m += 1

    # Add A * V[m]
    push!(W, A * V[m])

    # Update the Galerkin approximation
    M[m, m] = dot(V[m], W[m])
    for i = 1 : m - 1
      M[i, m] = dot(V[i], W[m])
      M[m, i] = dot(V[m], W[i])
    end

    # Compute the Schur decomp M Z = Z T with Z unitary & T upper triangular
    T, Z, θs = schur(view(M, 1 : m, 1 : m))

    # Sort the Schur form (only the most promising vector must be moved up front)
    permutation = schur_permutation(target, θs)
    Π = falses(m)
    Π[permutation[1]] = true
    _, _, Θs = ordschur!(T, Z, Π)

    θ = Θs[1]
    y = Z[:, 1]

    # Consider the first pre-Ritz vector
    # and lift it to Ritz vector (i.e. compute V * y)
    u = mv_product(V, y, m)

    # Compute the residual
    # Two choices here: either compute A * u via a matrix-vector product
    # or use A * u = A * V * y = W * y; whatever is least work.
    r = A * u - θ * u
    z = Q[:, 1 : k]' * r

    # r_proj = (I - QQ')r is the residual without other Schur directions.
    # Clearly we cannot expect |r| to be small, because u is a Schur vector,
    # not an eigenvector.
    r_proj = r - Q[:, 1 : k] * z

    push!(residuals, norm(r_proj))

    # Find one (or more) converged eigenpairs
    while norm(r_proj) ≤ ɛ

      # Store the approximate eigenpair
      R[k + 1, k + 1] = θ
      R[1 : k, k + 1] = z
      Q[:, k + 1] = u

      # A Schur vector has converged
      k += 1

      # Have we found all pairs?
      if k == pairs
        return Q, R, residuals
      end

      # Reorder the Schur form, moving the converged (first) one to the back
      Π = trues(m)
      Π[1] = false
      _, _, θs = ordschur!(T, Z, Π)

      # Reduce dimension of search space as one vector is removed
      m -= 1

      # Remove directions of the converged Schur vector by making a change of basis.
      # The new basis consists of all Schur vectors except the converged one.
      V_new::Vector{Vector{Complex{Float64}}} = []
      W_new::Vector{Vector{Complex{Float64}}} = []

      for i = 1 : m
        push!(V_new, mv_product(V, Z[:, i], m + 1))
        push!(W_new, A * V_new[i])
      end

      M = zeros(Complex{Float64}, max_dimension, max_dimension)
      M[1 : m, 1 : m] = T[1 : m, 1 : m]
      V = V_new
      W = W_new

      T = eye(Complex{Float64}, m)
      Z = T[1 : m, 1 : m]
      θs = θs[2 : end]

      # Sort the Schur form (only the most promising vector must be moved up front)
      permutation = schur_permutation(target, θs)
      Π = falses(m)
      Π[permutation[1]] = true
      _, _, Θs = ordschur!(T, Z, Π)

      θ = Θs[1]
      y = Z[:, 1]
      u = mv_product(V, y, m)
      r = A * u - θ * u
      z = Q[:, 1 : k]' * r
      r_proj = r - Q[:, 1 : k] * z

      push!(residuals, norm(r_proj))
    end

    # Do a restart
    if m == max_dimension

      # We want to make a change of basis so that V spans the most interesting
      # Scur vectors only. Therefore we must reorder the Schur form, taking the
      # first min_dimension Schur vectors to the first columns
      # The most interesting Schur vector will remain in the first colum.

      permutation = schur_permutation(target, θs)
      Π = falses(m)
      Π[permutation[1 : min_dimension]] = true
      ordschur!(T, Z, Π)

      my_V::Vector{Vector{Complex{Float64}}} = []
      my_W::Vector{Vector{Complex{Float64}}} = []

      for i = 1 : min_dimension
        push!(my_V, mv_product(V, Z[:, i], m))
        push!(my_W, A * my_V[i])
      end

      # Reduce the search space dimension
      m = min_dimension

      M = zeros(Complex{Float64}, max_dimension, max_dimension)
      M[1 : m, 1 : m] = T[1 : m, 1 : m]
      V = my_V
      W = my_W
    end

    # Solve the correction equation
    if iter < 3
      push!(V, r)
    else
      push!(V, solve_deflated_correction(solver, A, θ, Q[:, 1 : k], u, r))
    end

    iter += 1
  end

  # Not fully converged.
  return Q[:, 1 : k], R[1 : k, 1 : k], residuals
end

function mv_product{T}(V::Vector{Vector{T}}, x::Vector{T}, m)
  u = V[1] * x[1]
  for i = 2 : m
    @blas! u += x[i] * V[i]
  end
  u
end
