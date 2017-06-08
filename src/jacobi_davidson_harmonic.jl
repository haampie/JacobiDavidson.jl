function jacobi_davidson_harmonic{Alg <: CorrectionSolver, Tn}(
  A,                       # Some square Hermetian matrix
  solver::Alg;             # Solver for the correction equation
  pairs::Int = 5,          # Number of eigenpairs wanted
  max_dimension::Int = 20, # Maximal search space size
  min_dimension::Int = 10, # Minimal search space size
  max_iter::Int = 200,
  target::Target = Near(40.0), # Search target
  ɛ::Float64 = 1e-7,           # Maximum residual norm
  v0::Vector{Tn} = rand(Tn, size(A, 1))
)

  residuals::Vector{Float64} = []

  n = size(A, 1)

  # V's vectors span the search space
  V::Vector{Vector{Complex{Float64}}} = [v0]

  # AV will store (A - tI) * V, without any orthogonalization
  AV::Vector{Vector{Complex{Float64}}} = []

  # W will hold AV, but with its columns orthogonal: AV = W * MA
  W::Vector{Vector{Complex{Float64}}} = []

  # MA will be the upper triangular Petrov-Galerkin approximation of (A - tI): MA = ((A - tI)V)' * (A - tI)V = W' * W
  MA = zeros(Complex{Float64}, max_dimension, max_dimension)

  # M will hold the rhs matrix in the generalized eigenvalue problem, so M = ((A - tI)V)' * V = W' * V
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

    # Orthogonalize V[m + 1] w.r.t. {V[1], …, V[m]} using modified Gram-Schmidt
    for i = 1 : m
      V[m + 1] -= dot(V[m + 1], V[i]) * V[i]
    end

    # Normalize the new vector
    V[m + 1] *= 1.0 / norm(V[m + 1])

    # Store (A - tI) * V[m + 1]
    push!(AV, A * V[m + 1] - target.target * V[m + 1])
    
    # And copy AV[m + 1] to W as well, so that we can next orthogonalize
    push!(W, AV[m + 1])

    # Orthogonalize W[m + 1] with respect to Q[:, 1:k] using modified Gram-Schmidt
    for i = 1 : k
        W[m + 1] -= dot(Q[:, k], W[m + 1]) * Q[:, k]
    end

    # Orthogonalize W[m + 1] with respect to W[1..m] using modified Gram-Schmidt
    for i = 1 : m
        MA[i, m] = dot(W[m + 1], W[i])
        W[m + 1] -= MA[i, m] * W[i]
    end

    # Update the M = W' * V matrix
    M[m + 1, m + 1] = dot(V[m + 1], W[m + 1])
    for i = 1 : m
      M[i, m + 1] = dot(W[i], V[m + 1])
      M[m + 1, i] = dot(W[m + 1], V[i])
    end

    # Increment the search subspace dimension
    m += 1

    # Compute the generalized Schur decomposition
    #  M Z = Z T with Z unitary & T upper triangular
    F = schurfact(@view(MA[1 : m, 1 : m]), @view(MA[1 : m, 1 : m]))
    θs = F.alpha ./ F.beta

    # Sort the Schur form (only the single most promising vector must be moved up front)
    permutation = schur_permutation(target, θs)
    Π = falses(m)
    Π[permutation[1]] = true
    ordschur!(F, Π)

    # For expansion we work with the Rayleigh quotient, so that the residual is automatically
    # perpendicular to the Ritz vector.
    rayleigh = conj(F.alpha[1]) * F.beta[1]
    
    # Pre-ritz vector
    y = F.Z[:, 1]

    # Lift it to an actual Ritz vector V * y
    u = mv_product(V, y, m)

    # Compute the residual (A - tI)u - rayleigh * u = AV * y - rayleigh * u
    r = mv_product(AV, y, m) - rayleigh * u
    z = Q[:, 1 : k]' * r

    # r_proj = (I - QQ')r is the residual without other Schur directions.
    # Clearly we cannot expect |r| to be small, because u is a Schur vector,
    # not an eigenvector.
    r_proj = r - Q[:, 1 : k] * z

    push!(residuals, norm(r_proj))

    # Find one (or more) converged eigenpairs
    while norm(r_proj) ≤ ɛ

      # Store the approximate eigenpair
      R[k + 1, k + 1] = rayleigh + target.target
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
      ordschur!(F, Π)

      # Reduce dimension of search space as one vector is removed
      m -= 1

      # Remove directions of the converged Schur vector by making a change of basis.
      # The new basis consists of all Schur vectors except the converged one.
      V_new::Vector{Vector{Complex{Float64}}} = []
      AV_new::Vector{Vector{Complex{Float64}}} = []
      W_new::Vector{Vector{Complex{Float64}}} = []

      for i = 1 : m
        push!(V_new, mv_product(V, F.Z[:, i], m + 1))
        push!(AV_new, mv_product(AV, F.Z[:, i]), m + 1)
        push!(W_new, mv_product(W, F.Q[:, i], m + 1))
      end

      AV = AV_new
      V = V_new
      W = W_new

      # Update the projection matrices
      M = zeros(Complex{Float64}, max_dimension, max_dimension)
      MA = zeros(Complex{Float64}, max_dimension, max_dimension)
      M[1 : m, 1 : m] = F.T[1 : m, 1 : m]
      MA[1 : m, 1 : m] = F.S[1 : m, 1 : m]

      # Update the Schur decomp
      F = Base.LinAlg.GeneralizedSchur(
        MA[1 : m, 1 : m],
        M[1 : m, 1 : m],
        F.alpha[1 : m],
        F.beta[1 : m],
        eye(m),
        eye(m)
      )

      # Sort the Schur form (only the most promising vector must be moved up front)
      θs = F.alpha ./ F.beta
      permutation = schur_permutation(target, θs)
      Π = falses(m)
      Π[permutation[1]] = true
      ordschur!(F, Π)

      rayleigh = conj(F.alpha[1]) * F.beta[1]
      y = F.Z[:, 1]
      u = mv_product(V, y, m)
      r = mv_product(AV, y, m) - rayleigh * u
      z = Q[:, 1 : k]' * r
      r_proj = r - Q[:, 1 : k] * z
    end

    # Do a restart
    if m == max_dimension

      # We want to make a change of basis so that V spans the most interesting
      # Schur vectors only. Therefore we must reorder the Schur form, taking the
      # first min_dimension Schur vectors to the first columns
      # The most interesting Schur vector will remain in the first colum.

      permutation = schur_permutation(target, θs)
      Π = falses(m)
      Π[permutation[1 : min_dimension]] = true
      ordschur!(F, Π)

      my_V::Vector{Vector{Complex{Float64}}} = []
      my_AV::Vector{Vector{Complex{Float64}}} = []
      my_W::Vector{Vector{Complex{Float64}}} = []

      # Shrink to min_dimension
      m = min_dimension

      for i = 1 : m
        push!(my_V, mv_product(V, F.Z[:, i], max_dimension))
        push!(my_AV, mv_product(AV, F.Z[:, i]), max_dimension)
        push!(my_W, mv_product(W, F.Q[:, i], max_dimension))
      end

      AV = AV_new
      V = V_new
      W = W_new

      # Update the projection matrices
      M = zeros(Complex{Float64}, max_dimension, max_dimension)
      MA = zeros(Complex{Float64}, max_dimension, max_dimension)
      M[1 : m, 1 : m] = F.T[1 : m, 1 : m]
      MA[1 : m, 1 : m] = F.S[1 : m, 1 : m]

      # Update the Schur decomp
      F = Base.LinAlg.GeneralizedSchur(
        MA[1 : m, 1 : m],
        M[1 : m, 1 : m],
        F.alpha[1 : m],
        F.beta[1 : m],
        eye(min_dimension),
        eye(min_dimension)
      )

      # Sort the Schur form (only the most promising vector must be moved up front)
      θs = F.alpha ./ F.beta
      permutation = schur_permutation(target, θs)
      Π = falses(m)
      Π[permutation[1]] = true
      ordschur!(F, Π)

      rayleigh = conj(F.alpha[1]) * F.beta[1]
      y = F.Z[:, 1]
      u = mv_product(V, y, m)
      r = mv_product(AV, y, m) - rayleigh * u
      z = Q[:, 1 : k]' * r
      r_proj = r - Q[:, 1 : k] * z
    end

    # Solve the correction equation
    # push!(V, -r)
    push!(V, solve_deflated_correction(solver, A, rayleigh, @view(Q[:, 1 : k]), u, r))

    iter += 1
  end

  # Not fully converged.
  return Q[:, 1 : k], R[1 : k, 1 : k], residuals
end

function mv_product{T}(V::Vector{Vector{T}}, x, m)
  u = V[1] * x[1]
  for i = 2 : m
    u += x[i] * V[i]
  end
  u
end
