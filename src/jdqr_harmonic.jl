function jdqr_harmonic{Alg <: CorrectionSolver}(
  A,                       # Some square Hermetian matrix
  solver::Alg;             # Solver for the correction equation
  pairs::Int = 5,          # Number of eigenpairs wanted
  max_dimension::Int = 20, # Maximal search space size
  min_dimension::Int = 10, # Minimal search space size
  max_iter::Int = 200,
  τ::Complex128 = 40.0 + 0im,       # Search target
  ɛ::Float64 = 1e-7,       # Maximum residual norm
  T::Type = Complex128
)

  residuals::Vector{real(T)} = []

  n = size(A, 1)

  # V's vectors span the search space
  V = zeros(T, n, max_dimension)

  # AV will store (A - tI) * V, without any orthogonalization
  AV = zeros(T, n, max_dimension)

  # W will hold AV, but with its columns orthogonal: AV = W * MA
  W = zeros(T, n, max_dimension)

  # Projected matrices
  MA = zeros(T, max_dimension, max_dimension)
  M = zeros(T, max_dimension, max_dimension)

  # k is the number of converged eigenpairs
  k = 0

  # m is the current dimension of the search subspace
  m = 0

  # Q holds the converged eigenvectors as Schur vectors
  Q = zeros(T, n, pairs)

  # R is upper triangular and has the converged eigenvalues on the diagonal
  R = zeros(T, pairs, pairs)

  iter = 1

  harmonic_ritz_values::Vector{Vector{T}} = []
  converged_ritz_values::Vector{Vector{T}} = []

  while k <= pairs && iter <= max_iter
    if iter == 1
      V[:, m + 1] = rand(n) # Initialize with a random vector
    elseif iter < 5
      V[:, m + 1] = r # Expand with the residual ~ Arnoldi style
    else
      V[:, m + 1] = solve_deflated_correction(solver, A, rayleigh + τ, @view(Q[:, 1 : k]), u, r)
    end

    # Search space is orthogonalized
    # orthogonalize_to_columns!(@view(V[:, m + 1]), @view(V[:, 1 : m]))
    for i = 1 : m
      V[:, m + 1] -= dot(V[:, i], V[:, m + 1]) * V[:, i]
    end

    # Reorthogonalize
    for i = 1 : m
      V[:, m + 1] -= dot(V[:, i], V[:, m + 1]) * V[:, i]
    end
    
    # Normalize
    V[:, m + 1] /= norm(V[:, m + 1])

    # AV is just the product (A - τI)V
    AV[:, m + 1] = A * V[:, m + 1] - τ * V[:, m + 1]

    # Expand W with (A - τI)V, and then orthogonalize
    W[:, m + 1] = AV[:, m + 1]

    # Orthogonalize w.r.t. the converged Schur vectors using Gram-Schmidt
    for i = 1 : k
      W[:, m + 1] -= dot(Q[:, i], W[:, m + 1]) * Q[:, i]
    end

    # Reorthogonalize
    for i = 1 : k
      W[:, m + 1] -= dot(Q[:, i], W[:, m + 1]) * Q[:, i]
    end

    # Orthogonalize W[:, m + 1] w.r.t. previous columns of W
    for i = 1 : m
      MA[i, m + 1] = dot(W[:, i], W[:, m + 1])
      W[:, m + 1] -= MA[i, m + 1] * W[:, i]
    end

    # Reorthogonalize
    for i = 1 : m
      increment = dot(W[:, i], W[:, m + 1])
      MA[i, m + 1] += increment
      W[:, m + 1] -= increment * W[:, i]
    end

    # Normalize
    MA[m + 1, m + 1] = norm(W[:, m + 1])
    W[:, m + 1] /= MA[m + 1, m + 1]

    # Update M
    for i = 1 : m
      M[i, m + 1] = dot(W[:, i], V[:, m + 1])
      M[m + 1, i] = dot(W[:, m + 1], V[:, i])
    end
    M[m + 1, m + 1] = dot(W[:, m + 1], V[:, m + 1])

    # Assert orthogonality of V and W
    # Assert W * MA = (I - QQ') * (A - τI) * V
    # Assert that M = W' * V
    @assert norm(W[:, 1 : m + 1]' * W[:, 1 : m + 1] - eye(m + 1)) < 1e-12
    @assert norm(V[:, 1 : m + 1]' * V[:, 1 : m + 1] - eye(m + 1)) < 1e-12
    @assert norm(W[:, 1 : m + 1] * MA[1 : m + 1, 1 : m + 1] - AV[:, 1 : m + 1] + (Q[:, 1 : k] * (Q[:, 1 : k]' * AV[:, 1 : m + 1]))) < pairs * ɛ
    @assert norm(M[1 : m + 1, 1 : m + 1] - W[:, 1 : m + 1]' * V[:, 1 : m + 1]) < 1e-12

    # Finally increment the search space dimension
    m += 1

    # F is the Schur decomp
    # u is the approximate eigenvector
    # rayleigh is the Rayleigh quotient: approx. shifted eigenvalue.
    # r is its residual with Schur directions removed
    # z is the projection of the residual on Q

    F, u, rayleigh, r, z = extract(MA, M, V, AV, Q, m, k, ɛ, T)

    # Convergence history of the harmonic Ritz values
    push!(harmonic_ritz_values, τ + F[:alpha] ./ F[:beta])
    push!(converged_ritz_values, diag(@view(R[1 : k, 1 : k])))
    push!(residuals, norm(r))

    # An Ritz vector is converged
    while norm(r) ≤ ɛ
      println("Found an eigenvalue", rayleigh + τ)

      R[k + 1, k + 1] = rayleigh + τ # Eigenvalue
      R[k + 1, 1 : k] = z            # Makes AQ = QR
      Q[:, k + 1] = u
      k += 1

      # Add another one in the history
      push!(converged_ritz_values[iter], rayleigh + τ)

      # Make sure the Schur decomp AQ = QR is approximately correct
      @assert norm(A * @view(Q[:, k]) - @view(Q[:, 1 : k]) * @view(R[k, 1 : k])) < k * ɛ

      # Are we done yet?
      if k == pairs
        return Q, R, harmonic_ritz_values, converged_ritz_values, residuals
      end

      # Now remove this eigenvector direction from the search space.

      # Shrink V, W and AV and update M and MA.
      V[:, 1 : m - 1] = @view(V[:, 1 : m]) * @view(F[:right][:, 2 : m])
      W[:, 1 : m - 1] = @view(W[:, 1 : m]) * @view(F[:left][:, 2 : m])
      AV[:, 1 : m - 1] = @view(AV[:, 1 : m]) * @view(F[:right][:, 2 : m])

      M[1 : m - 1, 1 : m - 1] = F.T[2 : m, 2 : m]
      MA[1 : m - 1, 1 : m - 1] = F.S[2 : m, 2 : m]

      m -= 1

      # TODO: Can the search space become empty? Probably, but not likely.
      if m == 0
        return Q[:, 1 : k], R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals
      end

      F, u, rayleigh, r, z = extract(MA, M, V, AV, Q, m, k, ɛ, T)
    end

    if m == max_dimension
      println("Shrinking the search space.")

      # Move min_dimension of the smallest harmonic Ritz values up front
      smallest = selectperm(abs(F[:alpha] ./ F[:beta]), 1 : min_dimension)
      p = falses(m)
      p[smallest] = true
      ordschur!(F, p)

      m = min_dimension

      # Shrink V, W, AV, and update M and MA.
      V[:, 1 : m] = @view(V[:, 1 : max_dimension]) * @view(F[:right][:, 1 : m])
      W[:, 1 : m] = @view(W[:, 1 : max_dimension]) * @view(F[:left][:, 1 : m])
      AV[:, 1 : m] = @view(AV[:, 1 : max_dimension]) * @view(F[:right][:, 1 : m])
      M[1 : m, 1 : m] = F.T[1 : m, 1 : m]
      MA[1 : m, 1 : m] = F.S[1 : m, 1 : m]

    end

    iter += 1
  end

  Q[:, 1 : k], R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals

end

function extract(MA, M, V, AV, Q, m, k, ɛ, T)
  # Compute the Schur decomp to find the harmonic Ritz values
  F = schurfact(@view(MA[1 : m, 1 : m]), @view(M[1 : m, 1 : m]))

  # Move the smallest harmonic Ritz value up front
  smallest = indmin(abs(F[:alpha] ./ F[:beta]))
  p = falses(m)
  p[smallest] = true
  ordschur!(F, p)

  # Pre-ritz vector
  y = F.Z[:, 1]

  # Ritz vector
  u = @view(V[:, 1 : m]) * y

  # Rayleigh quotient = approx eigenvalue s.t. if r = (A-τI)u - rayleigh * u, then r ⟂ u
  rayleigh = conj(F.beta[1]) * F.alpha[1]
  
  # Residual r_tilde = (A-τI)u - rayleigh * u = AV*y - rayleigh * u
  r = @view(AV[:, 1 : m]) * y - rayleigh * u
  
  # Orthogonalize w.r.t. Q
  z = zeros(T, k)
  for i = 1 : k
    z[i] = dot(Q[:, i], r)
    r -= z[i] * Q[:, i]
  end

  # Repeat
  for i = 1 : k
    increment = dot(Q[:, i], r)
    z[i] += increment
    r -= increment * Q[:, i]
  end
  
  # Assert that the residual is perpendicular to the Ritz vector
  @assert abs(dot(r, u)) < ɛ

  @show norm(r)

  F, u, rayleigh, r, z
end
