function jdqr_harmonic_efficient{Alg <: CorrectionSolver}(
  A,                       # Some square Hermetian matrix
  solver::Alg;             # Solver for the correction equation
  pairs::Int = 5,          # Number of eigenpairs wanted
  max_dimension::Int = 20, # Maximal search space size
  min_dimension::Int = 10, # Minimal search space size
  max_iter::Int = 200,
  τ::Complex128 = 0.0 + 0im,       # Search target
  ɛ::Float64 = 1e-7,       # Maximum residual norm
  T::Type = Complex128
)

  residuals::Vector{real(T)} = []

  n = size(A, 1)

  # V's vectors span the search space
  V = [zeros(T, n) for i = 1 : max_dimension]

  # AV will store (A - tI) * V, without any orthogonalization
  AV = [zeros(T, n) for i = 1 : max_dimension]

  # W will hold AV, but with its columns orthogonal: AV = W * MA
  W = [zeros(T, n) for i = 1 : max_dimension]

  # Temporaries (trying to reduces #allocations here)
  V_tmp = [zeros(T, n) for i = 1 : max_dimension]
  AV_tmp = [zeros(T, n) for i = 1 : max_dimension]
  W_tmp = [zeros(T, n) for i = 1 : max_dimension]

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
      V[m + 1] = rand(n) # Initialize with a random vector
    elseif iter < 3
      V[m + 1] = r # Expand with the residual ~ Arnoldi style
    else
      V[m + 1] = solve_deflated_correction(solver, A, rayleigh + τ, @view(Q[:, 1 : k]), u, r)
    end

    m += 1

    # Search space is orthonormalized
    orthogonalize!(V[1 : m - 1], V[m])
    scale!(V[m], one(T) / norm(V[m]))

    # AV is just the product (A - τI)V
    A_mul_B!(AV[m], A, V[m])
    axpy!(-τ, V[m], AV[m])

    # Expand W with (A - τI)V, and then orthogonalize
    W[m] .= AV[m]

    # Orthogonalize w.r.t. the converged Schur vectors using Gram-Schmidt
    orthogonalize!(@view(Q[:, 1 : k]), W[m])

    # Orthonormalize W[:, m] w.r.t. previous columns of W
    orthogonalize_and_factorize!(W[1 : m - 1], W[m], @view(MA[1 : m - 1, m]))
    MA[m, m] = norm(W[m])
    scale!(W[m], one(T) / MA[m, m])

    # Update M
    M[m, m] = dot(W[m], V[m])
    @inbounds for i = 1 : m - 1
      M[i, m] = dot(W[i], V[m])
      M[m, i] = dot(W[m], V[i])
    end

    # Assert orthogonality of V and W
    # Assert W * MA = (I - QQ') * (A - τI) * V
    # Assert that M = W' * V
    # @assert norm(hcat(W[1 : m + 1]...)' * hcat(W[1 : m + 1]...) - eye(m + 1)) < 1e-12
    # @assert norm(hcat(V[1 : m + 1]...)' * hcat(V[1 : m + 1]...) - eye(m + 1)) < 1e-12
    # @assert norm(hcat(W[1 : m + 1]...) * MA[1 : m + 1, 1 : m + 1] - hcat(AV[1 : m + 1]...) + (Q[:, 1 : k] * (Q[:, 1 : k]' * hcat(AV[1 : m + 1]...)))) < pairs * ɛ
    # @assert norm(M[1 : m + 1, 1 : m + 1] - hcat(W[1 : m + 1]...)' * hcat(V[1 : m + 1]...)) < 1e-12

    # Finally increment the search space dimension

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
      # println("Found an eigenvalue", rayleigh + τ)

      R[k + 1, k + 1] = rayleigh + τ # Eigenvalue
      R[k + 1, 1 : k] = z            # Makes AQ = QR
      Q[:, k + 1] = u
      k += 1

      # Add another one in the history
      push!(converged_ritz_values[iter], rayleigh + τ)

      # Make sure the Schur decomp AQ = QR is approximately correct
      # @assert norm(A * @view(Q[:, k]) - @view(Q[:, 1 : k]) * @view(R[k, 1 : k])) < k * ɛ

      # Are we done yet?
      if k == pairs
        return Q, R, harmonic_ritz_values, converged_ritz_values, residuals
      end

      # Now remove this eigenvector direction from the search space.

      # Shrink V, W and AV
      for i = 1 : m - 1
        matrix_vector!(V_tmp[i], V[1 : m], @view(F[:right][:, i + 1]))
        matrix_vector!(AV_tmp[i], AV[1 : m], @view(F[:right][:, i + 1]))
        matrix_vector!(W_tmp[i], W[1 : m], @view(F[:left][:, i + 1]))
      end

      for i = 1 : m - 1
        copy!(V[i], V_tmp[i])
        copy!(AV[i], AV_tmp[i])
        copy!(W[i], W_tmp[i])
      end

      # Update the projection matrices M and MA.
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
      # println("Shrinking the search space.")

      # Move min_dimension of the smallest harmonic Ritz values up front
      smallest = selectperm(abs(F[:alpha] ./ F[:beta]), 1 : min_dimension)
      p = falses(m)
      p[smallest] = true
      ordschur!(F, p)

      # Shrink V, W, AV, and update M and MA.
      for i = 1 : min_dimension
        matrix_vector!(V_tmp[i], V, @view(F[:right][:, i]))
        matrix_vector!(AV_tmp[i], AV, @view(F[:right][:, i]))
        matrix_vector!(W_tmp[i], W, @view(F[:left][:, i]))
      end

      for i = 1 : min_dimension
        copy!(V[i], V_tmp[i])
        copy!(AV[i], AV_tmp[i])
        copy!(W[i], W_tmp[i])
      end

      m = min_dimension
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
  u = hcat(V[1 : m]...) * y
  # u = matrix_vector(V[1 : m], y)

  # Rayleigh quotient = approx eigenvalue s.t. if r = (A-τI)u - rayleigh * u, then r ⟂ u
  rayleigh = conj(F.beta[1]) * F.alpha[1]
  
  # Residual r = (A - τI)u - rayleigh * u = AV*y - rayleigh * u
  r = matrix_vector(AV[1 : m], y)
  axpy!(-rayleigh, u, r)
  
  # Orthogonalize w.r.t. Q
  z = zeros(T, k)
  orthogonalize_and_factorize!(@view(Q[:, 1 : k]), r, z)
  
  # Assert that the residual is perpendicular to the Ritz vector
  # @assert abs(dot(r, u)) < ɛ

  F, u, rayleigh, r, z
end

function orthogonalize!{T}(V::Vector{Vector{T}}, target)
  # Repeated Modified Gram-Schmidt

  for column = V
    axpy!(-dot(column, target), column, target)
  end

  for column = V
    axpy!(-dot(column, target), column, target)
  end
end

function orthogonalize!(V, target)
  # Repeated Modified Gram-Schmidt

  for idx = 1 : size(V, 2)
    column = @view(V[:, idx])
    axpy!(-dot(column, target), column, target)
  end

  for idx = 1 : size(V, 2)
    column = @view(V[:, idx])
    axpy!(-dot(column, target), column, target)
  end
end

function orthogonalize_and_factorize!{T}(V::Vector{Vector{T}}, target, factors)
  for idx = 1 : length(V)
    factors[idx] = dot(V[idx], target)
    axpy!(-factors[idx], V[idx], target)
  end

  # Reorthogonalize
  for idx = 1 : length(V)
    increment = dot(V[idx], target)
    factors[idx] += increment
    axpy!(-increment, V[idx], target)
  end
end

function orthogonalize_and_factorize!(V, target, factors)
  for idx = 1 : size(V, 2)
    column = @view(V[:, idx])
    factors[idx] = dot(column, target)
    axpy!(-factors[idx], column, target)
  end

  # Reorthogonalize
  for idx = 1 : size(V, 2)
    column = @view(V[:, idx])
    increment = dot(column, target)
    factors[idx] += increment
    axpy!(-increment, column, target)
  end
end

function matrix_vector!{T}(dest::Vector{T}, V::Vector{Vector{T}}, y)
  dest .= zeros(dest)

  for idx = 1 : length(V)
    axpy!(y[idx], V[idx], dest)
  end

  dest
end

matrix_vector{T}(V::Vector{Vector{T}}, y) = matrix_vector!(zeros(V[1]), V, y)