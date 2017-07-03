export jdqr_harmonic_matrix

function jdqr_harmonic_matrix{Alg <: CorrectionSolver}(
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
    V = zeros(T, n, max_dimension)

    # AV will store (A - tI) * V, without any orthogonalization
    AV = zeros(T, n, max_dimension)

    # W will hold AV, but with its columns orthogonal: AV = W * MA
    W = zeros(T, n, max_dimension)

    # Temporaries (trying to reduces #allocations here)
    # V_tmp = zeros(T, n, max_dimension)
    # AV_tmp = zeros(T, n, max_dimension)
    # W_tmp = zeros(T, n, max_dimension)

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
            view(V, :, m + 1) .= rand(n) # Initialize with a random vector
        elseif iter < 3
            view(V, :, m + 1) .= r # Expand with the residual ~ Arnoldi style
        else
            view(V, :, m + 1) .= solve_deflated_correction(solver, A, rayleigh + τ, view(Q, :, 1 : k), u, r)
        end

        m += 1

        # Search space is orthonormalized
        orthogonalize_and_normalize!(view(V, :, 1 : m - 1), view(V, :, m), zeros(T, m - 1), DGKS)

        # AV is just the product (A - τI)V
        A_mul_B!(view(AV, :, m), A, view(V, :, m))
        axpy!(-τ, view(V, :, m), view(AV, :, m))

        # Expand W with (A - τI)V, and then orthogonalize
        view(W, :, m) .= view(AV, :, m)

        # Orthogonalize w.r.t. the converged Schur vectors
        just_orthogonalize!(view(Q, :, 1 : k), view(W, :, m), DGKS)

        # Orthonormalize W[:, m] w.r.t. previous columns of W
        MA[m, m] = orthogonalize_and_normalize!(
            view(W, :, 1 : m - 1),
            view(W, :, m), 
            view(MA, 1 : m - 1, m),
            DGKS
        )

        # Update the right-most column and last row of M = W' * V
        # We can still save 1 inner product (right-bottom value is computed twice now)
        # Ac_mul_B!(view(M, 1 : m, m), view(W, :, 1 : m), view(V, :, m))
        # Ac_mul_B!(view(M, m, 1 : m), view(W, :, m), view(V, :, 1 : m))

        M[m, m] = dot(view(W, :, m), view(V, :, m))
        @inbounds for i = 1 : m - 1
            M[i, m] = dot(view(W, :, i), view(V, :, m))
            M[m, i] = dot(view(W, :, m), view(V, :, i))
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
        push!(converged_ritz_values, diag(view(R, 1 : k, 1 : k)))
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
            # @assert norm(A * @view(Q[:, k]) - @view(Q[:, 1 : k]) * @view(R[k, 1 : k])) < k * ɛ

            # Are we done yet?
            if k == pairs
                return Q, R, harmonic_ritz_values, converged_ritz_values, residuals
            end

            # Now remove this eigenvector direction from the search space.

            # Shrink V, W and AV
            V[:, 1 : m - 1] = view(V, :, 1 : m) * view(F[:right], :, 2 : m)
            AV[:, 1 : m - 1] = view(AV, :, 1 : m) * view(F[:right], :, 2 : m)
            W[:, 1 : m - 1] = view(W, :, 1 : m) * view(F[:left], :, 2 : m)

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
            println("Shrinking the search space.")

            # Move min_dimension of the smallest harmonic Ritz values up front
            smallest = selectperm(abs(F[:alpha] ./ F[:beta]), 1 : min_dimension)
            p = falses(m)
            p[smallest] = true
            ordschur!(F, p)

            # Shrink V, W, AV, and update M and MA.
            V[:, 1 : min_dimension] = V * view(F[:right], :, 1 : min_dimension)
            AV[:, 1 : min_dimension] = AV * view(F[:right], :, 1 : min_dimension)
            W[:, 1 : min_dimension] = W * view(F[:left], :, 1 : min_dimension)

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
  F = schurfact(view(MA, 1 : m, 1 : m), view(M, 1 : m, 1 : m))

  # Move the smallest harmonic Ritz value up front
  smallest = indmin(abs.(F[:alpha] ./ F[:beta]))
  p = falses(m)
  p[smallest] = true
  ordschur!(F, p)

  # Pre-ritz vector
  y = F.Z[:, 1]

  # Ritz vector
  u = view(V, :, 1 : m) * y

  # Rayleigh quotient = approx eigenvalue s.t. if r = (A-τI)u - rayleigh * u, then r ⟂ u
  rayleigh = conj(F.beta[1]) * F.alpha[1]
  
  # Residual r = (A - τI)u - rayleigh * u = AV*y - rayleigh * u
  r = view(AV, :, 1 : m) * y
  axpy!(-rayleigh, u, r)
  
  # Orthogonalize w.r.t. Q
  z = zeros(T, k)
  just_orthogonalize!(view(Q, :, 1 : k), r, z, DGKS)
  
  # Assert that the residual is perpendicular to the Ritz vector
  # @assert abs(dot(r, u)) < ɛ

  F, u, rayleigh, r, z
end

