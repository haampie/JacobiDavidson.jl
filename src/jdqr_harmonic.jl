export jdqr_harmonic

"""
`Q_all` will form a unitary matrix of Schur vectors
`Q` contains the locked + currently active Schur vectors
`R` will be upper triangular
`curr` is the current active Schur vector
`locked` is the number of converged Schur vecs
"""
type PartialSchur{matT <: StridedMatrix, viewT <: StridedMatrix, vecT <: StridedVector}
    Q_all::matT
    R::matT
    
    Q::viewT
    curr::vecT
    locked::Int
end

PartialSchur(Q, R) = PartialSchur(Q, R, view(Q, :, 1 : 1), view(Q, :, 1), 0)

function lock!(schur::PartialSchur)
    schur.locked += 1

    if schur.locked < size(schur.Q_all, 2)
        schur.Q = view(schur.Q_all, :, 1 : schur.locked + 1)
        schur.curr = view(schur.curr, :, schur.locked + 1)
    end

    schur
end

function jdqr_harmonic{Alg <: CorrectionSolver}(
    A,                       # Some square matrix
    solver::Alg;             # Solver for the correction equation
    pairs::Int = 5,          # Number of eigenpairs wanted
    min_dimension::Int = 10, # Minimal search space size
    max_dimension::Int = 20, # Maximal search space size
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
    large_matrix_tmp = zeros(T, n, max_dimension)

    # Projected matrices
    MA = zeros(T, max_dimension, max_dimension)
    M = zeros(T, max_dimension, max_dimension)

    # k is the number of converged eigenpairs
    k = 0

    # m is the current dimension of the search subspace
    m = 0

    schur = PartialSchur(zeros(T, n, pairs), zeros(T, pairs, pairs))

    # Current eigenvalue
    θ = zero(T)

    # Current residual vector
    r = Vector{T}(n)

    iter = 1

    harmonic_ritz_values::Vector{Vector{T}} = []
    converged_ritz_values::Vector{Vector{T}} = []

    while k <= pairs && iter <= max_iter
        m += 1

        if iter == 1
            rand!(view(V, :, 1))
        else
            solve_deflated_correction!(view(V, :, m), solver, A, θ, Q, r)
        end

        # Search space is orthonormalized
        orthogonalize_and_normalize!(view(V, :, 1 : m - 1), view(V, :, m), zeros(T, m - 1), DGKS)

        # AV is just the product (A - τI)V
        A_mul_B!(view(AV, :, m), A, view(V, :, m))
        axpy!(-τ, view(V, :, m), view(AV, :, m))

        # Expand W with (A - τI)V, and then orthogonalize
        copy!(view(W, :, m), view(AV, :, m))

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

        search = true

        while search

            F, λ, r = extract_harmonic!(
                view(MA, 1 : m, 1 : m), 
                view(M, 1 : m, 1 : m), 
                view(V, :, 1 : m), 
                view(AV, :, 1 : m), 
                schur,
                τ,
                T
            )

            resnorm = norm(r)

            # Convergence history of the harmonic Ritz values
            push!(harmonic_ritz_values, F.alpha ./ F.beta)
            push!(converged_ritz_values, diag(view(R, 1 : k, 1 : k)))
            push!(residuals, norm(r))

            # An Ritz vector is converged
            if resnorm ≤ ɛ
                println("Found an eigenvalue", λ)

                R[k + 1, k + 1] = λ # Eigenvalue
                k += 1

                lock!(schur)

                # Add another one in the history
                push!(converged_ritz_values[iter], λ)

                # Make sure the Schur decomp AQ = QR is approximately correct
                # @assert norm(A * @view(Q[:, k]) - @view(Q[:, 1 : k]) * @view(R[k, 1 : k])) < k * ɛ

                # Are we done yet?
                if k == pairs
                    return schur.Q, schur.R, harmonic_ritz_values, converged_ritz_values, residuals
                end

                # Now remove this eigenvector direction from the search space.

                # Shrink V, W and AV
                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(V, :, 1 : m), view(F[:right], :, 2 : m))
                large_matrix_tmp, V = V, large_matrix_tmp

                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(AV, :, 1 : m), view(F[:right], :, 2 : m))
                large_matrix_tmp, AV = AV, large_matrix_tmp

                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(W, :, 1 : m), view(F[:left], :, 2 : m))
                large_matrix_tmp, W = W, large_matrix_tmp

                # Update the projection matrices M and MA.
                copy!(view(M, 1 : m - 1, 1 : m - 1), view(F.T, 2 : m, 2 : m))
                copy!(view(MA, 1 : m - 1, 1 : m - 1), view(F.S, 2 : m, 2 : m))

                m -= 1

                # TODO: Can the search space become empty? Probably, but not likely.
                if m == 0
                    return Q[:, 1 : k], R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals
                end
            else
                search = false
            end
        end

        if m == max_dimension
            println("Shrinking the search space.")

            # Move min_dimension of the smallest harmonic Ritz values up front
            schur_sort!(F, SM(), 1 : min_dimension)

            # Shrink V, W, AV, and update M and MA.
            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), V, view(F[:right], :, 1 : min_dimension))
            large_matrix_tmp, V = V, large_matrix_tmp
            
            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), AV, view(F[:right], :, 1 : min_dimension))
            large_matrix_tmp, AV = AV, large_matrix_tmp

            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), W, view(F[:left], :, 1 : min_dimension))
            large_matrix_tmp, W = W, large_matrix_tmp

            m = min_dimension
            copy!(view(M, 1 : m, 1 : m), view(F.T, 1 : m, 1 : m))
            copy!(view(MA, 1 : m, 1 : m), view(F.S, 1 : m, 1 : m))
        end

        iter += 1
    end

    Q[:, 1 : k], R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals

end

function extract_harmonic!(MA, M, V, AV, schur, τ, T)
  # Compute the Schur decomp to find the harmonic Ritz values
  F = schurfact(MA, M)

  # Move the smallest harmonic Ritz value up front
  schur_sort!(F, SM(), 1)

  # Pre-ritz vector
  y = view(F.Z, :, 1)

  # Ritz vector
  A_mul_B!(schur.curr, V, y)

  # Rayleigh quotient = approx eigenvalue s.t. if r = (A-τI)u - rayleigh * u, then r ⟂ u
  rayleigh = conj(F.beta[1]) * F.alpha[1]
  
  # Residual r = (A - τI)u - rayleigh * u = AV*y - rayleigh * u
  A_mul_B!(r, AV, y)
  axpy!(-rayleigh, u, r)
  
  # Orthogonalize w.r.t. Q
  just_orthogonalize!(Q, r, DGKS)
  
  # Assert that the residual is perpendicular to the Ritz vector
  # @assert abs(dot(r, u)) < ɛ
  F, λ + τ, r
end

