export jdqr_harmonic

import Base.LinAlg.GeneralizedSchur

"""
`Q` is the pre-allocated work space of Schur vector
`R` will be the upper triangular factor

`Q` has the form [q₁, ..., qₖ, qₖ₊₁, qₖ₊₂, ..., q_pairs] where q₁ ... qₖ are already converged
and thus locked, while qₖ₊₁ is the active Schur vector that is converging, and the remaining
columns are just garbage data.

It is very useful to have views for these columns:
`Q_locked` is the matrix view [q₁, ..., qₖ]
`Q_active` is the column vector qₖ₊₁
`Q_all` is the matrix view of the non-garbage part [q₁, ..., qₖ, qₖ₊₁]
`locked` is the number of locked Schur vectors
"""
type PartialSchur{matT <: StridedMatrix, matViewT <: StridedMatrix, vecViewT <: StridedVector}
    Q::matT
    R::matT
    
    Q_locked::matViewT
    Q_active::vecViewT
    Q_all::matViewT

    locked::Int
end

PartialSchur(Q, R) = PartialSchur(
    Q,
    R,
    view(Q, :, 1 : 0), # Empty view initially
    view(Q, :, 1),
    view(Q, :, 1 : 1),
    0
)

function lock!(schur::PartialSchur)
    schur.locked += 1
    schur.Q_locked = view(schur.Q_all, :, 1 : schur.locked)

    # Don't extend beyond the max number of Schur vectors
    if schur.locked < size(schur.Q_all, 2)
        schur.Q_all = view(schur.Q_all, :, 1 : schur.locked + 1)
        schur.Q_active = view(schur.Q_all, :, schur.locked + 1)
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
    T::Type = Complex128,
    verbose::Bool = false
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
    λ = zero(T)

    # Current residual vector
    r = Vector{T}(n)

    iter = 1

    harmonic_ritz_values::Vector{Vector{T}} = []
    converged_ritz_values::Vector{Vector{T}} = []

    local F::GeneralizedSchur

    while k <= pairs && iter <= max_iter
        m += 1

        if iter == 1
            rand!(view(V, :, 1))
        else
            solve_deflated_correction!(view(V, :, m), solver, A, λ, schur.Q_all, r)
        end

        # Search space is orthonormalized
        orthogonalize_and_normalize!(view(V, :, 1 : m - 1), view(V, :, m), zeros(T, m - 1), DGKS)

        # AV is just the product (A - τI)V
        A_mul_B!(view(AV, :, m), A, view(V, :, m))
        axpy!(-τ, view(V, :, m), view(AV, :, m))

        # Expand W with (A - τI)V, and then orthogonalize
        copy!(view(W, :, m), view(AV, :, m))

        # Orthogonalize w.r.t. the converged Schur vectors
        just_orthogonalize!(schur.Q_locked, view(W, :, m), DGKS)

        # Orthonormalize W[:, m] w.r.t. previous columns of W
        MA[m, m] = orthogonalize_and_normalize!(
            view(W, :, 1 : m - 1),
            view(W, :, m), 
            view(MA, 1 : m - 1, m),
            DGKS
        )

        # Update the right-most column and last row of M = W' * V
        M[m, m] = dot(view(W, :, m), view(V, :, m))
        @inbounds for i = 1 : m - 1
            M[i, m] = dot(view(W, :, i), view(V, :, m))
            M[m, i] = dot(view(W, :, m), view(V, :, i))
        end

        # Assert orthogonality of V and W
        # Assert W * MA = (I - QQ') * (A - τI) * V
        # Assert that M = W' * V
        @assert norm(W[:, 1 : m]' * W[:, 1 : m] - eye(m)) < 1e-12
        @assert norm(V[:, 1 : m]' * V[:, 1 : m] - eye(m)) < 1e-12
        @assert norm(W[:, 1 : m] * MA[1 : m, 1 : m] - AV[:, 1 : m] + (schur.Q_locked * (schur.Q_locked' * AV[:, 1 : m]))) < pairs * ɛ
        @assert norm(M[1 : m, 1 : m] - W[:, 1 : m]' * V[:, 1 : m]) < 1e-12

        search = true

        while search

            F = schurfact(view(MA, 1 : m, 1 : m), view(M, 1 : m, 1 : m))
            λ = extract_harmonic!(F, view(V, :, 1 : m), view(AV, :, 1 : m), r, schur, τ)
            resnorm = norm(r)

            # Convergence history of the harmonic Ritz values
            push!(harmonic_ritz_values, F.alpha ./ F.beta)
            push!(converged_ritz_values, diag(view(schur.R, 1 : k, 1 : k)))
            push!(residuals, norm(r))

            # An Ritz vector is converged
            if resnorm ≤ ɛ
                println("Found an eigenvalue ", λ)

                schur.R[k + 1, k + 1] = λ # Eigenvalue
                k += 1

                lock!(schur)

                # Add another one in the history
                push!(converged_ritz_values[iter], λ)

                # Are we done yet?
                if k == pairs
                    return schur.Q, schur.R, harmonic_ritz_values, converged_ritz_values, residuals
                end

                # Now remove this Schur vector from the search space.

                # Shrink V, W and AV
                shrink!(large_matrix_tmp, view(V, :, 1 : m), view(AV, :, 1 : m), view(W, :, 1 : m), F, 2 : m)

                # Update the projection matrices M and MA.
                copy!(view(M, 1 : m - 1, 1 : m - 1), view(F.T, 2 : m, 2 : m))
                copy!(view(MA, 1 : m - 1, 1 : m - 1), view(F.S, 2 : m, 2 : m))

                m -= 1

                # TODO: Can the search space become empty? Probably, but not likely.
                if m == 0
                    return schur.Q[:, 1 : k], schur.R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals
                end
            else
                search = false
            end
        end

        if m == max_dimension
            println("Shrinking the search space.")

            # Move min_dimension of the smallest harmonic Ritz values up front
            keep = 1 : min_dimension
            schur_sort!(SM(), F, keep)
            shrink!(large_matrix_tmp, V, AV, W, F, keep)

            m = min_dimension
            copy!(view(M, 1 : m, 1 : m), view(F.T, 1 : m, 1 : m))
            copy!(view(MA, 1 : m, 1 : m), view(F.S, 1 : m, 1 : m))
        end

        iter += 1
    end

    schur.Q[:, 1 : k], schur.R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals
end

function shrink!(tmp, V, AV, W, F, range)
    # Shrink V, W, AV, and update M and MA.
    temporary = view(tmp, :, 1 : length(range))
    
    A_mul_B!(temporary, V, view(F[:right], :, range))
    copy!(V, temporary)
    
    A_mul_B!(temporary, AV, view(F[:right], :, range))
    copy!(AV, temporary)

    A_mul_B!(temporary, W, view(F[:left], :, range))
    copy!(W, temporary)
end

function extract_harmonic!(F::GeneralizedSchur, V, AV, r, schur, τ)
    # Compute the Schur decomp to find the harmonic Ritz values

    # Move the smallest harmonic Ritz value up front
    schur_sort!(SM(), F, 1)

    # Pre-ritz vector
    y = view(F.Z, :, 1)

    # Ritz vector
    A_mul_B!(schur.Q_active, V, y)

    # Rayleigh quotient λ = approx eigenvalue s.t. if r = (A-τI)u - λ * u, then r ⟂ u
    λ = conj(F.beta[1]) * F.alpha[1]

    # Residual r = (A - τI)u - λ * u = AV*y - λ * u
    A_mul_B!(r, AV, y)
    axpy!(-λ, schur.Q_active, r)

    # Orthogonalize w.r.t. Q
    just_orthogonalize!(schur.Q_locked, r, DGKS)

    # Assert that the residual is perpendicular to the Ritz vector
    # @assert abs(dot(r, schur.Q_active)) < 1e-5

    λ + τ
end

