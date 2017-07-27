export jdqr_harmonic

import Base.LinAlg.GeneralizedSchur

"""
`Q` is the pre-allocated work space of Schur vector
`R` will be the upper triangular factor

`Q` has the form [q₁, ..., qₖ, qₖ₊₁, qₖ₊₂, ..., q_pairs] where q₁ ... qₖ are already converged
and thus locked, while qₖ₊₁ is the active Schur vector that is converging, and the remaining
columns are just garbage data.

It is very useful to have views for these columns:
`locked` is the matrix view [q₁, ..., qₖ]
`active` is the column vector qₖ₊₁
`all` is the matrix view of the non-garbage part [q₁, ..., qₖ, qₖ₊₁]
`locked` is the number of locked Schur vectors
"""
mutable struct PartialSchur{matT <: StridedMatrix, matViewT <: StridedMatrix, vecViewT <: StridedVector}
    Q::matT
    R::matT
    
    locked::matViewT
    active::vecViewT
    all::matViewT

    num_locked::Int
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
    schur.num_locked += 1
    schur.locked = view(schur.Q, :, 1 : schur.num_locked)

    # Don't extend beyond the max number of Schur vectors
    if schur.num_locked < size(schur.Q, 2)
        schur.all = view(schur.Q, :, 1 : schur.num_locked + 1)
        schur.active = view(schur.Q, :, schur.num_locked + 1)
    end

    schur
end

"""
Useful for a basis for the search and test subspace.
"""
mutable struct SubSpace{matT <: StridedMatrix, matViewT <: StridedMatrix, vecViewT <: StridedVector}
    basis::matT
    all::matViewT
    curr::vecViewT
    prev::matViewT
end

SubSpace(V::AbstractMatrix) = SubSpace(V, view(V, :, 1 : 1), view(V, :, 1), view(V, :, 1 : 0))

function resize!(V::SubSpace, size::Int)
    V.prev = view(V.basis, :, 1 : size - 1)
    V.all = view(V.basis, :, 1 : size)
    V.curr = view(V.basis, :, size)

    V
end

function jdqr_harmonic(
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
) where {Alg <: CorrectionSolver}

    residuals::Vector{real(T)} = []

    n = size(A, 1)

    # V's vectors span the search space
    # AV will store (A - tI) * V, without any orthogonalization
    # W will hold AV, but with its columns orthogonal: AV = W * MA
    V = SubSpace(zeros(T, n, max_dimension))
    AV = SubSpace(zeros(T, n, max_dimension))
    W = SubSpace(zeros(T, n, max_dimension))

    # Temporaries (trying to reduces #allocations here)
    temporary = zeros(T, n, max_dimension)

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

    while k ≤ pairs && iter ≤ max_iter
        m += 1

        resize!(V, m)
        resize!(AV, m)
        resize!(W, m)

        if iter == 1
            rand!(V.curr)
        else
            solve_deflated_correction!(V.curr, solver, A, λ, schur.all, r)
        end

        # Search space is orthonormalized
        orthogonalize_and_normalize!(V.prev, V.curr, zeros(T, m - 1), DGKS)

        # AV is just the product (A - τI)V
        A_mul_B!(AV.curr, A, V.curr)
        axpy!(-τ, V.curr, AV.curr)

        # Expand W with (A - τI)V, and then orthogonalize
        copy!(W.curr, AV.curr)

        # Orthogonalize w.r.t. the converged Schur vectors
        just_orthogonalize!(schur.locked, W.curr, DGKS)

        # Orthonormalize W[:, m] w.r.t. previous columns of W
        MA[m, m] = orthogonalize_and_normalize!(
            W.prev,
            W.curr, 
            view(MA, 1 : m - 1, m),
            DGKS
        )

        # Update the right-most column and last row of M = W' * V
        M[m, m] = dot(W.curr, V.curr)
        @inbounds for i = 1 : m - 1
            M[i, m] = dot(view(W.basis, :, i), V.curr)
            M[m, i] = dot(W.curr, view(V.basis, :, i))
        end

        # Assert orthogonality of V and W
        # Assert W * MA = (I - QQ') * (A - τI) * V
        # Assert that M = W' * V
        # @assert norm(W[:, 1 : m]' * W[:, 1 : m] - eye(m)) < 1e-12
        # @assert norm(V[:, 1 : m]' * V[:, 1 : m] - eye(m)) < 1e-12
        # @assert norm(M[1 : m, 1 : m] - W[:, 1 : m]' * V[:, 1 : m]) < 1e-12
        # @assert norm(W[:, 1 : m] * MA[1 : m, 1 : m] - AV[:, 1 : m] + (schur.locked * (schur.locked' * AV[:, 1 : m]))) < pairs * ɛ

        search = true

        while search

            F = schurfact(view(MA, 1 : m, 1 : m), view(M, 1 : m, 1 : m))
            λ = extract_harmonic!(F, V.all, AV.all, r, schur, τ)
            resnorm = norm(r)

            # Convergence history of the harmonic Ritz values
            push!(harmonic_ritz_values, F.alpha ./ F.beta)
            push!(converged_ritz_values, diag(view(schur.R, 1 : k, 1 : k)))
            push!(residuals, resnorm)

            verbose && println("Residual = ", resnorm)

            # An Ritz vector is converged
            if resnorm ≤ ɛ
                verbose && println("Found an eigenvalue ", λ)

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
                keep = 2 : m
                shrink!(temporary, V, view(F[:right], :, keep), m - 1)
                shrink!(temporary, AV, view(F[:right], :, keep), m - 1)
                shrink!(temporary, W, view(F[:left], :, keep), m - 1)

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
            verbose && println("Shrinking the search space.")

            # Move min_dimension of the smallest harmonic Ritz values up front
            keep = 1 : min_dimension
            schur_sort!(SM(), F, keep)

            shrink!(temporary, V, view(F[:right], :, keep), min_dimension)
            shrink!(temporary, AV, view(F[:right], :, keep), min_dimension)
            shrink!(temporary, W, view(F[:left], :, keep), min_dimension)

            m = min_dimension
            copy!(view(M, 1 : m, 1 : m), view(F.T, 1 : m, 1 : m))
            copy!(view(MA, 1 : m, 1 : m), view(F.S, 1 : m, 1 : m))
        end

        iter += 1
    end

    schur.Q[:, 1 : k], schur.R[1 : k, 1 : k], harmonic_ritz_values, converged_ritz_values, residuals
end

function shrink!(temporary, subspace::SubSpace, combination::StridedMatrix, dimension)
    tmp = view(temporary, :, 1 : dimension)
    A_mul_B!(tmp, subspace.all, combination)
    copy!(subspace.all, tmp)
    resize!(subspace, dimension)
end

function extract_harmonic!(F::GeneralizedSchur, V, AV, r, schur, τ)
    # Compute the Schur decomp to find the harmonic Ritz values

    # Move the smallest harmonic Ritz value up front
    schur_sort!(SM(), F, 1)

    # Pre-ritz vector
    y = view(F.Z, :, 1)

    # Ritz vector
    A_mul_B!(schur.active, V, y)

    # Rayleigh quotient λ = approx eigenvalue s.t. if r = (A-τI)u - λ * u, then r ⟂ u
    λ = conj(F.beta[1]) * F.alpha[1]

    # Residual r = (A - τI)u - λ * u = AV*y - λ * u
    A_mul_B!(r, AV, y)
    axpy!(-λ, schur.active, r)

    # Orthogonalize w.r.t. Q
    just_orthogonalize!(schur.locked, r, DGKS)

    # Assert that the residual is perpendicular to the Ritz vector
    # @assert abs(dot(r, schur.active)) < 1e-5

    λ + τ
end

