export jdqz

import Base.LinAlg.axpy!

mutable struct QZ_product{matT}
    QZ::matT
    QZ_inv::matT
    LU
end

function jdqz(
    A,
    B,
    solver::Alg;             # Solver for the correction equation
    pairs::Int = 5,          # Number of eigenpairs wanted
    min_dimension::Int = 10, # Minimal search space size
    max_dimension::Int = 20, # Maximal search space size
    max_iter::Int = 200,
    τ::Complex128 = 0.0 + 0im,       # Search target
    ɛ::Float64 = 1e-7,       # Maximum residual norm
    numT::Type = Complex128,
    verbose::Bool = false
) where {Alg <: CorrectionSolver}

    k = 0
    m = 0
    n = size(A, 1)

    residuals = real(numT)[]

    # Harmonic Petrov values 
    ν = 1 / sqrt(1 + abs2(τ))
    μ = -τ * ν

    # Holds the final partial, generalized Schur decomposition where
    # AQ = ZS and BQ = ZT, with Q, Z orthonormal and S, T upper triangular.
    Q = zeros(numT, n, pairs)
    Z = zeros(numT, n, pairs)
    S = zeros(numT, pairs, pairs)
    T = zeros(numT, pairs, pairs)

    # Low-dimensional projections: MA = W'AV, MB = W'BV
    MA = zeros(numT, max_dimension, max_dimension)
    MB = zeros(numT, max_dimension, max_dimension)

    # Orthonormal search subspace
    V = zeros(numT, n, max_dimension)
    
    # Orthonormal test subspace
    W = zeros(numT, n, max_dimension)

    # AV will store A*V, without any orthogonalization
    AV = zeros(numT, n, max_dimension)

    # BV will store B*V, without any orthogonalization
    BV = zeros(numT, n, max_dimension)
    
    large_matrix_tmp = zeros(numT, n, max_dimension)

    # QZ = Q' * Z, which is used in the correction equation. QZ_inv = lufact!(copy!(QZ_inv, QZ))
    QZ = QZ_product(
        zeros(numT, pairs, pairs),
        zeros(numT, pairs, pairs),
        lufact(zeros(0, 0))
    )

    iter = 1

    r = zeros(numT, n)
    spare_vector = zeros(numT, n)

    local ζ::numT
    local η::numT
    local F

    # Idea: after the expansion work only with views
    # for instance V <- view(Vmatrix, :, 1 : m - 1) and v = view(Vmatrix, :, m)
    # so that it's less noisy down here.

    while k ≤ pairs && iter ≤ max_iter
        if iter == 1
            rand!(view(V, :, m + 1)) # Initialize with a random vector
        else
            solve_generalized_correction_equation!(solver, A, B, view(V, :, m + 1), view(Q, :, 1 : k + 1), view(Z, :, 1 : k + 1), QZ, ζ, η, r, spare_vector)
        end

        m += 1

        # Orthogonalize V[:, m] against the search subspace
        orthogonalize_and_normalize!(
            view(V, :, 1 : m - 1),
            view(V, :, m),
            zeros(numT, m - 1),
            DGKS
        )

        # AV = A*V and BV = B*V
        A_mul_B!(view(AV, :, m), A, view(V, :, m))
        A_mul_B!(view(BV, :, m), B, view(V, :, m))

        # w = ν * AV + μ * BV
        w = view(W, :, m)
        copy!(w, view(AV, :, m))
        scale!(w, ν)
        axpy!(μ, view(BV, :, m), w)

        # Orthogonalize w against the Schur vectors Z
        just_orthogonalize!(view(Z, :, 1 : k), w, DGKS)

        # Make w orthonormal w.r.t. W
        orthogonalize_and_normalize!(view(W, :, 1 : m - 1), w, zeros(numT, m - 1), DGKS)

        # Update the last row and column of MA = W' * A * V
        Ac_mul_B!(view(MA, 1 : m, m), view(W, :, 1 : m), view(AV, :, m))
        @inbounds for i = 1 : m - 1
            MA[m, i] = dot(view(W, :, m), view(AV, :, i))
        end

        # Update MB = W' * B * V
        Ac_mul_B!(view(MB, 1 : m, m), view(W, :, 1 : m), view(BV, :, m))
        @inbounds for i = 1 : m - 1
            MB[m, i] = dot(view(W, :, m), view(BV, :, i))
        end

        # Extract a Petrov pair = approximate Schur pair.
        should_extract = true

        while should_extract
            F, ζ, η = extract_generalized!(
                view(MA, 1 : m, 1 : m),
                view(MB, 1 : m, 1 : m),
                view(V, :, 1 : m),
                view(W, :, 1 : m),
                view(AV, :, 1 : m),
                view(BV, :, 1 : m),
                view(Z, :, 1 : k),
                view(Q, :, k + 1), # approximate schur vec
                view(Z, :, k + 1), # other approx schur vec
                r,
                τ,
                spare_vector
            )

            update_qz!(QZ, Q, Z, k + 1)

            resnorm = norm(r)

            push!(residuals, resnorm)
            
            verbose && println("Resnorm = ", resnorm)

            # Store converged Petrov pairs
            if resnorm ≤ ɛ
                verbose && println("Found an eigenvalue: ", ζ / η)

                # Store the eigenvalue (do this in-place?)
                S[k + 1, k + 1] = ζ
                T[k + 1, k + 1] = η

                # One more eigenpair converged, yay.
                k += 1

                # Was this the last eigenpair?
                if k == pairs
                    return Q, Z, S, T, residuals
                end

                # Remove the eigenvalue from the search subspace.
                # Shrink V, W, AV, and BV (can be done in-place; but probably not worth it)
                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(V, :, 1 : m), view(F[:right], :, 2 : m))
                copy!(view(V, :, 1 : m - 1), view(large_matrix_tmp, :, 1 : m - 1))

                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(AV, :, 1 : m), view(F[:right], :, 2 : m))
                copy!(view(AV, :, 1 : m - 1), view(large_matrix_tmp, :, 1 : m - 1))

                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(BV, :, 1 : m), view(F[:right], :, 2 : m))
                copy!(view(BV, :, 1 : m - 1), view(large_matrix_tmp, :, 1 : m - 1))

                A_mul_B!(view(large_matrix_tmp, :, 1 : m - 1), view(W, :, 1 : m), view(F[:left], :, 2 : m))
                copy!(view(W, :, 1 : m - 1), view(large_matrix_tmp, :, 1 : m - 1))
                
                # Update the projection matrices M and MA.
                copy!(view(MA, 1 : m - 1, 1 : m - 1), view(F.S, 2 : m, 2 : m))
                copy!(view(MB, 1 : m - 1, 1 : m - 1), view(F.T, 2 : m, 2 : m))

                # One vector is removed from the search space.
                m -= 1
            else
                should_extract = false
            end
        end

        if m == max_dimension
            verbose && println("Shrinking the search space.")
            push!(residuals, NaN)

            # Move min_dimension of the smallest harmonic Ritz values up front
            smallest = selectperm(abs.(F.alpha ./ F.beta - τ), 1 : min_dimension)
            perm = falses(m)
            perm[smallest] = true
            ordschur!(F, perm)

            # Shrink V, W, AV
            # Todo: V[:, 1], AV[:, 1], BV[:, 1] and W[:, 1] are already available, no need to recompute
            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), V, view(F[:right], :, 1 : min_dimension))
            copy!(view(V, :, 1 : min_dimension), view(large_matrix_tmp, :, 1 : min_dimension))

            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), AV, view(F[:right], :, 1 : min_dimension))
            copy!(view(AV, :, 1 : min_dimension), view(large_matrix_tmp, :, 1 : min_dimension))

            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), BV, view(F[:right], :, 1 : min_dimension))
            copy!(view(BV, :, 1 : min_dimension), view(large_matrix_tmp, :, 1 : min_dimension))

            A_mul_B!(view(large_matrix_tmp, :, 1 : min_dimension), W, view(F[:left], :, 1 : min_dimension))
            copy!(view(BV, :, 1 : min_dimension), view(large_matrix_tmp, :, 1 : min_dimension))

            # Shrink the spaces
            m = min_dimension

            # Update M and MA.
            copy!(view(MA, 1 : m, 1 : m), view(F.S, 1 : m, 1 : m))
            copy!(view(MB, 1 : m, 1 : m), view(F.T, 1 : m, 1 : m))
        end

        iter += 1
    end

    return Q[:, 1 : k], Z[:, 1 : k], S[1 : k, 1 : k], T[1 : k, 1 : k], residuals
end

function extract_generalized!(MA::StridedMatrix{T}, MB, V, W, AV, BV, Z, u, p, r, τ, spare_vector) where {T}
    m = size(MA, 1)

    F = schurfact(MA, MB)

    smallest = indmin(abs.(F.alpha ./ F.beta - τ))
    perm = falses(m)
    perm[smallest] = true
    ordschur!(F, perm)

    # MA * F[:right] = F[:left] * F.S
    # MB * F[:right] = F[:left] * F.T

    # Extract Petrov vector u = V * F[:right][:, 1]
    right_eigenvec = view(F[:right], :, 1)
    left_eigenvec = view(F[:left], :, 1)

    A_mul_B!(u, V, right_eigenvec)
    A_mul_B!(p, W, left_eigenvec)

    ζ = F.alpha[1]
    η = F.beta[1]

    # Residual
    # r = η * Au - ζ * Bu
    A_mul_B!(r, AV, right_eigenvec)
    A_mul_B!(spare_vector, BV, right_eigenvec)
    scale!(r, η)
    axpy!(-ζ, spare_vector, r)

    # r <- (I - ZZ')r
    just_orthogonalize!(Z, r, DGKS)

    F, ζ, η
end

"""
Updates the last row and column of QZ = Q' * Z
and computes the LU factorization of QZ
"""
function update_qz!(QZ, Q, Z, k)
    for i = 1 : k
        QZ.QZ[i, k] = dot(view(Q, :, i), view(Z, :, k))
        QZ.QZ[k, i] = dot(view(Q, :, k), view(Z, :, i))
    end
    copy!(view(QZ.QZ_inv, 1 : k, 1 : k), view(QZ.QZ, 1 : k, 1 : k))
    QZ.LU = lufact!(view(QZ.QZ_inv, 1 : k, 1 : k))
end