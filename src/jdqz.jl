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
    preconditioner = Identity(),
    pairs::Int = 5,          # Number of eigenpairs wanted
    min_dimension::Int = 10, # Minimal search space size
    max_dimension::Int = 20, # Maximal search space size
    max_iter::Int = 200,
    target::Target = LM(),       # Search target
    ɛ::Float64 = 1e-7,       # Maximum residual norm
    numT::Type = Complex128,
    verbose::Bool = false
) where {Alg <: CorrectionSolver}

    solver_reltol = one(real(numT))

    k = 0
    m = 0
    n = size(A, 1)

    residuals = real(numT)[]

    # Harmonic Petrov values
    τ = isa(target, Near) ? target.τ : rand(numT)
    
    ν = 1 / sqrt(1 + abs2(τ))
    μ = -τ * ν

    # Holds the final partial, generalized Schur decomposition where
    # AQ = ZS and BQ = ZT, with Q, Z orthonormal and S, T upper triangular.
    schur = PartialGeneralizedSchur(zeros(numT, n, pairs), zeros(numT, n, pairs), numT)

    # Preconditioned Z
    precZ = SubSpace(zeros(numT, n, pairs))

    # Low-dimensional projections: MA = W'AV, MB = W'BV
    MA = ProjectedMatrix(zeros(numT, max_dimension, max_dimension))
    MB = ProjectedMatrix(zeros(numT, max_dimension, max_dimension))

    # Orthonormal search subspace
    # Orthonormal test subspace
    # AV will store A*V, without any orthogonalization
    # BV will store B*V, without any orthogonalization
    # large_matrix_tmp is just a temporary
    V = SubSpace(zeros(numT, n, max_dimension))
    W = SubSpace(zeros(numT, n, max_dimension))
    AV = SubSpace(zeros(numT, n, max_dimension))
    BV = SubSpace(zeros(numT, n, max_dimension))
    large_matrix_tmp = zeros(numT, n, max_dimension)

    # QZ = Q' * (preconditioner \ Z), which is used in the correction equation. 
    # QZ_inv = lufact!(copy!(QZ_inv, QZ))
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
        solver_reltol /= 2

        m += 1

        resize!(V, m)
        resize!(AV, m)
        resize!(BV, m)
        resize!(W, m)
        resize!(MA, m)
        resize!(MB, m)

        if iter == 1
            rand!(V.curr) # Initialize with a random vector
        else
            solve_generalized_correction_equation!(solver, A, B, preconditioner, V.curr, schur.Q.all, schur.Z.all, precZ.all, QZ, ζ, η, r, spare_vector, solver_reltol)
        end

        m += 1

        # Orthogonalize V[:, m] against the search subspace
        orthogonalize_and_normalize!(V.prev, V.curr, zeros(numT, m - 1), DGKS)

        # AV = A*V and BV = B*V
        A_mul_B!(AV.curr, A, V.curr)
        A_mul_B!(BV.curr, B, V.curr)

        # W.curr = ν * AV + μ * BV
        copy!(W.curr, AV.curr)
        scale!(W.curr, ν)
        axpy!(μ, BV.curr, w)

        # Orthogonalize w against the Schur vectors Z
        just_orthogonalize!(schur.Z.prev, W.curr, DGKS)

        # Make W.curr orthonormal w.r.t. W.prev
        orthogonalize_and_normalize!(W.prev, W.curr, zeros(numT, m - 1), DGKS)

        # Update MA = W' * A * V
        # Update MB = W' * B * V
        MA.matrix[m, m] = dot(W.curr, AV.curr)
        MB.matrix[m, m] = dot(W.curr, BV.curr)
        @inbounds for i = 1 : m - 1
            MA.matrix[i, m] = dot(view(W.basis, :, i), AV.curr)
            MA.matrix[m, i] = dot(W.curr, view(AV.basis, :, i))
            MB.matrix[i, m] = dot(view(W.basis, :, i), BV.curr)
            MB.matrix[m, i] = dot(W.curr, view(BV.basis, :, i))
        end

        # Extract a Petrov pair = approximate Schur pair.
        should_extract = true

        while should_extract
            F = schurfact(MA.curr, MB.curr)
            
            ζ, η = extract_generalized!(F, V, W,
                AV,
                BV,
                schur,
                r,
                target,
                spare_vector
            )

            # Update last column of precZ = preconditioner \ Z
            copy!(precZ.curr, schur.Z.curr)
            A_ldiv_B!(preconditioner, precZ.curr)

            # Update the product Q' * precZ
            update_qz!(QZ, Q, precZ, k + 1)

            resnorm = norm(r)

            push!(residuals, resnorm)
            
            verbose && println("Resnorm = ", resnorm)

            # Store converged Petrov pairs
            if resnorm ≤ ɛ
                verbose && println("Found an eigenvalue: ", ζ / η)

                # Store the eigenvalue (do this in-place?)
                push!(schur.alphas, ζ)
                push!(schur.betas, η)

                # One more eigenpair converged, yay.
                k += 1

                # Was this the last eigenpair?
                if k == pairs
                    return schur, residuals
                end

                resize!(schur, k)

                # Reset the iterative solver tolerance
                solver_reltol = one(real(numT))

                # Remove the eigenvalue from the search subspace.
                # Shrink V, W, AV, and BV
                keep = 2 : m
                shrink!(large_matrix_tmp, V, view(F[:right], :, keep), m - 1)
                shrink!(large_matrix_tmp, AV, view(F[:right], :, keep), m - 1)
                shrink!(large_matrix_tmp, BV, view(F[:right], :, keep), m - 1)
                shrink!(large_matrix_tmp, W, view(F[:left], :, keep), m - 1)

                # Update the projection matrices M and MA.
                shrink!(MA, view(F.S, keep, keep), m - 1)
                shrink!(MB, view(F.T, keep, keep), m - 1)
                m -= 1
            else
                should_extract = false
            end
        end

        if m == max_dimension
            verbose && println("Shrinking the search space.")
            push!(residuals, NaN)

            keep = 1 : min_dimension

            # Move min_dimension of the smallest harmonic Ritz values up front
            schur_sort!(target, F, keep)

            # Shrink V, W, AV
            # Potential optimization: V[:, 1], AV[:, 1], BV[:, 1] and W[:, 1] are already available, 
            # no need to recompute
            shrink!(temporary, V, view(F[:right], :, keep), min_dimension)
            shrink!(temporary, AV, view(F[:right], :, keep), min_dimension)
            shrink!(temporary, BV, view(F[:right], :, keep), min_dimension)
            shrink!(temporary, W, view(F[:left], :, keep), min_dimension).
            shrink!(MA, view(F.S, keep, keep), min_dimension)
            shrink!(MB, view(F.T, keep, keep), min_dimension)
            m = min_dimension
        end

        iter += 1
    end

    return schur, residuals
end

function extract_generalized!(F, V, W, AV, BV, schur, r, target, spare_vector)
    schur_sort!(target, F, 1)

    # MA * F[:right] = F[:left] * F.S
    # MB * F[:right] = F[:left] * F.T

    # Extract Petrov vector u = V * F[:right][:, 1]
    right_eigenvec = view(F[:right], :, 1)
    left_eigenvec = view(F[:left], :, 1)

    A_mul_B!(schur.Q.curr, V.all, right_eigenvec)
    A_mul_B!(schur.Z.curr, W.all, left_eigenvec)

    ζ = F.alpha[1]
    η = F.beta[1]

    # Residual
    # r = η * Au - ζ * Bu
    A_mul_B!(r, AV.all, right_eigenvec)
    A_mul_B!(spare_vector, BV.all, right_eigenvec)
    scale!(r, η)
    axpy!(-ζ, spare_vector, r)

    # r <- (I - ZZ')r
    just_orthogonalize!(schur.Z.prev, r, DGKS)

    ζ, η
end

"""
Updates the last row and column of QZ = Q' * Z
and computes the LU factorization of QZ. Z can
be preconditioned.
"""
function update_qz!(QZ, Q, Z, k::Int)
    QZ.QZ[k, k] = dot(Q.curr, Z.curr)
    @inbounds for i = 1 : k - 1
        QZ.QZ[i, k] = dot(view(Q.basis, :, i), Z.curr)
        QZ.QZ[k, i] = dot(Q.curr, view(Z.basis, :, i))
    end
    copy!(view(QZ.QZ_inv, 1 : k, 1 : k), view(QZ.QZ, 1 : k, 1 : k))
    QZ.LU = lufact!(view(QZ.QZ_inv, 1 : k, 1 : k))
end