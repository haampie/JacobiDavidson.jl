export jdqz, Petrov, VariablePetrov, Harmonic

import LinearAlgebra.axpy!
import LinearAlgebra.GeneralizedSchur

mutable struct QZ_product{matT}
    QZ::matT
    QZ_inv::matT
    LU
end

abstract type TestSubspace end

struct Petrov <: TestSubspace end
struct VariablePetrov <: TestSubspace end
struct Harmonic <: TestSubspace end

function jdqz(
    A,
    B,
    solver::Alg;             # Solver for the correction equation
    preconditioner = Identity(),
    testspace::Type{<:TestSubspace} = Harmonic,
    pairs::Int = 5,          # Number of eigenpairs wanted
    min_dimension::Int = 10, # Minimal search space size
    max_dimension::Int = 20, # Maximal search space size
    max_iter::Int = 200,
    target::Target = LM(-1.0 + 0.0im),       # Search target
    ɛ::Float64 = 1e-7,       # Maximum residual norm
    numT::Type = ComplexF64,
    verbose::Bool = false
) where {Alg <: CorrectionSolver}

    solver_reltol = one(real(numT))
    residuals = real(numT)[]

    n = size(A, 1)
    iter = 1
    k = 0 # Number of eigenvalues found
    m = 0 # Size of search subspace

    if testspace == Harmonic
        γ = sqrt(1 + abs2(target.τ))
        ν = 1 / γ
        μ = -target.τ / γ
    else
        γ = sqrt(1 + abs2(target.τ))
        ν = conj(target.τ) / γ
        μ = 1 / γ
    end

    # Holds the final partial, generalized Schur decomposition where
    # AQ = ZS and BQ = ZT, with Q, Z orthonormal and S, T upper triangular.
    # precZ holds preconditioner \ Z
    pschur = PartialGeneralizedSchur(zeros(numT, n, pairs), zeros(numT, n, pairs), numT)
    precZ = SubSpace(zeros(numT, n, pairs))

    # Orthonormal search subspace
    # Orthonormal test subspace
    # AV will store A*V, without any orthogonalization
    # BV will store B*V, without any orthogonalization
    # temporary is just a temporary
    V = SubSpace(zeros(numT, n, max_dimension))
    W = SubSpace(zeros(numT, n, max_dimension))
    AV = SubSpace(zeros(numT, n, max_dimension))
    BV = SubSpace(zeros(numT, n, max_dimension))
    temporary = zeros(numT, n, max_dimension)

    # Low-dimensional projections: MA = W'AV, MB = W'BV
    MA = ProjectedMatrix(zeros(numT, max_dimension, max_dimension))
    MB = ProjectedMatrix(zeros(numT, max_dimension, max_dimension))

    # QZ = Q' * (preconditioner \ Z), which is used in the correction equation. 
    # QZ_inv = lufact!(copyto!(QZ_inv, QZ))
    QZ = QZ_product(
        zeros(numT, pairs, pairs),
        zeros(numT, pairs, pairs),
        lu(zeros(0, 0))
    )

    # Residual
    r = zeros(numT, n)
    spare_vector = zeros(numT, n)

    local α::numT
    local β::numT
    local F::GeneralizedSchur

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
            solve_generalized_correction_equation!(solver, A, B, preconditioner, V.curr, pschur.Q.all, pschur.Z.all, precZ.all, QZ, α, β, r, spare_vector, solver_reltol)
        end

        # Orthogonalize V[:, m] against the search subspace
        orthogonalize_and_normalize!(V.prev, V.curr, zeros(numT, m - 1), DGKS)

        # AV = A*V and BV = B*V
        mul!(AV.curr, A, V.curr)
        mul!(BV.curr, B, V.curr)

        # W.curr = ν * AV + μ * BV
        copyto!(W.curr, AV.curr)
        lmul!(ν, W.curr)
        axpy!(μ, BV.curr, W.curr)

        # Orthogonalize W.curr against the Schur vectors Z
        just_orthogonalize!(pschur.Z.prev, W.curr, DGKS)

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
            F = schur(MA.curr, MB.curr)
            
            α, β = extract_generalized!(F, V, W, AV, BV, pschur, r, target, spare_vector)

            # Update last column of precZ = preconditioner \ Z
            ldiv!(precZ.curr, preconditioner, pschur.Z.curr)

            # Update the product Q' * precZ
            update_qz!(QZ, pschur.Q, precZ, k + 1)

            resnorm = norm(r)

            push!(residuals, resnorm)
            
            verbose && println("Resnorm = ", resnorm)

            # Store converged Petrov pairs
            if resnorm ≤ ɛ
                verbose && println("Found an eigenvalue: ", α / β)

                # Store the eigenvalue (do this in-place?)
                push!(pschur.alphas, α)
                push!(pschur.betas, β)

                # One more eigenpair converged, yay.
                k += 1

                # Was this the last eigenpair?
                if k == pairs
                    return pschur, residuals
                end

                resize!(pschur, k + 1)
                resize!(precZ, k + 1)

                # Reset the iterative solver tolerance
                solver_reltol = one(real(numT))

                # Remove the eigenvalue from the search subspace.
                # Shrink V, W, AV, and BV
                keep = 2 : m
                shrink!(temporary, V, view(F.right, :, keep), m - 1)
                shrink!(temporary, AV, view(F.right, :, keep), m - 1)
                shrink!(temporary, BV, view(F.right, :, keep), m - 1)
                shrink!(temporary, W, view(F.left, :, keep), m - 1)

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

            keep = 1 : min_dimension

            # Move min_dimension of the smallest harmonic Ritz values up front
            schur_sort!(target, F, keep)

            # Shrink V, W, AV
            # Potential optimization: V[:, 1], AV[:, 1], BV[:, 1] and W[:, 1] are already available, 
            # no need to recompute
            shrink!(temporary, V, view(F.right, :, keep), min_dimension)
            shrink!(temporary, AV, view(F.right, :, keep), min_dimension)
            shrink!(temporary, BV, view(F.right, :, keep), min_dimension)
            shrink!(temporary, W, view(F.left, :, keep), min_dimension)
            shrink!(MA, view(F.S, keep, keep), min_dimension)
            shrink!(MB, view(F.T, keep, keep), min_dimension)
            m = min_dimension
        end

        iter += 1

        if testspace == VariablePetrov
            γ = sqrt(abs2(α) + abs2(β))
            ν = conj(α) / γ
            μ = conj(β) / γ
        end
    end

    return pschur, residuals
end

function extract_generalized!(F, V, W, AV, BV, pschur, r, target, spare_vector)
    schur_sort!(target, F, 1)

    # MA * F.right = F.left * F.S
    # MB * F.right = F.left * F.T

    # Extract Petrov vector u = V * F.right[:, 1]
    right_eigenvec = view(F.right, :, 1)
    left_eigenvec = view(F.left, :, 1)

    mul!(pschur.Q.curr, V.all, right_eigenvec)
    mul!(pschur.Z.curr, W.all, left_eigenvec)

    α = F.alpha[1]
    β = F.beta[1]

    # Residual
    # r = β * Au - α * Bu
    mul!(r, AV.all, right_eigenvec, β, false)
    mul!(r, BV.all, right_eigenvec, -α, true)

    # r <- (I - ZZ')r
    just_orthogonalize!(pschur.Z.prev, r, DGKS)

    α, β
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
    copyto!(view(QZ.QZ_inv, 1 : k, 1 : k), view(QZ.QZ, 1 : k, 1 : k))

    # Maybe this can be updated from the previous LU decomp?
    QZ.LU = lu!(view(QZ.QZ_inv, 1 : k, 1 : k))
end
