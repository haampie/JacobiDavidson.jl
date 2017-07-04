export jdqz

function jdqz{Alg <: CorrectionSolver}(
    A,
    B,
    solver::Alg;             # Solver for the correction equation
    pairs::Int = 5,          # Number of eigenpairs wanted
    min_dimension::Int = 10, # Minimal search space size
    max_dimension::Int = 20, # Maximal search space size
    max_iter::Int = 200,
    τ::Complex128 = 0.0 + 0im,       # Search target
    ɛ::Float64 = 1e-7,       # Maximum residual norm
    numT::Type = Complex128
)
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

    iter = 1

    r = zeros(numT, n)

    # Idea: after the expansion work only with views
    # for instance V <- view(Vmatrix, :, 1 : m - 1) and v = view(Vmatrix, :, m)
    # so that it's less noisy down here.

    # Idea: store approximate Petrov pairs immediately inside Q and Z, rather than u and p
    # there is room for it already and it is necessary in the correction equation anyway.
    while k ≤ pairs && iter ≤ max_iter
        if iter == 1
            view(V, :, m + 1) .= rand(n) # Initialize with a random vector
        else
            view(V, :, m + 1) .= solve_generalized_correction_equation(solver, A, B, view(Q, :, 1 : k + 1), view(Z, :, 1 : k + 1), ζ, η, r)
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
        F, ã, b̃, ζ, η = extract_generalized!(
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
            τ
        )

        push!(residuals, norm(r))

        # Store converged Petrov pairs
        while norm(r) ≤ ɛ
            println("Found an eigenvalue: ", ζ / η)

            # Store the eigenvalue
            S[k + 1, k + 1] = ζ
            T[k + 1, k + 1] = η

            # Store the projections
            S[1 : k, k + 1] = ã
            T[1 : k, k + 1] = b̃

            # One more eigenpair converged, yay.
            k += 1

            # Was this the last eigenpair?
            if k == pairs
                return Q, Z, S, T, residuals
            end

            # Remove the eigenvalue from the search subspace.
            # Shrink V, W, AV, and BV (can be done in-place; but probably not worth it)
            V[:, 1 : m - 1] = view(V, :, 1 : m) * view(F[:right], :, 2 : m)
            AV[:, 1 : m - 1] = view(AV, :, 1 : m) * view(F[:right], :, 2 : m)
            BV[:, 1 : m - 1] = view(BV, :, 1 : m) * view(F[:right], :, 2 : m)
            W[:, 1 : m - 1] = view(W, :, 1 : m) * view(F[:left], :, 2 : m)
            
            # Update the projection matrices M and MA.
            MA[1 : m - 1, 1 : m - 1] .= F.S[2 : m, 2 : m]
            MB[1 : m - 1, 1 : m - 1] .= F.T[2 : m, 2 : m]

            # One vector is removed from the search space.
            m -= 1

            # Extract the next approximate eigenpair.
            # This can be done more efficient as there is no need to recompute
            # the whole Schur decomposition. We already have that and only need
            # to reorder it.
            F, ã, b̃, ζ, η = extract_generalized!(
                view(MA, 1 : m, 1 : m),
                view(MB, 1 : m, 1 : m),
                view(V, :, 1 : m),
                view(W, :, 1 : m),
                view(AV, :, 1 : m),
                view(BV, :, 1 : m),
                view(Z, :, 1 : k),
                view(Q, :, k + 1),
                view(Z, :, k + 1),
                r,
                τ
            )
        end

        if m == max_dimension
            println("Shrinking the search space.")
            push!(residuals, NaN)

            # Move min_dimension of the smallest harmonic Ritz values up front
            smallest = selectperm(abs.(F.alpha ./ F.beta - τ), 1 : min_dimension)
            perm = falses(m)
            perm[smallest] = true
            ordschur!(F, perm)

            # Shrink V, W, AV
            # Todo: V[:, 1], AV[:, 1], BV[:, 1] and W[:, 1] are already available, no need to recompute
            V[:, 1 : min_dimension] = V * view(F[:right], :, 1 : min_dimension)
            AV[:, 1 : min_dimension] = AV * view(F[:right], :, 1 : min_dimension)
            BV[:, 1 : min_dimension] = BV * view(F[:right], :, 1 : min_dimension)
            W[:, 1 : min_dimension] = W * view(F[:left], :, 1 : min_dimension)

            # Shrink the spaces
            m = min_dimension

            # Update M and MA.
            MA[1 : m, 1 : m] .= F.S[1 : m, 1 : m]
            MB[1 : m, 1 : m] .= F.T[1 : m, 1 : m]
        end

        iter += 1
    end

    return Q[:, 1 : k], Z[:, 1 : k], S[1 : k, 1 : k], T[1 : k, 1 : k], residuals
end

function extract_generalized!{T}(MA::StridedMatrix{T}, MB, V, W, AV, BV, Z, u, p, r, τ)
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

    # Make this non-allocating
    Au = AV * right_eigenvec
    Bu = BV * right_eigenvec

    ζ = F.alpha[1]
    η = F.beta[1]

    # Residual
    # r = η * Au - ζ * Bu
    copy!(r, Au)
    scale!(r, η)
    axpy!(-ζ, Bu, r)

    # This could be stored in-place as well.
    ã = Z' * Au
    b̃ = Z' * Bu

    # r -= Z * (η * ã - ζ * b̃)
    BLAS.gemv!('N', -η, Z, ã, one(T), r)
    BLAS.gemv!('N', ζ, Z, b̃, one(T), r)

    F, ã, b̃, ζ, η
end