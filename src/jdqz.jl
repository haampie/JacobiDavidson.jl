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
    T::Type = Complex128
)
    k = 0
    m = 0
    n = size(A, 1)

    # Harmonic Petrov values 
    ν = 1 / sqrt(1 + abs2(τ))
    μ = -τ * ν

    # Holds the final partial, generalized Schur decomposition where
    # AQ = ZS and BQ = ZT, with Q, Z orthonormal and S, T upper triangular.
    Q = zeros(T, n, pairs)
    Z = zeros(T, n, pairs)
    S = zeros(T, pairs, pairs)
    T = zeros(T, pairs, pairs)

    # Low-dimensional projections: MA = W'AV, MB = W'BV
    MA = zeros(T, max_dimension, max_dimension)
    MB = zeros(T, max_dimension, max_dimension)

    # Orthonormal search subspace
    V = zeros(T, n, max_dimension)
    
    # Orthonormal test subspace
    W = zeros(T, n, max_dimension)

    # AV will store A*V, without any orthogonalization
    AV = zeros(T, n, max_dimension)

    # BV will store B*V, without any orthogonalization
    BV = zeros(T, n, max_dimension)

    # Idea: after the expansion work only with views
    # for instance V <- view(V, :, 1 : m - 1) and v = view(V, :, m)
    # so that it's less noisy down here.

    # Idea: store approximate Petrov pairs immediately inside Q and Z, rather than u and p
    # there is room for it already and it is necessary in the correction equation anyway.
    while k ≤ pairs && iter ≤ max_iter
        if iter == 1
            view(V, :, m + 1) .= rand(n) # Initialize with a random vector
        elseif iter < 3
            view(V, :, m + 1) .= r # Expand with the residual ~ Arnoldi style
        else
            view(V, :, m + 1) .= solve_deflated_correction(solver, A, rayleigh + τ, view(Q, :, 1 : k), u, r)
        end

        m += 1

        # Orthogonalize V[:, m] against the search subspace
        orthogonalize_and_normalize!(
            view(V, :, 1 : m - 1),
            view(V, :, m + 1),
            zeros(T, m - 1), 
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
        orthogonalize_and_normalize!(view(W, :, 1 : m - 1), w, zeros(T, m - 1), DGKS)

        # Update MA = W' * A * V
        MA[m, m] = dot(view(W, :, m), view(AV, :, m))
        @inbounds for i = 1 : m - 1
            MA[i, m] = dot(view(W, :, i), view(AV, :, m))
            MA[m, i] = dot(view(W, :, m), view(AV, :, i))
        end

        # Update MB = W' * B * V
        MB[m, m] = dot(view(W, :, m), view(BV, :, m))
        @inbounds for i = 1 : m - 1
            M[i, m] = dot(view(W, :, i), view(BV, :, m))
            M[m, i] = dot(view(W, :, m), view(BV, :, i))
        end

        # Extract a Petrov pair = approximate Schur pair.
        F, u, p, r, ã, b̃ = extract_generalized(
            view(MA, 1 : m, 1 : m),
            view(MB, 1 : m, 1 : m),
            view(V, :, 1 : m),
            view(W, :, 1 : m),
            view(AV, :, 1 : m),
            view(BV, :, 1 : m),
            view(Z, :, 1 : k),
            τ
        )

        # Store converged Petrov pairs
        while norm(r) ≤ ε
            println("Found an eigenvalue")

            # Store the eigenvalue
            S[k + 1, k + 1] = ζ
            T[k + 1, k + 1] = η

            # Store the projections
            S[1 : k, k + 1] = ã
            T[1 : k, k + 1] = b̃
            
            # Store the Schur vectors (this will be redundant when storing the approximations here from the start)
            Q[:, k + 1] .= u
            Z[:, k + 1] .= p

            # One more eigenpair converged, yay.
            k += 1

            # Was this the last eigenpair?
            if k == pairs
                return Q, Z, S, T
            end

            # Remove the eigenvalue from the search subspace.
            # Shrink V, W, AV, and BV (can be done in-place; but probably not worth it)
            V[:, 1 : m - 1] = view(V, :, 1 : m) * view(F.right, :, 2 : m)
            AV[:, 1 : m - 1] = view(AV, :, 1 : m) * view(F.right, :, 2 : m)
            BV[:, 1 : m - 1] = view(BV, :, 1 : m) * view(F.right, :, 2 : m)
            W[:, 1 : m - 1] = view(W, :, 1 : m) * view(F.left, :, 2 : m)
            
            # Update the projection matrices M and MA.
            MA[1 : m - 1, 1 : m - 1] .= F.S[2 : m, 2 : m]
            MB[1 : m - 1, 1 : m - 1] .= F.T[2 : m, 2 : m]

            # One vector is removed from the search space.
            m -= 1

            # Extract the next approximate eigenpair.
            # This can be done more efficient as there is no need to recompute
            # the whole Schur decomposition. We already have that and only need
            # to reorder it.
            F, u, p, r, ã, b̃ = extract_generalized(
                view(MA, 1 : m, 1 : m),
                view(MB, 1 : m, 1 : m),
                view(V, :, 1 : m),
                view(W, :, 1 : m),
                view(AV, :, 1 : m),
                view(BV, :, 1 : m),
                view(Z, :, 1 : k),
                τ
            )
        end

        if m == max_dimension
            println("Shrinking the search space.")

            # Move min_dimension of the smallest harmonic Ritz values up front
            smallest = selectperm(abs.(F.alpha] ./ F.beta - τ), 1 : min_dimension)
            p = falses(m)
            p[smallest] = true
            ordschur!(F, p)

            # Shrink V, W, AV
            # Todo: V[:, 1], AV[:, 1], BV[:, 1] and W[:, 1] are already available, no need to recompute
            V[:, 1 : min_dimension] = V * view(F.right, :, 1 : min_dimension)
            AV[:, 1 : min_dimension] = AV * view(F.right, :, 1 : min_dimension)
            BV[:, 1 : min_dimension] = AV * view(F.right, :, 1 : min_dimension)
            W[:, 1 : min_dimension] = W * view(F.left, :, 1 : min_dimension)

            # Shrink the spaces
            m = min_dimension

            # Update M and MA.
            MA[1 : m, 1 : m] .= F.S[1 : m, 1 : m]
            MB[1 : m, 1 : m] .= F.T[1 : m, 1 : m]
        end
    end
end

function extract_generalized(MA, MB, V, W, AV, BV, Z, τ)
    m = size(MA, 1)

    F = schurfact(MA, MB)

    smallest = indmin(abs.(F.alpha ./ F.beta - τ))
    p = falses(m)
    p[smallest] = true
    ordschur!(F, p)

    # MA * F.right = F.left * F.S
    # MB * F.right = F.left * F.T

    # TODO: don't allocate each time.
    # Petrov vector
    u = V * F.right[:, 1]
    p = V * F.left[:, 1]
    Au = AV * F.right[:, 1]
    Bu = BV * F.right[:, 1]

    ζ = F.alpha[1]
    η = F.beta[1]

    # Residual
    # TODO: Make this BLAS.
    r = η * Au - ζ Bu

    ã = Z' * Au
    b̃ = Z' * Ab

    # TODO: make this BLAS.
    r -= Z * (η * ã - ζ * b̃)

    F, u, p, r, ã, b̃
end