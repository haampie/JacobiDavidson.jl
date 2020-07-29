export jdqr

import LinearAlgebra.GeneralizedSchur

"""
Compute an approximate partial Schur decomposition of a square matrix A

    partial_schur, _ = jdqr(A, pairs = 5)

Accepts the following keyword arguments:

- `pairs`: number of Schur vectors and values requested
- `subspace_dimensions`: a range `min:max` that determines the size of the search subspace.
  In every outer iteration the search subspace is expanded to the `max` size and then 
  trimmed to the `min` size. When the search subspace is trimmed, the best approximate Schur
  vectors are kept. A large search subspace speeds up convergence, but is computationally 
  more expensive. It is advisable to set `min` a bit larger than `pairs`.
- `max_iter`: maximum number of steps. If the algorithm exceeds `max_iter` steps, it might
  not have converged for all `pairs` of Schur vectors.
- `target`: determines which eigenvalues to find. Options: Near(σ) where σ is a complex
  number in the complex plane near which we want to find eigenvalues; LargestMagnitude(σ), 
  SmallestMagnitude(σ), LargestRealPart(σ), SmallestRealPart(σ), LargestImaginaryPart(σ),
  SmallestImaginaryPart(σ), where σ is a complex number / an educated guess.
- `tolerance`: the accuracy at which Schur vectors are considered converged, based on 
  the residual norm.
- `solver`: An iterative correction equation solver used for the expansion of the search
  subspace.
- `verbosity=0` silent, `verbosity==1` prints a progress bar, `verbosity==2` prints more detailed debug output.
"""
function jdqr(
    A;
    solver::CorrectionSolver = BiCGStabl(size(A, 1), max_mv_products = 10, l = 2),
    pairs::Integer = 5,
    subspace_dimensions::AbstractUnitRange{<:Integer} = 10:20,
    max_iter::Integer = 200,
    target::Target = Near(0.0 + 0im),
    tolerance::Float64 = sqrt(eps(real(eltype(A)))),
    T::Type = ComplexF64,
    verbosity::Number = 0,
    verbose::Bool = false
)
    # `verbose = true` overrides verbosity only when verbosity is not set
    verbosity = verbosity == 0 && verbose ? 2 : verbosity

    solver_reltol = one(real(T))
    residuals::Vector{real(T)} = []

    n = size(A, 1)

    # V's vectors span the search space
    # AV will store (A - tI) * V, without any orthogonalization
    # W will hold AV, but with its columns orthogonal: AV = W * MA
    V = SubSpace(zeros(T, n, last(subspace_dimensions)))
    AV = SubSpace(zeros(T, n, last(subspace_dimensions)))
    W = SubSpace(zeros(T, n, last(subspace_dimensions)))

    # Temporaries (trying to reduces #allocations here)
    temporary = zeros(T, n, last(subspace_dimensions))

    # Projected matrices
    MA = ProjectedMatrix(zeros(T, last(subspace_dimensions), last(subspace_dimensions)))
    M = ProjectedMatrix(zeros(T, last(subspace_dimensions), last(subspace_dimensions)))

    # k is the number of converged eigenpairs
    k = 0

    # m is the current dimension of the search subspace
    m = 0

    pschur = PartialSchur(zeros(T, n, pairs), T)

    # Current eigenvalue
    λ = zero(T)

    # Current residual vector
    r = Vector{T}(undef, n)

    iter = 1

    harmonic_ritz_values::Vector{Vector{T}} = []
    converged_ritz_values::Vector{Vector{T}} = []

    local F::GeneralizedSchur
    local lastλ = T(NaN)
    local resnorm = real(T(Inf))

    progress = verbosity==1 ? Progress(max_iter) : nothing

    while k ≤ pairs && iter ≤ max_iter
        solver_reltol /= 2

        m += 1

        ### Expand
        resize!(V, m)
        resize!(AV, m)
        resize!(W, m)
        resize!(M, m)
        resize!(MA, m)

        if iter == 1
            rand!(V.curr)
        else
            solve_deflated_correction!(solver, A, V.curr, pschur.all, λ, r, solver_reltol)
        end

        # Search space is orthonormalized
        orthogonalize_and_normalize!(V.prev, V.curr, zeros(T, m - 1), DGKS)

        # AV is just the product (A - τI)V
        mul!(AV.curr, A, V.curr)
        axpy!(-target.τ, V.curr, AV.curr)

        # Expand W with (A - τI)V, and then orthogonalize
        copyto!(W.curr, AV.curr)

        # Orthogonalize w.r.t. the converged Schur vectors
        just_orthogonalize!(pschur.locked, W.curr, DGKS)

        # Orthonormalize W[:, m] w.r.t. previous columns of W
        MA.matrix[m, m] = orthogonalize_and_normalize!(W.prev, W.curr, view(MA.matrix, 1 : m - 1, m), DGKS)

        # Update the right-most column and last row of M = W' * V
        M.matrix[m, m] = dot(W.curr, V.curr)
        @inbounds for i = 1 : m - 1
            M.matrix[i, m] = dot(view(W.basis, :, i), V.curr)
            M.matrix[m, i] = dot(W.curr, view(V.basis, :, i))
        end

        # Assert orthogonality of V and W
        # Assert W * MA = (I - QQ') * (A - τI) * V
        # Assert that M = W' * V
        # @assert norm(W.all'W.all - I) < 1e-12
        # @assert norm(V.all'V.all - I) < 1e-12
        # @assert norm(M.curr - W.all'V.all) < 1e-12
        # @assert norm(W.all * MA.curr - AV.all + (pschur.locked * (pschur.locked'AV.all))) < pairs * tolerance

        search = true

        ### Extract
        while search

            F = schur(MA.curr, M.curr)
            λ = extract_harmonic!(F, V, AV, r, pschur, target.τ)
            resnorm = norm(r)

            # Convergence history of the harmonic Ritz values
            push!(harmonic_ritz_values, F.alpha ./ F.beta)
            push!(converged_ritz_values, copy(pschur.values))
            push!(residuals, resnorm)

            verbosity > 1 && println("Residual = ", resnorm)

            # A Ritz vector is converged
            if resnorm ≤ tolerance
                verbosity > 1 && println("Found an eigenvalue ", λ)
                lastλ = λ
                push!(converged_ritz_values[iter], λ)

                push!(pschur.values, λ)
                lock!(pschur)

                k += 1

                # Are we done yet?
                k == pairs && return pschur, harmonic_ritz_values, converged_ritz_values, residuals

                # Now remove this Schur vector from the search space.
                keep = 2 : m
                shrink!(temporary, V, view(F.right, :, keep), m - 1)
                shrink!(temporary, AV, view(F.right, :, keep), m - 1)
                shrink!(temporary, W, view(F.left, :, keep), m - 1)
                shrink!(M, view(F.T, keep, keep), m - 1)
                shrink!(MA, view(F.S, keep, keep), m - 1)

                m -= 1

                # TODO: Can the search space become empty? Probably, but not likely.
                m == 0 && throw(ErrorException("Search space became empty: TODO."))

                solver_reltol = one(real(T))
            else
                search = false
            end
        end

        ### Restart
        if m == last(subspace_dimensions)
            verbosity > 1 && println("Shrinking the search space.")

            # Move min_dimension of the smallest harmonic Ritz values up front
            keep = 1 : first(subspace_dimensions)
            schur_sort!(SmallestMagnitude(0.0+0im), F, keep)

            shrink!(temporary, V, view(F.right, :, keep), first(subspace_dimensions))
            shrink!(temporary, AV, view(F.right, :, keep), first(subspace_dimensions))
            shrink!(temporary, W, view(F.left, :, keep), first(subspace_dimensions))
            shrink!(M, view(F.T, keep, keep), first(subspace_dimensions))
            shrink!(MA, view(F.S, keep, keep), first(subspace_dimensions))
            m = first(subspace_dimensions)
        end

        iter += 1
        isnothing(progress) ||
            ProgressMeter.next!(progress,
                                showvalues=[(:Residual, resnorm),
                                            (:λ, lastλ),
                                            (:Pairs, "$(k)/$(pairs)")])
    end

    pschur, harmonic_ritz_values, converged_ritz_values, residuals
end

function extract_harmonic!(F::GeneralizedSchur, V::SubSpace, AV::SubSpace, r, pschur, τ)
    # Compute the Schur decomp to find the harmonic Ritz values

    # Move the smallest harmonic Ritz value up front
    schur_sort!(SmallestMagnitude(0.0 + 0im), F, 1)

    # Pre-ritz vector
    y = view(F.Z, :, 1)

    # Ritz vector
    mul!(pschur.active, V.all, y)

    # Rayleigh quotient λ = approx eigenvalue s.t. if r = (A-τI)u - λ * u, then r ⟂ u
    λ = conj(F.beta[1]) * F.alpha[1]

    # Residual r = (A - τI)u - λ * u = AV*y - λ * u
    mul!(r, AV.all, y)
    axpy!(-λ, pschur.active, r)

    # Orthogonalize w.r.t. Q
    just_orthogonalize!(pschur.locked, r, DGKS)

    # Assert that the residual is perpendicular to the Ritz vector
    # @assert abs(dot(r, pschur.active)) < 1e-5

    λ + τ
end

