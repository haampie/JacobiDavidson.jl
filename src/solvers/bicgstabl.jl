import Base: start, next, done

mutable struct BiCGStabIterable{precT, matT, vecT <: AbstractVector, vect2 <: AbstractVector, smallMatT <: AbstractMatrix, realT <: Real, scalarT <: Number}
    A::matT
    b::vecT
    l::Int

    x::vect2
    r_shadow::vecT
    rs::smallMatT
    us::smallMatT

    max_mv_products::Int
    mv_products::Int
    reltol::realT
    residual::realT

    Pl::precT

    γ::vecT
    ω::scalarT
    σ::scalarT
    M::smallMatT
end

bicgstabl_iterator(A, b, l; kwargs...) = bicgstabl_iterator!(zeros(b), A, b, l; initially_zero = true, kwargs...)

function bicgstabl_iterator!(x, A, b, l::Int = 2;
    Pl = Identity(),
    max_mv_products = min(30, size(A, 1)),
    initially_zero = false,
    tol = sqrt(eps(real(eltype(b))))
)
    T = eltype(b)
    n = size(A, 1)
    mv_products = 0

    # Large vectors.
    r_shadow = rand(T, n)
    rs = Matrix{T}(n, l + 1)
    us = zeros(T, n, l + 1)

    residual = view(rs, :, 1)
    
    # Compute the initial residual rs[:, 1] = b - A * x
    # Avoid computing A * 0.
    if initially_zero
        copy!(residual, b)
    else
        A_mul_B!(residual, A, x)
        @blas! residual -= one(T) * b
        @blas! residual *= -one(T)
        mv_products += 1
    end

    # Apply the left preconditioner
    A_ldiv_B!(Pl, residual)

    γ = zeros(T, l)
    ω = σ = one(T)

    nrm = norm(residual)

    # For the least-squares problem
    M = zeros(T, l + 1, l + 1)

    # Stopping condition based on relative tolerance.
    reltol = nrm * tol

    BiCGStabIterable(A, b, l, x, r_shadow, rs, us,
        max_mv_products, mv_products, reltol, nrm,
        Pl,
        γ, ω, σ, M
    )
end

converged(it::BiCGStabIterable) = it.residual ≤ it.reltol

start(::BiCGStabIterable) = 0

done(it::BiCGStabIterable, iteration::Int) = it.mv_products ≥ it.max_mv_products || converged(it)

function next(it::BiCGStabIterable, iteration::Int)
    T = eltype(it.b)
    L = 2 : it.l + 1

    it.σ = -it.ω * it.σ
    
    ## BiCG part
    for j = 1 : it.l
        ρ = dot(it.r_shadow, view(it.rs, :, j))
        β = ρ / it.σ
        
        # us[:, 1 : j] .= rs[:, 1 : j] - β * us[:, 1 : j]
        for i = 1 : j
            @blas! view(it.us, :, i) *= -β
            @blas! view(it.us, :, i) += one(T) * view(it.rs, :, i)
        end

        # us[:, j + 1] = Pl \ (A * us[:, j])
        next_u = view(it.us, :, j + 1)
        A_mul_B!(next_u, it.A, view(it.us, :, j))
        A_ldiv_B!(it.Pl, next_u)

        it.σ = dot(it.r_shadow, next_u)
        α = ρ / it.σ

        # rs[:, 1 : j] .= rs[:, 1 : j] - α * us[:, 2 : j + 1]
        for i = 1 : j
            @blas! view(it.rs, :, i) -= α * view(it.us, :, i + 1)
        end
        
        # rs[:, j + 1] = Pl \ (A * rs[:, j])
        next_r = view(it.rs, :, j + 1)
        A_mul_B!(next_r, it.A , view(it.rs, :, j))
        A_ldiv_B!(it.Pl, next_r)
        
        # x = x + α * us[:, 1]
        @blas! it.x += α * view(it.us, :, 1)
    end

    # Bookkeeping
    it.mv_products += 2 * it.l

    ## MR part
    
    # M = rs' * rs
    Ac_mul_B!(it.M, it.rs, it.rs)

    # γ = M[L, L] \ M[L, 1] 
    F = lufact!(view(it.M, L, L))
    A_ldiv_B!(it.γ, F, view(it.M, L, 1))

    # This could even be BLAS 3 when combined.
    BLAS.gemv!('N', -one(T), view(it.us, :, L), it.γ, one(T), view(it.us, :, 1))
    BLAS.gemv!('N', one(T), view(it.rs, :, 1 : it.l), it.γ, one(T), it.x)
    BLAS.gemv!('N', -one(T), view(it.rs, :, L), it.γ, one(T), view(it.rs, :, 1))

    it.ω = it.γ[it.l]
    it.residual = norm(view(it.rs, :, 1))

    it.residual, iteration + 1
end
