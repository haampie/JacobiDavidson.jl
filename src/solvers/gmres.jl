import Base: start, next, done
import Base.LinAlg: Givens, givensAlgorithm
import Base.A_ldiv_B!

type ArnoldiDecomp{T, matT}
    A::matT
    V::Matrix{T} # Orthonormal basis vectors
    H::Matrix{T} # Hessenberg matrix
end

ArnoldiDecomp{matT}(A::matT, order::Int, T::Type) = ArnoldiDecomp{T, matT}(
    A,
    zeros(T, size(A, 1), order + 1),
    zeros(T, order + 1, order)
)

type Residual{numT, resT}
    current::resT # Current, absolute, preconditioned residual
    accumulator::resT # Used to compute the residual on the go
    nullvec::Vector{numT} # Vector in the null space of H to compute residuals
    β::resT # the initial residual
end

Residual(order, T::Type) = Residual{T, real(T)}(
    one(real(T)),
    one(real(T)),
    ones(T, order + 1),
    one(real(T))
)

type GMRESIterable{preclT, precrT, vecT <: AbstractVector, arnoldiT <: ArnoldiDecomp, residualT <: Residual, resT <: Real}
    Pl::preclT
    Pr::precrT
    x::vecT
    b::vecT
    Ax::vecT # Some room to work in.

    arnoldi::arnoldiT
    residual::residualT

    mv_products::Int
    maxiter::Int
    reltol::resT
    β::resT
end

converged(g::GMRESIterable) = g.residual.current ≤ g.reltol

start(::GMRESIterable) = 1

done(g::GMRESIterable, iteration::Int) = iteration ≥ g.maxiter || converged(g)

function next(g::GMRESIterable, iteration::Int)

    # Arnoldi step: expand
    expand!(g.arnoldi, g.Pl, g.Pr, iteration, g.Ax)
    g.mv_products += 1

    # Orthogonalize V[:, k + 1] w.r.t. V[:, 1 : k]
    g.arnoldi.H[iteration + 1, iteration] = orthogonalize_and_normalize!(
        view(g.arnoldi.V, :, 1 : iteration),
        view(g.arnoldi.V, :, iteration + 1),
        view(g.arnoldi.H, 1 : iteration, iteration)
    )

    # Implicitly computes the residual
    update_residual!(g.residual, g.arnoldi, iteration)

    # Computation of x only at the end of the iterations.
    if done(g, iteration + 1)

        # Solve the projected problem Hy = β * e1 in the least-squares sense
        rhs = solve_least_squares!(g.arnoldi, g.β, iteration + 1)

        # And improve the solution x ← x + Pr \ (V * y)
        update_solution!(g.x, view(rhs, 1 : iteration), g.arnoldi, g.Pr, iteration + 1, g.Ax)
    end

    g.residual.current, iteration + 1
end

gmres_iterable(A, b; kwargs...) = gmres_iterable!(zeros(b), A, b; initially_zero = true, kwargs...)

function gmres_iterable!(x, A, b;
    Pl = Identity(),
    Pr = Identity(),
    tol = sqrt(eps(real(eltype(b)))),
    maxiter::Int = min(20, size(A, 1)),
    initially_zero = false
)
    T = eltype(b)

    # Approximate solution
    arnoldi = ArnoldiDecomp(A, maxiter, T)
    residual = Residual(maxiter, T)
    mv_products = initially_zero == true ? 1 : 0

    # Workspace vector to reduce the # allocs.
    Ax = similar(b)
    residual.current = init!(arnoldi, x, b, Pl, Ax, initially_zero = initially_zero)
    init_residual!(residual, residual.current)

    reltol = tol * residual.current

    GMRESIterable(Pl, Pr, x, b, Ax, 
        arnoldi, residual, 
        mv_products, maxiter, reltol, residual.current
    )
end

function update_residual!(r::Residual, arnoldi::ArnoldiDecomp, k::Int)
    # Cheaply computes the current residual
    r.nullvec[k + 1] = -conj(dot(view(r.nullvec, 1 : k), view(arnoldi.H, 1 : k, k)) / arnoldi.H[k + 1, k])
    r.accumulator += abs2(r.nullvec[k + 1])
    r.current = r.β / √r.accumulator
end

function init!{T}(arnoldi::ArnoldiDecomp{T}, x, b, Pl, Ax; initially_zero::Bool = false)
    # Initialize the Krylov subspace with the initial residual vector
    # This basically does V[1] = Pl \ (b - A * x) and then normalize
    
    first_col = view(arnoldi.V, :, 1)

    copy!(first_col, b)

    # Potentially save one MV product
    if !initially_zero
        A_mul_B!(Ax, arnoldi.A, x)
        @blas! first_col -= one(T) * Ax
    end

    A_ldiv_B!(Pl, first_col)

    # Normalize
    β = norm(first_col)
    @blas! first_col *= one(T) / β
    β
end

@inline function init_residual!{numT,resT}(r::Residual{numT, resT}, β)
    r.accumulator = one(resT)
    r.β = β
end

function solve_least_squares!{T}(arnoldi::ArnoldiDecomp{T}, β, k::Int)
    # Compute the least-squares solution to Hy = β e1 via Given's rotations
    rhs = zeros(T, k)
    rhs[1] = β

    H = Hessenberg(view(arnoldi.H, 1 : k, 1 : k - 1))
    A_ldiv_B!(H, rhs)

    rhs
end

function update_solution!{T}(x, y, arnoldi::ArnoldiDecomp{T}, Pr::Identity, k::Int, Ax)
    # Update x ← x + V * y

    # TODO: find the SugarBLAS alternative
    BLAS.gemv!('N', one(T), view(arnoldi.V, :, 1 : k - 1), y, one(T), x)
end

function update_solution!{T}(x, y, arnoldi::ArnoldiDecomp{T}, Pr, k::Int, Ax)
    # Computing x ← x + Pr \ (V * y) and use Ax as a work space
    A_mul_B!(Ax, view(arnoldi.V, :, 1 : k - 1), y)
    A_ldiv_B!(Pr, Ax)
    @blas! x += one(T) * Ax
end

function expand!(arnoldi::ArnoldiDecomp, Pl::Identity, Pr::Identity, k::Int, Ax)
    # Simply expands by A * v without allocating
    A_mul_B!(view(arnoldi.V, :, k + 1), arnoldi.A, view(arnoldi.V, :, k))
end

function expand!(arnoldi::ArnoldiDecomp, Pl, Pr::Identity, k::Int, Ax)
    # Expands by Pl \ (A * v) without allocating
    nextV = view(arnoldi.V, :, k + 1)
    A_mul_B!(nextV, arnoldi.A, view(arnoldi.V, :, k))
    A_ldiv_B!(Pl, nextV)
end

function expand!(arnoldi::ArnoldiDecomp, Pl, Pr, k::Int, Ax)
    # Expands by Pl \ (A * (Pr \ v)). Avoids allocation by using Ax.
    nextV = view(arnoldi.V, :, k + 1)
    A_ldiv_B!(nextV, Pr, view(arnoldi.V, :, k))
    A_mul_B!(Ax, arnoldi.A, nextV)
    copy!(nextV,  Ax)
    A_ldiv_B!(Pl, nextV)
end

mutable struct Hessenberg{T<:AbstractMatrix}
    H::T # H is assumed to be Hessenberg of size (m + 1) × m
end

@inline Base.size(H::Hessenberg, args...) = size(H.H, args...)

"""
Solve Hy = rhs for a non-square Hessenberg matrix.
Note that `H` is also modified as is it converted
to an upper triangular matrix via Given's rotations
"""
function A_ldiv_B!(H::Hessenberg, rhs)
    # Implicitly computes H = QR via Given's rotations
    # and then computes the least-squares solution y to
    # |Hy - rhs| = |QRy - rhs| = |Ry - Q'rhs|

    width = size(H, 2)

    # Hessenberg -> UpperTriangular; also apply to r.h.s.
    @inbounds for i = 1 : width
        c, s, _ = givensAlgorithm(H.H[i, i], H.H[i + 1, i])
        
        # Skip the first sub-diagonal since it'll be zero by design.
        H.H[i, i] = c * H.H[i, i] + s * H.H[i + 1, i]

        # Remaining columns
        @inbounds for j = i + 1 : width
            tmp = -conj(s) * H.H[i, j] + c * H.H[i + 1, j]
            H.H[i, j] = c * H.H[i, j] + s * H.H[i + 1, j]
            H.H[i + 1, j] = tmp
        end

        # Right hand side
        tmp = -conj(s) * rhs[i] + c * rhs[i + 1]
        rhs[i] = c * rhs[i] + s * rhs[i + 1]
        rhs[i + 1] = tmp
    end

    # Solve the upper triangular problem.
    U = UpperTriangular(view(H.H, 1 : width, 1 : width))
    A_ldiv_B!(U, view(rhs, 1 : width))
    nothing
end