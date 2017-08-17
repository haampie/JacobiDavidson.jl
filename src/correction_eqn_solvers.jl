import Base.LinAlg.BLAS: axpy!, gemv!

export gmres_solver, bicgstabl_solver, exact_solver

abstract type CorrectionSolver end

struct gmres_solver <: CorrectionSolver
    iterations::Int
    tolerance::Float64
    arnoldi::ArnoldiDecomp
    residual::Residual
    function gmres_solver(n::Int; iterations::Int = 5, tolerance::Float64 = 1e-6)
        new(iterations, tolerance, ArnoldiDecomp(n, iterations, Complex128))
    end
end

struct bicgstabl_solver <: CorrectionSolver
    max_mv_products::Int
    tolerance::Float64
    l::Int
    r_shadow
    rs
    us
    γ
    M

    function bicgstabl_solver(n::Int; l::Int = 2, max_mv_products::Int = 5, tolerance::Float64 = 1e-6)
        r_shadow = rand(Complex128, n)
        rs = Matrix{Complex128}(n, l + 1)
        us = Matrix{Complex128}(n, l + 1)
        γ = Vector{Complex128}(l)
        M = Matrix{Complex128}(l + 1, l + 1)

        new(max_mv_products, tolerance, l, r_shadow, rs, us, γ, M)
    end
end

struct exact_solver <: CorrectionSolver end

function solve_deflated_correction!(solver::exact_solver, A, x, Q, θ, r::AbstractVector, tol)
    # The exact solver is mostly useful for testing Jacobi-Davidson
    # method itself and should result in quadratic convergence.
    # However, in general the correction equation should be solved
    # only approximately in a fixed number of GMRES / BiCGStab
    # iterations to reduce computation costs.

    # Here we solve the augmented system
    # [A - θI, Q; Q' 0][t; y] = [-r; 0] for t,
    # which is equivalent to solving (I - QQ')(A - θI)(I - QQ')t = r
    # for t ⟂ Q.

    n = size(A, 1)
    m = size(Q, 2)
    Ã = [(A - θ * speye(n)) Q; Q' zeros(m, m)]
    rhs = [r; zeros(m, 1)]
    copy!(x, (Ã \ rhs)[1 : n])
end

function solve_deflated_correction!(solver::bicgstabl_solver, A, x, Q, θ, r::AbstractVector, tol)
    n = size(A, 1)
    T = eltype(x)

    # y = (I - QQ')(A - θI)x
    matrix = LinearMap{T}((y, x) -> begin
        A_mul_B!(y, A, x)
        axpy!(-θ, x, y)
        just_orthogonalize!(Q, x, DGKS)
    end, n, ismutating = true)

    mv_products = 0

    residual = view(solver.rs, :, 1)

    # The initial residual rs[:, 1] = b - A * x
    copy!(residual, r)
    fill!(view(solver.us, :, 1), zero(T))
    fill!(x, zero(T))

    ω = σ = one(T)

    just_orthogonalize!(Q, residual, DGKS)
    nrm = norm(residual)

    # Stopping condition based on relative tolerance.
    reltol = nrm * tol

    iterable = BiCGStabIterable(matrix, r, solver.l, x, solver.r_shadow, solver.rs, solver.us,
        solver.max_mv_products, mv_products, reltol, nrm,
        Identity(),
        solver.γ, ω, σ, solver.M
    )

    for res = iterable 
        # println(res)
    end

    nothing
end

function solve_generalized_correction_equation!(solver::exact_solver, A, B, preconditioner, x, Q, Z, precZ, QZ, α, β, r, spare_vector, tol)
  n = size(A, 1)
  m = size(Q, 2)
  # Assuming both A and B are sparse while Q and Z are dense, let's try to avoid constructing a huge dense matrix.
  # Let C = β * A - α * B
 
  # We have to solve:
  # |C  Z| |t| = |-r| 
  # |Q' O| |z|   |0 |

  # Use the Schur complement trick with S = -Q' * inv(C) * Z
  # |C  Z| = |I            O| |C Z|
  # |Q' O|   |Q' * inv(C)  I| |O S|

  # And solve two systems with inv(C) occurring multiple times.
  C = β * A - α * B # Completely sparse
  y = Q' * (C \ r)
  S = Q' * (C \ Z) # Schur complement
  z = -S \ y
  copy!(x, C \ (-r - Z * z))
end

struct QZpreconditioner{T}
    Q
    Z
    QZ
    preconditioner
end

function A_ldiv_B!(QZ::QZpreconditioner{T}, x) where {T}
    # x ← preconditioner \ x
    A_ldiv_B!(QZ.preconditioner, x)
    
    # x ← (I - Z inv(Q'Z) Q')x
    h = Ac_mul_B(QZ.Q, x)
    A_ldiv_B!(QZ.QZ.LU, h)
    gemv!('N', -one(T), QZ.Z, h, one(T), x)
end

"""
Solve the problem ``(I - ZZ^*)(βA - αB)(I - QQ^*)t = b``
this is simplified in Krylov subspaces as solving
``(βA - αB)t = b``
with the left preconditioner `Pl = (I - Z inv(Q'Z) Q')`
"""
function solve_generalized_correction_equation!(solver::gmres_solver, A, B, preconditioner, x, Q, Z, precZ, QZ, α, β, r, spare_vector, tol)

    n = size(A, 1)
    T = eltype(x)

    # Preconditioner
    Pl = QZpreconditioner{T}(Q, precZ, QZ, preconditioner)
    
    # Linear operator
    Ax_minus_Bx = LinearMap{T}((y, x) -> begin
        # y = (βA - αB) * x
        A_mul_B!(y, B, x)
        scale!(-α, y)
        A_mul_B!(spare_vector, A, x)
        axpy!(β, spare_vector, y)
    end, n, ismutating = true)

    residual = Residual(solver.iterations, T)

    # Start with a zero guess
    fill!(x, zero(T))

    # Initialize the first vector of the krylov basis
    first_col = view(solver.arnoldi.V, :, 1)
    copy!(first_col, r)
    A_ldiv_B!(Pl, first_col)
    residual.current = norm(first_col)
    scale!(first_col, inv(residual.current))

    # Update some residual stuff
    residual.accumulator = one(real(T))
    residual.β = residual.current

    iterable = GMRESIterable(Ax_minus_Bx, Pl, Identity(), x, r, 
        spare_vector, solver.arnoldi, residual, 0, 
        solver.iterations, tol * residual.current, residual.current
    )

    for res = iterable 
        # println(res)
    end

    nothing
end

function solve_generalized_correction_equation!(solver::bicgstabl_solver, A, B, preconditioner, x, Q, Z, precZ, QZ, α, β, r, spare_vector, tol)
    n = size(A, 1)
    T = eltype(x)

    Pl = QZpreconditioner{T}(Q, precZ, QZ, preconditioner)
    
    Ax_minus_Bx = LinearMap{T}((y, x) -> begin
        # y = (βA - αB) * x
        A_mul_B!(y, B, x)
        scale!(-α, y)
        A_mul_B!(spare_vector, A, x)
        axpy!(β, spare_vector, y)
    end, n, ismutating = true)

    mv_products = 0

    residual = view(solver.rs, :, 1)

    # The initial residual rs[:, 1] = b - A * x
    copy!(residual, r)
    fill!(view(solver.us, :, 1), zero(T))
    fill!(x, zero(T))

    # Apply the left preconditioner
    A_ldiv_B!(Pl, residual)

    ω = σ = one(T)

    nrm = norm(residual)

    # Stopping condition based on relative tolerance.
    reltol = nrm * tol

    iterable = BiCGStabIterable(Ax_minus_Bx, r, solver.l, x, solver.r_shadow, solver.rs, solver.us,
        solver.max_mv_products, mv_products, reltol, nrm,
        Pl,
        solver.γ, ω, σ, solver.M
    )

    for res = iterable 
        # println(res)  
    end

    nothing
end