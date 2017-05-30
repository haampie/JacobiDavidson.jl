abstract CorrectionSolver

type gmres_solver <: CorrectionSolver
  iterations::Int
  tolerance::Float64
  gmres_solver(;iterations::Int = 5, tolerance::Float64 = 1e-6) = new(iterations, tolerance)
end

type exact_solver <: CorrectionSolver
end

function solve_correction{T <: AbstractLinearMap}(solver::gmres_solver, A::T, θ, u::AbstractVector)
  n = size(A, 1)

  # Define the residual mapping
  R = LinearMap(x -> A * x - θ * x, nothing, n)

  # Projection Cⁿ → Cⁿ ∖ span {u}: Px = (I - uu')x
  P = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)

  # Coefficient matrix A - θI restricted map: Cⁿ ∖ span {u} -> Cⁿ ∖ span {u}
  C = P * R * P

  # Residual
  r = R * u

  gmres(C, -r, max_iter = solver.iterations, ɛ = solver.tolerance)
end

function solve_correction{T <: AbstractLinearMap}(solver::exact_solver, A::T, θ, u::AbstractVector)
  # The exact solver is mostly useful for testing Jacobi-Davidson
  # method itself and should result in quadratic convergence.
  # However, in general the correction equation should be solved
  # only approximately in a fixed number of GMRES / BiCGStab
  # iterations to reduce computation costs.

  # This method is *not* matrix-free (TODO: assert this or specify solve_correction)

  # Here we solve the augmented system
  # [A - θI, u; u' 0][t; y] = [-r; 0] for t,
  # which is equivalent to solving (A - θI) t = -r
  # for t ⟂ u.

  n = size(A, 1)
  r = A * u - θ * u
  Ã = [(A.lmap - θ * speye(n)) u; u' 0]
  rhs = [-r; 0]
  (Ã \ rhs)[1 : n]
end

function solve_deflated_correction{T <: AbstractLinearMap}(solver::exact_solver, A::T, θ, X::AbstractMatrix, u::AbstractVector)
  # The exact solver is mostly useful for testing Jacobi-Davidson
  # method itself and should result in quadratic convergence.
  # However, in general the correction equation should be solved
  # only approximately in a fixed number of GMRES / BiCGStab
  # iterations to reduce computation costs.

  # This method is *not* matrix-free (TODO: assert this or specify solve_correction)

  # Here we solve the augmented system
  # [A - θI, [X, u]; [X, u]' 0][t; y] = [-r; 0] for t,
  # which is equivalent to solving (A - θI) t = -r
  # for t ⟂ X and t ⟂ u.

  n = size(A, 1)
  r = A * u - θ * u
  Q = [X u]
  m = size(Q, 2)
  Ã = [(A.lmap - θ * speye(n)) Q; Q' zeros(m, m)]
  rhs = [-r; zeros(m, 1)]
  (Ã \ rhs)[1 : n]
end

function gmres{T <: AbstractLinearMap}(A::T, b::AbstractVector; max_iter::Int = 5, ɛ::Float64 = 1e-6)
  # This is a poor man's impl. of GMRES, but it will be replaced by
  # a solver from IterativeSolvers.jl anyway; although a minimalist version could be faster.
  # Zero initial guess ensures the Krylov subspace is orthogonal to the approximate eigenvector.

  n = size(A, 1)
  β = norm(b)
  V = zeros(Complex{Float64}, n, max_iter + 1)
  H = zeros(Complex{Float64}, max_iter + 1, max_iter)
  V[:, 1] = b / complex(β)

  # Create a Krylov subspace of dimension max_iter
  for k = 1 : max_iter
    # Add the (k + 1)th basis vector
    expand!(V, H, A * V[:, k], k)
  end

  # Solve the low-dimensional problem
  e₁ = zeros(max_iter + 1)
  e₁[1] = β
  y = H \ e₁

  # Project back to the large dimensional solution
  V[:, 1 : max_iter] * y
end
