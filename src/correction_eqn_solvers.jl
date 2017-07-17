abstract type CorrectionSolver end

type gmres_solver <: CorrectionSolver
  iterations::Int
  tolerance::Float64
  gmres_solver(;iterations::Int = 5, tolerance::Float64 = 1e-6) = new(iterations, tolerance)
end

type exact_solver <: CorrectionSolver
end

function solve_deflated_correction(solver::exact_solver, A, θ, X::AbstractMatrix, u::AbstractVector, r::AbstractVector)
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
  Q = [X u]
  m = size(Q, 2)
  Ã = [(A - θ * speye(n)) Q; Q' zeros(m, m)]
  rhs = [-r; zeros(m, 1)]
  (Ã \ rhs)[1 : n]
end

function solve_deflated_correction(solver::gmres_solver, A, θ, X::AbstractMatrix, u::AbstractVector, r::AbstractVector)
  n = size(A, 1)

  # Define the residual mapping
  R = LinearMap(x -> A * x - θ * x, nothing, n)

  # Projection Cⁿ → Cⁿ ∖ span {u}: P1x = (I - uu')x
  P1 = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)

  # Projection Cⁿ → Cⁿ ∖ span {X}: P2x = (I - XX')x
  P2 = LinearMap(x -> x - X * (X' * x), nothing, n; ishermitian = true)

  # Coefficient matrix A - θI restricted map: Cⁿ ∖ span {Q} -> Cⁿ ∖ span {Q}
  C = P2 * P1 * R

  gmres(C, -r, max_iter = solver.iterations, tol = solver.tolerance)
end

function solve_generalized_correction_equation(solver::exact_solver, A, B, Q, Z, ζ, η, r)
  n = size(A, 1)
  m = size(Q, 2)
  # Assuming both A and B are sparse while Q and Z are dense, let's try to avoid constructing a huge dense matrix.
  # Let C = η * A - ζ * B

  # We have to solve:
  # |C  Z| |t| = |-r| 
  # |Q' O| |z|   |0 |

  # Use the Schur complement trick with S = -Q' * inv(C) * Z
  # |C  Z| = |I            O| |C Z|
  # |Q' O|   |Q' * inv(C)  I| |O S|

  # And solve two systems with inv(C) occurring multiple times.
  C = η * A - ζ * B # Completely sparse
  y = Q' * (C \ r)
  S = Q' * (C \ Z) # Schur complement
  z = -S \ y
  t = C \ (-r - Z * z)
end

function solve_generalized_correction_equation(solver::gmres_solver, A, B, Q, Z, ζ, η, r)
  n = size(A, 1)
  
  P1 = LinearMap(x -> x - Z * (Z' * x), nothing, n; ishermitian = true)
  R = LinearMap(x -> η * A * x - ζ * B * x, nothing, n)
  P2 = LinearMap(x -> x - Q * (Q' * x), nothing, n; ishermitian = true)

  C = P2 * P1 * R
end
