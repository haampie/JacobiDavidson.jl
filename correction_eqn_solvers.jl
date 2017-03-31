abstract CorrectionSolver
immutable exact <: CorrectionSolver end
immutable gmres <: CorrectionSolver end

function gmres_solver{T <: AbstractLinearMap}(A::T, θ::AbstractFloat, u::AbstractVector)
  # Define the residual mapping
  R = LinearMap(x -> A * x - θ * x, nothing, n)

  # Projection Cⁿ → Cⁿ ∖ span {u}: Px = (I - uu')x
  P = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)

  # Coefficient matrix A - θI : Cⁿ ∖ span {u} -> Cⁿ ∖ span {u}
  C = P * R * P

  # Residual
  r = R * u

  gmres(C, -r)
end

function gmres{T <: AbstractLinearMap}(A::T, b::AbstractVector; max_iter::Int = 5)
  n = size(A, 1)
  β = norm(b)
  V = zeros(Complex{Float64}, n, max_iter + 1)
  H = zeros(Complex{Float64}, max_iter + 1, max_iter)
  V[:, 1] = b / complex(β)

  # history = Float64[]
  
  # Create a Krylov subspace of dimension max_iter
  for k = 1 : max_iter
    # Add the (k + 1)th basis vector
    expand!(V, H, A * V[:, k], k)

    # Solve the system
    e₁ = zeros(k + 1)
    e₁[1] = β
    y = H[1 : k + 1, 1 : k] \ e₁
    x = V[:, 1 : k] * y
    # push!(history, norm(A * x - b))
  end

  # Solve the low-dimensional problem
  e₁ = zeros(max_iter + 1)
  e₁[1] = β
  y = H \ e₁

  # Project back to the large dimensional solution
  V[:, 1 : max_iter] * y
end

function exact_solver{T <: AbstractLinearMap}(A::T, )