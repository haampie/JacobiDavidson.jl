using Base.LinAlg.axpy!

function gmres(A, b; max_iter::Int = 20, tol = sqrt(eps(real(eltype(b)))))
  T = eltype(b)

  # Approximate solution
  x = zeros(T, size(A, 1))
  arnoldi = ArnoldiDecomp(A, max_iter, T)
  residual = Residual(max_iter, T)
  relres::Vector{real(T)} = [one(T)]
 
  # Set the first basis vector
  β::real(T) = init!(arnoldi, x, b)

  # And initialize the residual
  init_residual!(residual)

  k = 1
  
  while k ≤ max_iter && residual.current > tol

    # Arnoldi step: expand and orthogonalize
    expand!(arnoldi, k)
    orthogonalize!(arnoldi, k)
    update_residual!(residual, arnoldi, k)
    push!(relres, residual.current)

    k += 1
  end

  # Solve the projected problem Hy = β * e1 in the least-squares sense.
  extract!(x, arnoldi, β, k)

  return x, relres
end

type ArnoldiDecomp{T}
  A
  V::Vector{Vector{T}} # Orthonormal basis vectors
  H::Matrix{T}         # Hessenberg matrix
end

ArnoldiDecomp(A, order::Int, T::Type) = ArnoldiDecomp{T}(
  A,
  [zeros(T, size(A, 1)) for i = 1 : order + 1],
  zeros(T, order + 1, order)
)

type Residual{numT, resT}
  current::resT # Current relative residual
  accumulator::resT # Used to compute the residual on the go
  nullvec::Vector{numT} # Vector in the null space to compute residuals
end

Residual(order, T::Type) = Residual{T, real(T)}(
  one(real(T)),
  one(real(T)),
  ones(T, order + 1)
)

function update_residual!{numT, resT}(r::Residual{numT, resT}, arnoldi::ArnoldiDecomp, k::Int)
  # Cheaply computes the current residual
  r.nullvec[k + 1] = -conj(dot(r.nullvec[1 : k], @view(arnoldi.H[1 : k, k])) / arnoldi.H[k + 1, k])
  r.accumulator += abs2(r.nullvec[k + 1])
  r.current = one(resT) / √r.accumulator
end

function init!{T}(arnoldi::ArnoldiDecomp{T}, x, b)
  # Initialize the Krylov subspace with the initial residual vector
  
  # This basically does V[1] = b - A * x
  copy!(arnoldi.V[1], b)
  axpy!(-one(T), arnoldi.A * x, arnoldi.V[1])
  β = norm(arnoldi.V[1])
  scale!(arnoldi.V[1], one(T) / β)
  β
end

@inline function init_residual!{numT, resT}(r::Residual{numT, resT})
  r.accumulator = one(resT)
end

function extract!{T}(x, arnoldi::ArnoldiDecomp{T}, β, k::Int)
  # Computes & updates the solution
  rhs = zeros(T, k)
  rhs[1] = β
  y = @view(arnoldi.H[1 : k, 1 : k - 1]) \ rhs

  # Update x ← x + V * y
  for i = 1 : k - 1
    axpy!(y[i], arnoldi.V[i], x)
  end
end

@inline function expand!(arnoldi::ArnoldiDecomp, k::Int)
  # Simply expands by A * v
  A_mul_B!(arnoldi.V[k + 1], arnoldi.A, arnoldi.V[k])
end

function orthogonalize!{T}(arnoldi::ArnoldiDecomp{T}, k::Int)
  # Orthogonalize using Gram-Schmidt
  for j = 1 : k
    arnoldi.H[j, k] = dot(arnoldi.V[j], arnoldi.V[k + 1])
    axpy!(-arnoldi.H[j, k], arnoldi.V[j], arnoldi.V[k + 1])
  end

  # Reorthogonalize
  for j = 1 : k
    increment = dot(arnoldi.V[j], arnoldi.V[k + 1])
    arnoldi.H[j, k] += increment
    axpy!(-increment, arnoldi.V[j], arnoldi.V[k + 1])
  end

  arnoldi.H[k + 1, k] = norm(arnoldi.V[k + 1])
  scale!(arnoldi.V[k + 1], one(T) / arnoldi.H[k + 1, k])
end