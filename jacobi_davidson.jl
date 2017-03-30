function jacobi_davidson(A::AbstractMatrix, iterations::Int; k::Int = 2)
  # First construct an orthonormal basis
  # for the Krylov subspace
  V, H = arnoldi(A, iterations)

  # Factorize the Hessenberg matrix into Schur form
  F = schurfact(complex(H[1 : iterations, 1 : iterations]))

  # Reorder the Schur form with the `k` largest eigs in front
  p = sortperm(abs(F[:values]), rev = true)
  Π = falses(iterations)
  Π[p[1 : k]] = true
  ordschur!(F, Π)

  # Take a Ritz pair pair (θ, y)
  θ = F[:values][1]
  y = F[:vectors][:, 1]

  # Approximate eigenpair (θ, u)
  u = V[:, 1 : iterations] * y

  # Residual
  r = A * u - θ * u

  ### Solve the correction equation

  # Matrix-vector product
  M = x -> begin
    x -= (u' * x) * u
    x = A * x - θ * x
    x -= (u' * x) * u
  end
  (I - u * u') * (A - θI)
end

function expand!(V, H, w)
  k = size(H, 2)
  # Orthogonalize using Gramm-Schmidt
  for j = 1 : k
    H[j, k] = dot(w, V[:, j])
    w -= H[j, k] * V[:, j]
  end

  H[k + 1, k] = norm(w)
  V[:, k + 1] = w / H[k + 1, k]
end

function arnoldi(A::AbstractMatrix, dimension::Int)
  n = size(A, 1)
  H = zeros(dimension + 1, dimension)
  V = zeros(n, dimension + 1)
  V[:, 1] = rand(n)
  V[:, 1] /= norm(V[:, 1])

  for k = 1 : dimension
    expand!(V, H, A * V[:, k])
  end

  V, H
end

function gmres(A::AbstractMatrix)

function poisson_matrix(n::Int = 100)
  t = -(n + 1) ^ 2
  SymTridiagonal(-2 * t * ones(n), t * ones(n - 1))
end

function testing()

end