module Tst

using Plots
using JacobiDavidson
using LinearMaps

function generalized(; n = 200, min = 10, max = 20)
  srand(50)
  A = 100 * speye(Complex128, n) + sprand(Complex128, n, n, .5)
  B = 100 * speye(Complex128, n) + sprand(Complex128, n, n, .5)

  values = eigvals(full(A), full(B))

  Q, Z, S, T, residuals = jdqz(A, B, gmres_solver(), τ = 1.0 + 0im, pairs = 20)
  
  found = diag(S) ./ diag(T)

  p1 = scatter(real(values), imag(values))
  scatter!(real(found), imag(found), marker = :+)

  p2 = plot(residuals, yscale = :log10)

  p1, p2
end

function test_harmonic(; n = 500, τ = 1.52 + 0.0im)
  srand(4)

  A = spdiagm(
    (fill(-1.0, n - 1), 
    fill(2.0, n),
    fill(-1.0, n - 1)),
    (-1, 0, 1)
  )

  λs = real(eigvals(full(A)))

  Q, R, ritz_hist, conv_hist, residuals = jdqr_harmonic(
    A,
    exact_solver(),
    pairs = 10,
    min_dimension = 10,
    max_dimension = 20,
    max_iter = 300,
    ɛ = 1e-5,
    τ = τ
  )

  # Total number of iterations
  iterations = length(ritz_hist)

  @show iterations
  @show diag(R)

  # Plot the target as a horizontal line
  p = plot([0.0, iterations + 1.0], [real(τ), real(τ)], linewidth=5, legend = :none, layout = (2, 1))

  # Plot the actual eigenvalues
  scatter!(fill(iterations + 1, n), λs, xlabel = "Iteration", ylims = (minimum(λs) - 2.0, maximum(λs) + 2.0), marker = (:diamond, :black), subplot = 1)
  
  # Plot the approximate eigenvalues per iteration
  for (k, (ritzvalues, eigenvalues)) = enumerate(zip(ritz_hist, conv_hist))
    scatter!(k * ones(length(ritzvalues)), real(ritzvalues), marker = (:+, :green), subplot = 1)
    scatter!(k * ones(length(eigenvalues)), real(eigenvalues), marker = (:hexagon, :yellow), subplot = 1)
  end

  plot!(residuals, xlabel = "Iteration", label = "Residual", xticks = 0.0 : 50.0 : 350.0, yaxis = :log10, subplot = 2)

  p
end
end
