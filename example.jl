module Tst

import LinearMaps: LinearMap

using Plots
using JacobiDavidson

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

function hermetian_example(; n = 200, min = 10, max = 30, )
  B = sprand(n, n, .2) + 100.0 * speye(n);
  B[1, 1] = 30.0
  A = Symmetric(B)

  exact = exact_solver()

  @time D, X, res = jacobi_davidson_hermetian(
    LinearMap(A, isposdef = false),
    exact,
    min_dimension = min,
    max_dimension = max
  )

  X, D, res
end

function test_harmonic(; n = 1000, τ = 6.0 + 0.0im)
  srand(4)

  A = spdiagm(
    (fill(-1.0, n - 1), 
    fill(2.0, n),
    fill(-1.2, n - 1)),
    (-1, 0, 1)
  )

  λs = real(eigvals(full(A)))

  # Q, R, ritz_hist, conv_hist, residuals = jdqr_harmonic_simpler(
  #   A,
  #   gmres_solver(iterations = 5),
  #   pairs = 10,
  #   min_dimension = 10,
  #   max_dimension = 15,
  #   max_iter = 300,
  #   ɛ = 1e-8,
  #   τ = τ
  # )
  srand(4)

  @time Q1, R1, harmonic_ritz_values1, converged_ritz_values1, residuals1 = jdqr_harmonic_matrix(
    A,
    gmres_solver(iterations = 10),
    pairs = 10,
    min_dimension = 10,
    max_dimension = 20,
    max_iter = 300,
    ɛ = 1e-5,
    τ = τ
  )

  @show norm(Q1' * Q1)

  srand(4)

  @time Q2, R2, harmonic_ritz_values2, converged_ritz_values2, residuals2 = jdqr_harmonic(
    A,
    gmres_solver(iterations = 10),
    pairs = 10,
    min_dimension = 10,
    max_dimension = 50,
    max_iter = 300,
    ɛ = 1e-5,
    τ = τ
  )

  p = plot(residuals1, yscale = :log10, label = "new")
  plot!(residuals2, yscale = :log10, label = "old")

  return p

  # Converged eigenvalues.
  @show diag(R)

  # Total number of iterations
  iterations = length(ritz_hist)

  @show iterations

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
