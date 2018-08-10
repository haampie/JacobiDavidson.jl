module Tst

using Plots
using JacobiDavidson
using LinearMaps

import LinearAlgabra: ldiv!

function myA!(y::AbstractVector{T}, x::AbstractVector{T}) where {T<:Number}
  for i = 1 : length(x)
    @inbounds y[i] = sqrt(i) * x[i]
  end
end

function myB!(y::AbstractVector{T}, x::AbstractVector{T}) where {T<:Number}
  for i = 1 : length(x)
    @inbounds y[i] = x[i] / sqrt(i)
  end
end

struct SuperPreconditioner{numT <: Number}
  target::numT
end

function ldiv!(p::SuperPreconditioner{T}, x::AbstractVector{T}) where {T<:Number}
  for i = 1 : length(x)
    @inbounds x[i] = x[i] * sqrt(i) / (i - p.target)
  end
end

function another_example(; n = 1000, target = Near(31.0 + 0.1im))
  A = LinearMap{Float64}(myA!, n; ismutating = true)
  B = LinearMap{Float64}(myB!, n; ismutating = true)
  P = SuperPreconditioner(target.τ)

  schur2, residuals2 = jdqz(
    A, B, gmres_solver(n, iterations = 3),
    preconditioner = P,
    testspace = Harmonic,
    target = target,
    pairs = 5,
    ɛ = 1e-9,
    min_dimension = 10,
    max_dimension = 20,
    max_iter = 100,
    verbose = true
  )

  schur, residuals = jdqz(
    A, B, bicgstabl_solver(n, max_mv_products = 10, l = 2),
    preconditioner = P,
    testspace = Harmonic,
    target = target,
    pairs = 5,
    ɛ = 1e-9,
    min_dimension = 10,
    max_dimension = 20,
    max_iter = 100,
    verbose = true
  )

  plot(residuals, yscale = :log10, label = "BiCGStab", marker = :x)
  plot!(residuals2, yscale = :log10, label = "GMRES", marker = :x)
end

function generalized(; n = 1_000, target = Near(0.5 + 0.1im))
  srand(50)
  A = 2 * speye(ComplexF64, n) + sprand(ComplexF64, n, n, 1 / n)
  B = 2 * speye(ComplexF64, n) + sprand(ComplexF64, n, n, 1 / n)

  values = eigvals(full(A), full(B))

  @time schur, residuals = jdqz(
    A, B,
    gmres_solver(n, iterations = 10),
    target = target,
    pairs = 10,
    min_dimension = 10,
    max_dimension = 15,
    max_iter = 300,
    verbose = true
  )
  
  found = schur.alphas ./ schur.betas

  p1 = scatter(real(values), imag(values), label = "eig")
  scatter!(real(found), imag(found), marker = :+, label = "jdqz")
  scatter!([real(target.τ)], [imag(target.τ)], marker = :star, label = "Target")

  p2 = plot(residuals, yscale = :log10)

  p1, p2
end

function test_harmonic_2(n = 1000; τ = 1.52 + 0.0im)
  srand(4)

  A = spdiagm(
    (fill(-1.0, n - 1), 
    fill(2.0, n),
    fill(-1.0, n - 1)),
    (-1, 0, 1)
  )
  
  B = speye(n)

  values = eigvals(full(A), full(B))

  Q, Z, S, T, residuals = jdqz(
    A,
    B,
    exact_solver(),
    pairs = 10,
    min_dimension = 10,
    max_dimension = 20,
    max_iter = 300,
    ɛ = 1e-5,
    τ = τ
  )

  found = diag(S) ./ diag(T)

  p1 = scatter(real(values), imag(values), ylims = (-1, 1))
  scatter!(real(found), imag(found), marker = :+)

  p2 = plot(residuals, yscale = :log10)

  p1, p2
end

function test_harmonic(; n = 500, τ = 0.01 + 0.02im)
  srand(4)

  A = spdiagm(
    (fill(-1.0, n - 1), 
    fill(2.0, n),
    fill(-1.0, n - 1)),
    (-1, 0, 1)
  )

  schur, ritz_hist, conv_hist, residuals = jdqr(
    A,
    bicgstabl_solver(A, max_mv_products = 30, l = 2), 
    pairs = 10,
    min_dimension = 5,
    max_dimension = 10,
    max_iter = 300,
    ɛ = 1e-5,
    τ = τ,
    verbose = true
  )

  λs = real(eigvals(full(A)))

  # Total number of iterations
  iterations = length(ritz_hist)

  @show iterations
  @show schur.values

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
