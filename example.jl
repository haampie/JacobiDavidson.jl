module Tst

using Plots
using JacobiDavidson
using LinearMaps
using LinearAlgebra
using SparseArrays

import LinearAlgebra: ldiv!

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
  return x
end

function ldiv!(y::AbstractVector{T}, p::SuperPreconditioner{T}, x::AbstractVector{T}) where {T<:Number}
  for i = 1 : length(x)
    @inbounds y[i] = x[i] * sqrt(i) / (i - p.target)
  end
  return y
end

function another_example(; n = 1000, target = Near(31.0 + 0.1im))
  A = LinearMap{Float64}(myA!, n; ismutating = true)
  B = LinearMap{Float64}(myB!, n; ismutating = true)
  P = SuperPreconditioner(target.τ)

  schur2, residuals2 = jdqz(
    A, B, 
    solver = GMRES(n, iterations = 3),
    preconditioner = P,
    testspace = Harmonic,
    target = target,
    pairs = 5,
    tolerance = 1e-9,
    subspace_dimensions = 10:20,
    max_iter = 100,
    verbosity = 1
  )

  schur, residuals = jdqz(
    A, B, 
    solver = BiCGStabl(n, max_mv_products = 10, l = 2),
    preconditioner = P,
    testspace = Harmonic,
    target = target,
    pairs = 5,
    tolerance = 1e-9,
    subspace_dimensions = 10:20,
    max_iter = 100,
    verbosity = 1
  )

  plot(residuals, yscale = :log10, label = "BiCGStab", marker = :x)
  plot!(residuals2, yscale = :log10, label = "GMRES", marker = :x)
end

function generalized(; n = 1_000, target = Near(0.5 + 0.1im))
  A = 2I + sprand(ComplexF64, n, n, 1 / n)
  B = 2I + sprand(ComplexF64, n, n, 1 / n)

  values = eigvals(Matrix(A), Matrix(B))

  @time schur, residuals = jdqz(
    A, B,
    solver = GMRES(n, iterations = 10),
    target = target,
    pairs = 10,
    subspace_dimensions = 10:15,
    max_iter = 300,
    verbosity = 1
  )
  
  found = schur.alphas ./ schur.betas

  p1 = scatter(real(values), imag(values), label = "eig")
  scatter!(real(found), imag(found), marker = :+, label = "jdqz")
  scatter!([real(target.τ)], [imag(target.τ)], marker = :star, label = "Target")

  p2 = plot(residuals, yscale = :log10)

  p1, p2
end

function test_harmonic_2(n = 1000; τ = Near(1.52 + 0.0im))
  A = spdiagm(
    -1 => fill(-1.0, n - 1), 
     0 => fill(2.0, n),
     1 => fill(-1.0, n - 1)
  )
  
  B = spdiagm(0 => fill(1.0, n))

  values = eigvals(Matrix(A), Matrix(B))

  schur, residuals = jdqz(
    A,
    B,
    solver = Exact(),
    pairs = 10,
    subspace_dimensions = 10:20,
    max_iter = 300,
    tolerance = 1e-5,
    target = τ
  )

  found = schur.alphas ./ schur.betas

  p1 = scatter(real(values), imag(values), ylims = (-1, 1))
  scatter!(real(found), imag(found), marker = :+)

  p2 = plot(residuals, yscale = :log10)

  p1, p2
end

function test_harmonic(; n = 500, τ = Near(0.01 + 0.02im))
  A = spdiagm(
    -1 => fill(-1.0, n - 1), 
     0 => fill(2.0, n),
     1 => fill(-1.0, n - 1)
  )

  schur, ritz_hist, conv_hist, residuals = jdqr(
    A,
    solver = BiCGStabl(size(A, 1), max_mv_products = 30, l = 2), 
    pairs = 10,
    subspace_dimensions = 5:10,
    max_iter = 300,
    tolerance = 1e-5,
    target = τ,
    verbosity = 1
  )

  λs = real(eigvals(Matrix(A)))

  # Total number of iterations
  iterations = length(ritz_hist)

  @show iterations
  @show schur.values

  # Plot the target as a horizontal line
  p = plot([0.0, iterations + 1.0], [real(τ.τ), real(τ.τ)], linewidth=5, legend = :none, layout = (2, 1))

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
