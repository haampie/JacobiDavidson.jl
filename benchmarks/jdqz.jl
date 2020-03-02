module JDQZBench

using JacobiDavidson
using BenchmarkTools

function bench_jdqz_allocs(; n = 10_000, τ = 0.0 + 0.01im)
  srand(4)
  
  A = 2 * speye(ComplexF64, n) + sprand(ComplexF64, n, n, 1 / n)
  B = 2 * speye(ComplexF64, n) + sprand(ComplexF64, n, n, 1 / n)

  result = jdqz(
    A,
    B,
    BiCGStabl(A, max_mv_products = 10), 
    τ = 0.0 + 0.0im, 
    pairs = 20, 
    max_iter = 1000,
    subspace_dimensions = 10:15,
    tolerance = 1e-8,
    verbose = false
  )
end

function bench_jdqz(; n = 1_000, τ = 0.0 + 0.01im)
  srand(4)
  
  A = 2 * speye(ComplexF64, n) + sprand(ComplexF64, n, n, 1 / n)
  B = 2 * speye(ComplexF64, n) + sprand(ComplexF64, n, n, 1 / n)

  eig_bench = @benchmark eigs($A, $B, nev = 20, sigma = 0.0 + 0.0im, ritzvec = true, tol = 1e-8)

  jdqz_bench = @benchmark $jdqz(
    $A,
    $B,
    $BiCGStabl($A, max_mv_products = 10), 
    τ = 0.0 + 0.0im, 
    pairs = 20,
    max_iter = 1000,
    subspace_dimensions = 10:15,
    tolerance = 1e-8,
  )

  eig_bench, jdqz_bench
end


end

JDQZBench.bench_jdqz_allocs()
Profile.clear()
@profile JDQZBench.bench_jdqz_allocs()
