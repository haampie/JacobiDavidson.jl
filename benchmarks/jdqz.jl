module JDQZBench

using JacobiDavidson
using BenchmarkTools

function bench_jdqz_allocs(; n = 100_000, τ = 0.0 + 0.01im)
  srand(4)
  
  A = 2 * speye(Complex128, n) + sprand(Complex128, n, n, 1 / n)
  B = 2 * speye(Complex128, n) + sprand(Complex128, n, n, 1 / n)

  result = jdqz(
    A,
    B,
    bicgstabl_solver(A, max_mv_products = 10), 
    τ = 0.0 + 0.0im, 
    pairs = 20, 
    max_iter = 1000,
    min_dimension = 10,
    max_dimension = 15,
    ɛ = 1e-8,
    verbose = true
  )
end

function bench_jdqz(; n = 1_000, τ = 0.0 + 0.01im)
  srand(4)
  
  A = 2 * speye(Complex128, n) + sprand(Complex128, n, n, 1 / n)
  B = 2 * speye(Complex128, n) + sprand(Complex128, n, n, 1 / n)

  @benchmark $jdqz(
    $A,
    $B,
    $bicgstabl_solver($A, max_mv_products = 10), 
    τ = 0.0 + 0.0im, 
    pairs = 20, 
    max_iter = 1000,
    min_dimension = 10,
    max_dimension = 15,
    ɛ = 1e-8,
  )
end


end

JDQZBench.bench_jdqz_allocs()
Profile.clear()
@profile JDQZBench.bench_jdqz_allocs()