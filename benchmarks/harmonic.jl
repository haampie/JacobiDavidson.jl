module Bench

using JacobiDavidson
using BenchmarkTools

function bench_harmonic(; n = 100, τ = 2.0 + 0.01im)
  
  A = spdiagm(
    (fill(-1.0, n - 1), 
    fill(2.0, n), 
    fill(-1.2, n - 1)), 
    (-1, 0, 1)
  )

  bench_new = @benchmark JacobiDavidson.jdqr_harmonic_efficient(
    $A,
    $(GMRES(iterations = 5)),
    pairs = 10,
    subspace_dimensions = 10:15,
    max_iter = 300,
    tolerance = 1e-8,
    τ = $τ
  ) setup = (srand(4))

  bench_old = @benchmark JacobiDavidson.jdqr_harmonic(
    $A,
    $(GMRES(iterations = 5)),
    pairs = 10,
    subspace_dimensions = 10:15,
    max_iter = 300,
    tolerance = 1e-8,
    τ = $τ
  ) setup = (srand(4))

  return bench_new, bench_old
end

function bench_harmonic_allocs(; n = 100, τ = 2.0 + 0.01im)
  
  A = spdiagm(
    (fill(-1.0, n - 1), 
    fill(2.0, n), 
    fill(-1.2, n - 1)), 
    (-1, 0, 1)
  )

  srand(4)

  result = JacobiDavidson.jdqr_harmonic_efficient(
    A,
    GMRES(iterations = 5),
    pairs = 10,
    subspace_dimensions = 10:15,
    max_iter = 300,
    tolerance = 1e-8,
    τ = τ
  )

  Profile.clear_malloc_data()


  srand(4)

  result = JacobiDavidson.jdqr_harmonic_efficient(
    A,
    GMRES(iterations = 5),
    pairs = 10,
    subspace_dimensions = 10:15,
    max_iter = 300,
    tolerance = 1e-8,
    τ = τ
  )
end
end

Bench.bench_harmonic_allocs()