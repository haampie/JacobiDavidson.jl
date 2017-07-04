module JDQZBench

using JacobiDavidson

function bench_jdqz_allocs(; n = 100, τ = 2.0 + 0.01im)
  srand(4)
  
  A = 100 * speye(Complex128, n) + sprand(Complex128, n, n, .5)
  B = 100 * speye(Complex128, n) + sprand(Complex128, n, n, .5)

  result = jdqz(
    A,
    B,
    exact_solver(),
    pairs = 10,
    min_dimension = 10,
    max_dimension = 15,
    max_iter = 300,
    ɛ = 1e-8,
    τ = τ
  )

  Profile.clear_malloc_data()

  srand(4)

  result = jdqz(
    A,
    B,
    exact_solver(),
    pairs = 10,
    min_dimension = 10,
    max_dimension = 15,
    max_iter = 300,
    ɛ = 1e-8,
    τ = τ
  )
end
end

JDQZBench.bench_jdqz_allocs()