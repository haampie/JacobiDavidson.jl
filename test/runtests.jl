using Test

using JacobiDavidson, LinearAlgebra, LinearMaps

@testset "Schur permutations" begin
    A = rand(ComplexF64, 10, 10)
    B = rand(ComplexF64, 10, 10)
    F = schur(A, B)

    τ = 1.0 + 0.0im
    eigenvalues = F.alpha ./ F.beta

    # Test whether the ordering is correct
    closest_to_one = first(schur_permutation(Near(τ), F, 1:1))
    @test abs.(eigenvalues[closest_to_one] - τ) ≤ minimum(abs.(eigenvalues .- τ))

    largest_magnitude = first(schur_permutation(LargestMagnitude(τ), F, 1:1))
    @test abs(eigenvalues[largest_magnitude]) ≥ maximum(abs.(eigenvalues))

    smallest_magnitude = first(schur_permutation(SmallestMagnitude(τ), F, 1:1))
    @test abs(eigenvalues[smallest_magnitude]) ≤ minimum(abs.(eigenvalues))

    largest_real_part = first(schur_permutation(LargestRealPart(τ), F, 1:1))
    @test real(eigenvalues[largest_real_part]) ≥ maximum(real.(eigenvalues))

    smallest_real_part = first(schur_permutation(SmallestRealPart(τ), F, 1:1))
    @test real(eigenvalues[smallest_real_part]) ≤ minimum(real.(eigenvalues))

    largest_imaginary_part = first(schur_permutation(LargestImaginaryPart(τ), F, 1:1))
    @test imag(eigenvalues[largest_imaginary_part]) ≥ maximum(imag.(eigenvalues))

    smallest_imaginary_part = first(schur_permutation(SmallestImaginaryPart(τ), F, 1:1))
    @test imag(eigenvalues[smallest_imaginary_part]) ≤ minimum(imag.(eigenvalues))

    # Test whether one can reorder the Schur decomp with multiple eigenvalues
    take = 1 : 3; keep = 4 : 10
    schur_sort!(Near(τ), F, take)
    sorted = F.alpha ./ F.beta
    @test maximum(abs.(sorted[take] .- τ)) ≤ minimum(abs.(sorted[keep] .- τ))

    # Test whether one can reorder the Schur decomp with a single eigenvalue
    schur_sort!(SmallestMagnitude(τ), F, 1)
    sorted = F.alpha ./ F.beta
    @test all(abs(sorted[1]) .≤ abs.(sorted[2 : end]))
end

@testset "Jacobi–Davidson" begin
    function myA!(y, x)
        for i = 1 : length(x)
            @inbounds y[i] = sqrt(i) * x[i]
        end
    end

    function myB!(y, x)
        for i = 1 : length(x)
            @inbounds y[i] = x[i] / sqrt(i)
        end
    end

    struct SuperPreconditioner{numT <: Number}
        target::numT
    end

    function LinearAlgebra.ldiv!(p::SuperPreconditioner{T}, x::AbstractVector{T}) where {T<:Number}
        for i = 1 : length(x)
            @inbounds x[i] = x[i] * sqrt(i) / (i - p.target)
        end
        return x
    end

    function LinearAlgebra.ldiv!(y::AbstractVector{T}, p::SuperPreconditioner{T}, x::AbstractVector{T}) where {T<:Number}
        for i = 1 : length(x)
            @inbounds y[i] = x[i] * sqrt(i) / (i - p.target)
        end
        return y
    end

    n = 10_000
    target = Near(5_000.1 + 0.0im)
    A = LinearMap{Float64}(myA!, n; ismutating = true)
    B = LinearMap{Float64}(myB!, n; ismutating = true)
    P = SuperPreconditioner(target.τ)

    @testset "JDQR" begin
        @testset "Verbosity = $(verbosity)" for verbosity = 1:2
            pschur, residuals = jdqr(A,
                                     solver = BiCGStabl(n, max_mv_products = 10, l = 2),
                                     target = target,
                                     pairs = 5,
                                     tolerance = 1e-9,
                                     subspace_dimensions = 10:20,
                                     max_iter = 1000,
                                     verbosity = verbosity
                                     )
            verbosity == 1 && println("\n\n\n")
            λ = sort(real(pschur.values))

            @test λ ≈ sqrt.(2*(4998:0.5:5000)) rtol=1e-10
        end
    end

    @testset "JDQZ" begin
        @testset "Correction solver = $(label)" for (label,solver) in [("BiCGStabl", BiCGStabl(n, max_mv_products = 10, l = 2)),
                                                                       ("GMRES", GMRES(n))]
            @testset "Verbosity = $(verbosity)" for verbosity = 1:2
                pschur, residuals = jdqz(A, B,
                                         solver = solver,
                                         preconditioner = P,
                                         testspace = Harmonic,
                                         target = target,
                                         pairs = 5,
                                         tolerance = 1e-9,
                                         subspace_dimensions = 10:20,
                                         max_iter = 100,
                                         verbosity = verbosity
                                         )
                verbosity == 1 && println("\n\n\n")
                λ = sort(real(pschur.alphas ./ pschur.betas))

                @test λ ≈ 4998:5002 rtol=1e-10
            end
        end
    end
end
