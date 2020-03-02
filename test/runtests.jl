using Test

using JacobiDavidson, LinearAlgebra

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

