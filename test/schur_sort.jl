using JacobiDavidson, Base.Test

A = rand(Complex128, 10, 10)
B = rand(Complex128, 10, 10)
F = schurfact(A, B)

τ = 1.0 + 0.0im
eigenvalues = F.alpha ./ F.beta

# Test whether the ordering is correct
closest_to_one = schur_permutation(Near(τ), F, 1)
@test abs(eigenvalues[closest_to_one] - τ) ≤ minimum(abs.(eigenvalues .- τ))

largest_magnitude = schur_permutation(JacobiDavidson.LM(), F, 1)
@test abs(eigenvalues[largest_magnitude]) ≥ maximum(abs.(eigenvalues))

smallest_magnitude = schur_permutation(SM(), F, 1)
@test abs(eigenvalues[smallest_magnitude]) ≤ minimum(abs.(eigenvalues))

largest_real_part = schur_permutation(LR(), F, 1)
@test real(eigenvalues[largest_real_part]) ≥ maximum(real.(eigenvalues))

smallest_real_part = schur_permutation(SR(), F, 1)
@test real(eigenvalues[smallest_real_part]) ≤ minimum(real.(eigenvalues))

largest_imaginary_part = schur_permutation(LI(), F, 1)
@test imag(eigenvalues[largest_imaginary_part]) ≥ maximum(imag.(eigenvalues))

smallest_imaginary_part = schur_permutation(SI(), F, 1)
@test imag(eigenvalues[smallest_imaginary_part]) ≤ minimum(imag.(eigenvalues))

# Test whether one can reorder the Schur decomp with multiple eigenvalues
take = 1 : 3; keep = 4 : 10
schur_sort!(Near(τ), F, take)
sorted = F.alpha ./ F.beta
@test maximum(abs.(sorted[take] .- τ)) ≤ minimum(abs.(sorted[keep] .- τ))

# Test whether one can reorder the Schur decomp with a single eigenvalue
schur_sort!(SM(), F, 1)
sorted = F.alpha ./ F.beta
@test all(abs(sorted[1]) .≤ abs.(sorted[2 : end]))