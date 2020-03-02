import LinearAlgebra: Schur, GeneralizedSchur

export Near, SmallestMagnitude, LargestMagnitude, 
       LargestRealPart, SmallestRealPart, 
       LargestImaginaryPart, SmallestImaginaryPart, 
       schur_permutation, schur_sort!

abstract type Target end

# For finding eigenvalues near a target
struct Near{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest magnitude
struct SmallestMagnitude{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the largest magnitude
struct LargestMagnitude{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the largest real part
struct LargestRealPart{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest real part
struct SmallestRealPart{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the largest imaginary part
struct LargestImaginaryPart{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest imaginary part
struct SmallestImaginaryPart{T} <: Target
  τ::Complex{T}
end

eigval(F::GeneralizedSchur, idx::Int) = F.alpha[idx] / F.beta[idx]

schur_permutation(target::Near, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx) - target.τ))

schur_permutation(target::LargestMagnitude, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx)), rev = true)

schur_permutation(target::SmallestMagnitude, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx)))

schur_permutation(target::LargestRealPart, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> real(eigval(F, idx)), rev = true)

schur_permutation(target::SmallestRealPart, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> real(eigval(F, idx)))

schur_permutation(target::LargestImaginaryPart, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> imag(eigval(F, idx)), rev = true)

schur_permutation(target::SmallestImaginaryPart, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> imag(eigval(F, idx)))

function schur_sort!(target::Target, F::GeneralizedSchur, k::OrdinalRange)
  best = vec(schur_permutation(target, F, k))
  perm = falses(length(F.alpha))
  perm[best] .= true
  ordschur!(F, perm)
end

schur_sort!(target::Target, F::GeneralizedSchur, k::Int) =
    schur_sort!(target, F, k:k)
