import LinearAlgebra: Schur, GeneralizedSchur

export Near, SM, LM, LR, SR, LI, SI, schur_permutation, schur_sort!

abstract type Target end

# For finding eigenvalues near a target
struct Near{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest magnitude
struct SM{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the largest magnitude
struct LM{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the largest real part
struct LR{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest real part
struct SR{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the largest imaginary part
struct LI{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest imaginary part
struct SI{T} <: Target
  τ::Complex{T}
end

eigval(F::GeneralizedSchur, idx::Int) = F.alpha[idx] / F.beta[idx]

schur_permutation(target::Near, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx) - target.τ))

schur_permutation(target::LM, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx)), rev = true)

schur_permutation(target::SM, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx)))

schur_permutation(target::LR, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> real(eigval(F, idx)), rev = true)

schur_permutation(target::SR, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> real(eigval(F, idx)))

schur_permutation(target::LI, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> imag(eigval(F, idx)), rev = true)

schur_permutation(target::SI, F::GeneralizedSchur, k::OrdinalRange) =
  partialsortperm(1 : length(F.alpha), k, by = idx -> imag(eigval(F, idx)))

function schur_sort!(target::Target, F::GeneralizedSchur, k::OrdinalRange)
  best = vec(schur_permutation(target, F, k))
  perm = falses(length(F.alpha))
  perm[best] .= true
  ordschur!(F, perm)
end

schur_sort!(target::Target, F::GeneralizedSchur, k::Int) =
    schur_sort!(target, F, k:k)
