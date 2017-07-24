import Base.LinAlg: Schur, GeneralizedSchur

export Near, SM, LM, LR, SR, LI, SI, schur_permutation, schur_sort!

abstract type Target end

# For finding eigenvalues near a target
struct Near{T} <: Target
  τ::Complex{T}
end

# For finding eigenvalues with the smallest magnitude
struct SM <: Target end

# For finding eigenvalues with the largest magnitude
struct LM <: Target end

# For finding eigenvalues with the largest real part
struct LR <: Target end

# For finding eigenvalues with the smallest real part
struct SR <: Target end

# For finding eigenvalues with the largest imaginary part
struct LI <: Target end

# For finding eigenvalues with the smallest imaginary part
struct SI <: Target end

eigval(F::GeneralizedSchur, idx::Int) = F.alpha[idx] / F.beta[idx]

schur_permutation(target::Near, F::GeneralizedSchur, k::Union{Integer, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx) - target.τ))

schur_permutation(target::LM, F::GeneralizedSchur, k::Union{Int, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx)), rev = true)

schur_permutation(target::SM, F::GeneralizedSchur, k::Union{Int, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> abs(eigval(F, idx)))

schur_permutation(target::LR, F::GeneralizedSchur, k::Union{Int, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> real(eigval(F, idx)), rev = true)

schur_permutation(target::SR, F::GeneralizedSchur, k::Union{Int, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> real(eigval(F, idx)))

schur_permutation(target::LI, F::GeneralizedSchur, k::Union{Int, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> imag(eigval(F, idx)), rev = true)

schur_permutation(target::SI, F::GeneralizedSchur, k::Union{Int, OrdinalRange}) = 
  selectperm(1 : length(F.alpha), k, by = idx -> imag(eigval(F, idx)))

function schur_sort!(target::Target, F::GeneralizedSchur, k::Union{Int, OrdinalRange})
  best = schur_permutation(target, F, k)
  perm = falses(length(F.alpha))
  perm[best] = true
  ordschur!(F, perm)
end

