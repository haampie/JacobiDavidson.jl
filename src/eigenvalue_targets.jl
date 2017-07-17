import Base.LinAlg: Schur

abstract type Target end

# For finding eigenvalues near a target
immutable Near <: Target
  target::Complex{Float64}
end

# For finding eigenvalues with the largest magnitude
immutable SM <: Target end

# For finding eigenvalues with the smallest magnitude
immutable LM <: Target end

# For finding eigenvalues with the largest real part
immutable LR <: Target end

# For finding eigenvalues with the smallest real part
immutable SR <: Target end

schur_permutation(target::Near, θs) = sortperm(abs(θs - target.target))
schur_permutation(target::LM, θs) = sortperm(abs(θs), rev = true)
schur_permutation(target::SM, θs) = sortperm(abs(θs))
schur_permutation(target::LR, θs) = sortperm(real(θs), rev = true)
schur_permutation(target::SR, θs) = sortperm(real(θs))
