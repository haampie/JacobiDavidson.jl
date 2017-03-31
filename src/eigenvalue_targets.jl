import Base.LinAlg: Schur

abstract Target

# For finding eigenvalues near a target
type Near <: Target
  target::Complex{Float64}
end

# For finding eigenvalues with the largest magnitude
type SM <: Target end

# For finding eigenvalues with the smallest magnitude
type LM <: Target end
    
# For finding eigenvalues with the largest real part
type LR <: Target end
    
# For finding eigenvalues with the smallest real part
type SR <: Target end

function schur_permutation(target::Near, F::Schur)
  sortperm(abs(F[:values] - target.target))
end

function schur_permutation(target::LM, F::Schur)
  sortperm(abs(F[:values]), rev = true)
end

function schur_permutation(target::SM, F::Schur)
  sortperm(abs(F[:values]))
end

function schur_permutation(target::LR, F::Schur)
  sortperm(real(F[:values]), rev = true)
end

function schur_permutation(target::SR, F::Schur)
  sortperm(real(F[:values]))
end
