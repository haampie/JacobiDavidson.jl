# Since Julia does not natively support an *explicit* ordering
# of the Schur diagonal elements (only a splitting in two groups)
# this code might be useful later, but not right now


# Returns a Givens rotation that swaps two consecutive
# diagonal elements [λ α; 0 μ] --> [μ β; 0 λ]
function given_swap(λ, μ, α)
  t = (μ - λ) / α
  c = 1 / √(1 + abs(t) ^ 2)
  s = c * t
  c, s
end

function swap_schur!(Q::AbstractMatrix, S::AbstractMatrix, from::Int, to::Int)
  # Move S[from, from] to S[to, to] via Givens rotations
  for l = from - 1 : -1 : to

    # Givens entries
    c, s = given_swap(S[l, l], S[l + 1, l + 1], S[l, l + 1])
    
    # Givens matrix
    G = [c conj(s); -s c]

    # Swap S[l, l] and S[l + 1, l + 1]
    coords = l : l + 1
    S[:, coords] = S[:, coords] * G'
    S[coords, :] = G * S[coords, :]
    Q[:, coords] = Q[:, coords] * G'

    # Zero out off-diaongal element (nonzero due to rounding errors)
    S[l + 1, l] = zero(S[l + 1, l])
  end
end

function permute_schur!(Q::AbstractMatrix, S::AbstractMatrix, permutation::Array{Int})
  # Permutes the eigenvalues on the diagonal of the Schur
  # matrix S via a series of Givens rotations.
  # 
  # Given the Schur factorization A = Q S Q', we apply 
  # a product of Givens rotations Q₂ s.t.
  # S₂ = Q₂ S Q₂' has a permuted diagonal: diag(S₂) = diag(S)[permutation]
  # and finally we get A = (Q Q₂') (Q₂ S Q₂') (Q Q₂')'
  # as a new Schur decomposition
  # 
  # `permutation` can be partial if only a few eigenvectors
  # are necessary.

  for to = 1 : length(permutation)
    from = permutation[to]
    swap_schur!(Q, S, from, to)
  end
end

function sort_schur!()