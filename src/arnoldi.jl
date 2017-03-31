function expand!(V::AbstractMatrix, H::AbstractMatrix, w::AbstractVector, k::Int)

  # Orthogonalize using (single) Modified Gramm-Schmidt
  for j = 1 : k
    H[j, k] = dot(w, V[:, j])
    w -= H[j, k] * V[:, j]
  end

  H[k + 1, k] = norm(w)
  V[:, k + 1] = w / H[k + 1, k]
end