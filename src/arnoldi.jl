function expand!(V::AbstractMatrix, H::AbstractMatrix, k::Int)

  # Orthogonalize using (single) Modified Gramm-Schmidt
  for j = 1 : k
    H[j, k] = dot(V[:, k + 1], V[:, j])
    V[:, k + 1] -= H[j, k] * V[:, j]
  end

  H[k + 1, k] = norm(V[:, k + 1])
  V[:, k + 1] *= 1.0 / H[k + 1, k]
end
