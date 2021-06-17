export DGKS, ClassicalGramSchmidt, ModifiedGramSchmidt
export orthogonalize_and_normalize!

abstract type OrthogonalizationMethod end
struct DGKS <: OrthogonalizationMethod end
struct ClassicalGramSchmidt <: OrthogonalizationMethod end
struct ModifiedGramSchmidt <: OrthogonalizationMethod end

# Default to MGS, good enough for solving linear systems.
@inline orthogonalize_and_normalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}) where {T} = orthogonalize_and_normalize!(V, w, h, ModifiedGramSchmidt)

function orthogonalize_and_normalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, kind::Type{<:OrthogonalizationMethod}) where {T}
    nrm = just_orthogonalize!(V, w, h, kind)
    if isnothing(nrm)
        nrm = norm(w)
    end

    lmul!(one(T) / nrm, w)

    nrm
end

function cgs!(w::AbstractVector{T}, V, h) where T
    # Orthogonalize using BLAS-2 ops
    mul!(h, V', w)
    mul!(w, V, h, -one(T), one(T))
end

function just_orthogonalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, ::Type{ClassicalGramSchmidt}) where {T}
    cgs!(V, w, h)
    nothing
end

function just_orthogonalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, ::Type{ModifiedGramSchmidt}) where {T}
    # Orthogonalize using BLAS-1 ops and column views.
    for i = 1 : size(V, 2)
        column = view(V, :, i)
        h[i] = dot(column, w)
        BLAS.axpy!(-h[i], column, w)
    end

    nothing
end

function just_orthogonalize!(V::StridedMatrix{T}, w::StridedVector{T}, ::Type{DGKS}) where {T}
    h = Vector{T}(undef, size(V, 2))
    cgs!(w, V, h)
    nrm = norm(w)

    # Constant used by ARPACK.
    η = one(real(T)) / √2

    projection_size = norm(h)

    # Repeat as long as the DGKS condition is satisfied
    # Typically this condition is true only once.
    while nrm < η * projection_size
        cgs!(w, V, h)
        projection_size = norm(h)
        nrm = norm(w)
    end

    nothing
end

function just_orthogonalize!(V::StridedMatrix{T}, w::StridedVector{T}, h::StridedVector{T}, ::Type{DGKS}) where {T}
    cgs!(w, V, h)
    nrm = norm(w)

    # Constant used by ARPACK.
    η = one(real(T)) / √2

    projection_size = norm(h)

    correction = Vector{T}(undef, 0)

    # Repeat as long as the DGKS condition is satisfied
    # Typically this condition is true only once.
    while nrm < η * projection_size
        resize!(correction, length(h))
        cgs!(w, V, correction)
        projection_size = norm(correction)
        BLAS.axpy!(one(T), correction, h)
        nrm = norm(w)
    end

    nrm
end
