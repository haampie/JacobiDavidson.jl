"""
Useful for a basis for the search and test subspace.
"""
mutable struct SubSpace{matT <: StridedMatrix, matViewT <: StridedMatrix, vecViewT <: StridedVector}
    basis::matT
    all::matViewT
    curr::vecViewT
    prev::matViewT
end

SubSpace(V::AbstractMatrix) = SubSpace(V, view(V, :, 1 : 1), view(V, :, 1), view(V, :, 1 : 0))

function resize!(V::SubSpace, size::Int)
    V.prev = view(V.basis, :, 1 : size - 1)
    V.all = view(V.basis, :, 1 : size)
    V.curr = view(V.basis, :, size)

    V
end

"""
Holds a small projected matrix = W'AV and a view `curr` to the currently active part.
"""
mutable struct ProjectedMatrix{matT <: StridedMatrix, matViewT <: StridedMatrix}
    matrix::matT
    curr::matViewT
end

ProjectedMatrix(M::StridedMatrix) = ProjectedMatrix(M, view(M, 1 : 0, 1 : 0))

function resize!(M::ProjectedMatrix, size::Int)
    M.curr = view(M.matrix, 1 : size, 1 : size)
end

"""
`Q` is the pre-allocated work space of Schur vector
`R` will be the upper triangular factor

`Q` has the form [q₁, ..., qₖ, qₖ₊₁, qₖ₊₂, ..., q_pairs] where q₁ ... qₖ are already converged
and thus locked, while qₖ₊₁ is the active Schur vector that is converging, and the remaining
columns are just garbage data.

It is very useful to have views for these columns:
`locked` is the matrix view [q₁, ..., qₖ]
`active` is the column vector qₖ₊₁
`all` is the matrix view of the non-garbage part [q₁, ..., qₖ, qₖ₊₁]
`locked` is the number of locked Schur vectors
"""
mutable struct PartialSchur{matT <: StridedMatrix, vecT <: AbstractVector, matViewT <: StridedMatrix, vecViewT <: StridedVector}
    Q::matT
    values::vecT
    
    locked::matViewT
    active::vecViewT
    all::matViewT

    num_locked::Int
end

PartialSchur(Q, numT) = PartialSchur(
    Q,
    numT[],
    view(Q, :, 1 : 0), # Empty view initially
    view(Q, :, 1),
    view(Q, :, 1 : 1),
    0
)

function lock!(schur::PartialSchur)
    schur.num_locked += 1
    schur.locked = view(schur.Q, :, 1 : schur.num_locked)

    # Don't extend beyond the max number of Schur vectors
    if schur.num_locked < size(schur.Q, 2)
        schur.all = view(schur.Q, :, 1 : schur.num_locked + 1)
        schur.active = view(schur.Q, :, schur.num_locked + 1)
    end

    schur
end

mutable struct PartialGeneralizedSchur{subT <: SubSpace, vecT <: AbstractVector}
    Q::subT
    Z::subT
    alphas::vecT
    betas::vecT
end

PartialGeneralizedSchur(Q, Z, numT) = PartialGeneralizedSchur(
    SubSpace(Q),
    SubSpace(Z),
    numT[],
    numT[]
)

@inline function resize!(schur::PartialGeneralizedSchur, size::Int)
    resize!(schur.Q, size)
    resize!(schur.Z, size)
end

function shrink!(temporary, subspace::SubSpace, combination::StridedMatrix, dimension)
    tmp = view(temporary, :, 1 : dimension)
    A_mul_B!(tmp, subspace.all, combination)
    copy!(subspace.all, tmp)
    resize!(subspace, dimension)
end

function shrink!(M::ProjectedMatrix, replacement, dimension)
    copy!(view(M.matrix, 1 : dimension, 1 : dimension), replacement)
    resize!(M, dimension)
end