# Identity preconditioner
struct Identity end

Base.:\(::Identity, x) = copy(x)
Base.A_ldiv_B!(::Identity, x) = x
Base.A_ldiv_B!(y, ::Identity, x) = copy!(y, x)