module Tests

using JacobiDavidson, Test

function run()
    @testset "Schur permutations" begin include("schur_sort.jl") end
end

end

Tests.run()

nothing
