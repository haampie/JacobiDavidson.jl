using JacobiDavidson
using Documenter

makedocs(;
    modules=[JacobiDavidson],
    authors="Harmen Stoppels",
    repo="https://github.com/haampie/JacobiDavidson.jl/blob/{commit}{path}#{line}",
    sitename="JacobiDavidson.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://haampie.github.io/JacobiDavidson.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Correction equation" => "solvers.md"
    ],
    doctest = false
)

deploydocs(;
    repo="github.com/haampie/JacobiDavidson.jl",
)
