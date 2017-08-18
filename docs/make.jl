using Documenter, JacobiDavidson

makedocs(
    modules = [JacobiDavidson],
    clean = true,
    format = :html,
    sitename = "JacobiDavidson.jl",
    authors = "Harmen Stoppels",
    pages = [
        "Home" => "index.md",
        "Correction equation" => "solvers.md"
    ]
)

deploydocs(
    repo = "github.com/haampie/JacobiDavidson.jl.git",
    target = "build",
    osname = "linux",
    julia  = "0.6",
    deps = nothing,
    make = nothing
)