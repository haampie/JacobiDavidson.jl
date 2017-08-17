using Documenter, JacobiDavidson

makedocs(
    modules = [JacobiDavidson],
    clean = false,
    format = :html,
    sitename = "JacobiDavidson.jl",
    authors = "Harmen Stoppels",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "Correction equation" => "solvers.md"
    ]
)

deploydocs(
    repo = "github.com/haampie/JacobiDavidson.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)