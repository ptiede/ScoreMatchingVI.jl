using ScoreMatchingVI
using Documenter

DocMeta.setdocmeta!(ScoreMatchingVI, :DocTestSetup, :(using ScoreMatchingVI); recursive=true)

makedocs(;
    modules=[ScoreMatchingVI],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/ScoreMatchingVI.jl/blob/{commit}{path}#{line}",
    sitename="ScoreMatchingVI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/ScoreMatchingVI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/ScoreMatchingVI.jl",
    devbranch="main",
)
