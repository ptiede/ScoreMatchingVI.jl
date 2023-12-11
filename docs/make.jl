using ScoreMatchinVI
using Documenter

DocMeta.setdocmeta!(ScoreMatchinVI, :DocTestSetup, :(using ScoreMatchinVI); recursive=true)

makedocs(;
    modules=[ScoreMatchinVI],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/ScoreMatchinVI.jl/blob/{commit}{path}#{line}",
    sitename="ScoreMatchinVI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/ScoreMatchinVI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/ScoreMatchinVI.jl",
    devbranch="main",
)
