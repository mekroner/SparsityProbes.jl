using SparsityProbes
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(SparsityProbes, :DocTestSetup, :(using SparsityProbes); recursive=true)
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)
makedocs(;
    modules=[SparsityProbes],
    plugins=[bib],
    authors=
        "Magnus Kroner <kroner@campus.tu-berlin.de>,
        Sai Krishna Mandagiri <mandagiri@campus.tu-berlin.de>,
        Yun-Ting Chiu <yun-ting.chiu@campus.tu-berlin.de>"
    ,
    sitename="SparsityProbes.jl",
    format=Documenter.HTML(;
        canonical="https://mekroner.github.io/SparsityProbes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "References" => "references.md"
    ],
)

deploydocs(;
    repo="github.com/mekroner/SparsityProbes.jl",
    devbranch="main",
)
