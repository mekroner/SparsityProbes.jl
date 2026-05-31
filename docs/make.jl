using SparsityProbes
using Documenter

DocMeta.setdocmeta!(SparsityProbes, :DocTestSetup, :(using SparsityProbes); recursive=true)

makedocs(;
    modules=[SparsityProbes],
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
    ],
)

deploydocs(;
    repo="github.com/mekroner/SparsityProbes.jl",
    devbranch="main",
)
