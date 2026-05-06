using InterpolativeQTT
using Documenter

DocMeta.setdocmeta!(InterpolativeQTT, :DocTestSetup, :(using InterpolativeQTT); recursive = true)

makedocs(;
    modules = [InterpolativeQTT],
    authors = "Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename = "InterpolativeQTT.jl",
    format = Documenter.HTML(;
        canonical = "https://github.com/tensor4all/InterpolativeQTT.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API Reference" => "apireference.md",
    ]

)

deploydocs(; repo = "github.com/tensor4all/InterpolativeQTT.jl.git", devbranch = "main", target = "build", branch = "gh-pages")
