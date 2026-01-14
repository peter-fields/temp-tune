############ matplotlib_helpers.jl (updated) ############
# Robust rc setup using rcParams.update + safe getters.
# Works with Option B (usetex=false). Keeps SVG text editable.


# --- Notebook setup (isolate config; update rc safely; guard rails) ----------
"""
    setup_notebook_for_paper(; venue="revtex", backend="Agg", enforce_local_conda=true)

Project-safe initialization for PyPlot with paper-ready defaults:
- Sets `MPLCONFIGDIR = joinpath(projectdir(), ".mplconfig")` to avoid user rc.
- Sets `CONDA_JL_HOME = joinpath(projectdir(), ".conda")` for project-local Python.
- Selects Matplotlib backend (default `"Agg"`).
"""

function setup_notebook_for_paper(; backend::String="Agg")
    # 0) Isolate Matplotlib from ~/.matplotlib
    ENV["MPLCONFIGDIR"] = joinpath(projectdir(), ".mplconfig")
    try; mkpath(ENV["MPLCONFIGDIR"]); catch; end

    # 1) Keep Conda local to the project; set backend
    ENV["CONDA_JL_HOME"] = joinpath(projectdir(), ".conda")
    ENV["MPLBACKEND"]    = backend

    # set_plot_style_for_paper(; venue=venue)

    return nothing
end

function using_Py( enforce_local_conda=true )
    if !haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] != joinpath(projectdir(), ".mplconfig"))
        @warn """
        ENV["MPLCONFIGDIR"] not set up yet. Run setup_notebook_for_paper first! Exiting...
        """
        return nothing
    end
    
    @eval using PyCall
        expected_root = joinpath(projectdir(), ".conda")
        current_py    = PyCall.python
        using_local   = occursin(expected_root, current_py)
        if enforce_local_conda && !using_local
            @warn "PyCall is using a different Python:
              current : $current_py
              expected: $expected_root
            Restart the kernel after rebuilding PyCall in this project:
              ENV[\"CONDA_JL_HOME\"] = \"$expected_root\"
              ENV[\"PYTHON\"] = \"\"
              import Pkg; Pkg.build(\"PyCall\")"
            return nothing
        end

        # 3) Load PyPlot and handle potential API mismatch
        try
            @eval using PyPlot
        catch err
            msg = sprint(showerror, err)
            if occursin("register_cmap", msg)
                @warn "PyPlot vs Matplotlib mismatch detected.
                Fix one of:
                  • import Pkg; Pkg.update(\"PyPlot\")
                  • using Conda; Conda.add(\"matplotlib=3.7.*\")   # pin older Matplotlib
                Then restart the kernel."
                        end
            rethrow(err)
        end
end

function setup_rc_reset()
    _mpl = PyPlot.matplotlib
    _rc  = _mpl["rcParams"]
    _pget(key::String) = _rc[:get](key, "(missing)")   # Python dict get(key, default)

    current_py = PyCall.python
    
    # 4) Hard reset, then apply style (rc update)
    _mpl[:rcdefaults]()
    
    # 5) Print concise sanity info (safe getters)
    # println("Python     : ", current_py)
    # println("Matplotlib : ", pyimport("matplotlib").__version__, " | backend=", PyPlot.matplotlib.get_backend())
    # println("Sizes (pt) : label=", _pget("axes.labelsize"),
    #         "  tick=", _pget("xtick.labelsize"),
    #         "  legend=", _pget("legend.fontsize"))
    # println(" setup and check complete. exiting...\n")
    return nothing
end

function set_plot_style_for_paper(; venue::String="revtex",
                                   label_pt::Union{Nothing,Real}=nothing,
                                   tick_pt::Union{Nothing,Real}=nothing,
                                   legend_pt::Union{Nothing,Real}=nothing,
                                   math_font::String="cm",
                                   use_tex::Bool=false, columns::Int=1)
    #columns argument meaningless for now. keep for later use if needed.

    defaults = Dict(
        "revtex"   => (label=10.0, tick=8.0,  legend=8.0),
        "nature"   => (label=9.0,  tick=7.0,  legend=7.0),
        "science"  => (label=9.0,  tick=7.0,  legend=7.0),
        "elsevier" => (label=9.0,  tick=7.5,  legend=7.5),
        "springer" => (label=9.0,  tick=7.5,  legend=7.5),
    )
    venue = haskey(defaults, venue) ? venue : "revtex"
    
    _mpl = PyPlot.matplotlib
    _rc  = _mpl["rcParams"]
    _pget(key::String) = _rc[:get](key, "(missing)")   # Python dict get(key, default)

    L = isnothing(label_pt)  ? defaults[venue].label  : Float64(label_pt)
    T = isnothing(tick_pt)   ? defaults[venue].tick   : Float64(tick_pt)
    G = isnothing(legend_pt) ? defaults[venue].legend : Float64(legend_pt)

    # Choose a sensible sans-serif family available on the system
    chosen = "DejaVu Sans"
    try
        fm = _mpl["font_manager"]
        names = Set([String(f[:name]) for f in fm[:fontManager][:ttflist]])
        chosen = "Nimbus Sans" in names ? "Nimbus Sans" :
                 "Arial"     in names ? "Arial"     :
                 "DejaVu Sans"
    catch
        # fall back to DejaVu Sans silently
    end

    # Reset FIRST if you want a clean slate (the caller can also do this)
    # _mpl[:rcdefaults]()  # leave commented; call it in setup_notebook or your startup

    # Atomically update rcParams (no missing-key lookups)
    _rc.update(PyDict(Dict(
        "text.usetex"        => use_tex,
        "mathtext.fontset"   => math_font,
        "font.family"        => "sans-serif",
        "font.sans-serif"    => [chosen, "Arial", "DejaVu Sans"],
        "font.size"          => L,
        "axes.labelsize"     => L,
        "axes.labelpad"      => 4.0,
        "axes.titlepad"      => 6.0,
        "xtick.labelsize"    => T,
        "ytick.labelsize"    => T,
        "legend.fontsize"    => G,
        "svg.fonttype"       => "path",  # keep text editable
        "pdf.fonttype"       => 42,
        "ps.fonttype"        => 42,
        "lines.linewidth"    => 1.,
        "lines.markersize"   => 5.
    )))
    
    current_py = PyCall.python
    
    # Report what actually stuck (safe getters)
    println("Plot style set → venue=$venue, font='$chosen'")
    println("Sizes (pt): label=", _pget("axes.labelsize"),
            "  tick=", _pget("xtick.labelsize"),
            "  legend=", _pget("legend.fontsize"),
            "  mathtext.fontset=", _pget("mathtext.fontset"))
    println("Python     : ", current_py)
    println("Matplotlib : ", pyimport("matplotlib").__version__, " | backend=", 
        PyPlot.matplotlib.get_backend())
     println(" set plot style finished. exiting...\n")
    
    return nothing
end

function set_size_for_paper(columns::Int=1, aspect::Float64=0.62;
                            venue::String="revtex", fraction::Float64=1.0,
                            override_pt::Union{Nothing,Float64}=nothing,
                            override_mm::Union{Nothing,Float64}=nothing)
    if override_pt !== nothing && override_mm !== nothing
        throw(ArgumentError("Specify only one of override_pt or override_mm."))
    end
    IN_PER_PT = 1.0 / 72.27
    PT_PER_MM = 72.27 / 25.4
    presets_pt = Dict(
        "revtex"   => Dict(1 => 246.0,                    2 => 510.0),
        "nature"   => Dict(1 => 89.0  * PT_PER_MM,        2 => 183.0 * PT_PER_MM),
        "science"  => Dict(1 => 85.0  * PT_PER_MM,        2 => 176.0 * PT_PER_MM),
        "elsevier" => Dict(1 => 84.0  * PT_PER_MM,        2 => 174.0 * PT_PER_MM),
        "springer" => Dict(1 => 88.0  * PT_PER_MM,        2 => 183.0 * PT_PER_MM),
    )
    venue = haskey(presets_pt, venue) ? venue : "revtex"
    width_pt = override_pt !== nothing ? override_pt :
               override_mm !== nothing ? override_mm * PT_PER_MM :
               presets_pt[venue][columns]
    width_in  = width_pt * IN_PER_PT * fraction
    height_in = width_in * aspect
    return (width_in, height_in)
end

function inspect_fonts_resolved(ax)
    # Requires PyPlot already loaded
    mpl = PyPlot.matplotlib
    fm  = mpl[:font_manager]
    rc  = mpl["rcParams"]

    println("=== Font inspection (resolved) ===")

    # collect text artists
    items = [
        ("Title", ax.title),
        ("X label", ax.xaxis.label),
        ("Y label", ax.yaxis.label),
    ]
    append!(items, [("X tick[$i]", t) for (i,t) in enumerate(ax.get_xticklabels()[1:5:end])])
    append!(items, [("Y tick[$i]", t) for (i,t) in enumerate(ax.get_yticklabels()[1:5:end])])
    if (leg = ax.get_legend()) !== nothing
        append!(items, [("Legend[$i]", t) for (i,t) in enumerate(leg.get_texts())])
    end

    for (lbl, t) in items
        txt   = t.get_text()
        sz    = t.get_size()
        fam   = t.get_fontfamily()            # generic family list
        fp    = t.get_fontproperties()        # FontProperties (requested)
        ismath = occursin("\$", String(txt))  # crude mathtext detection

        # # Resolve to the actual font file Matplotlib will use
        # (fallback_to_default=true so we always get something)
        fontpath = fm[:findfont](fp; fallback_to_default=true)
        # # Read the face name from that file
        resolved = mpl[:font_manager][:FontProperties](fname=fontpath).get_name()

        println("[$lbl] size=$(sz), family=$(fam), resolved=\"$(resolved)\"")
        # println("      path=$(fontpath)")
        if ismath
            println("      NOTE: contains math; mathtext is rendered as vector paths (fontset=$(rc[:get]("mathtext.fontset","(default)")))") 
        end
    end

    println("Global: text.usetex=$(rc[:get]("text.usetex", false)), mathtext.fontset=$(rc[:get]("mathtext.fontset","(default)"))")
    println("=== End ===")
    return nothing
end


function inspect_fonts(ax)
    mpl = PyPlot.matplotlib
    rc  = mpl["rcParams"]

    println("=== Font inspection for Axes ===")

    # Axes labels
    for (lbl, obj) in [("X label", ax.xaxis.label),
                       ("Y label", ax.yaxis.label),
                       ("Title", ax.title)]
        println(lbl, ": text='", obj.get_text(),
                "', size=", obj.get_size(),
                ", family=", obj.get_fontfamily())
    end

    # Ticks
    xt = ax.get_xticklabels()
    yt = ax.get_yticklabels()
    if !isempty(xt)
        println("X tick[0]: '", xt[1].get_text(),
                "', size=", xt[1].get_size(),
                ", family=", xt[1].get_fontfamily())
    end
    if !isempty(yt)
        println("Y tick[0]: '", yt[1].get_text(),
                "', size=", yt[1].get_size(),
                ", family=", yt[1].get_fontfamily())
    end

    # Legend
    leg = ax.get_legend()
    if leg !== nothing
        for (i, t) in enumerate(leg.get_texts())
            println("Legend[", i, "]: text='", t.get_text(),
                    "', size=", t.get_size(),
                    ", family=", t.get_fontfamily())
        end
    end

    # Global mathtext info
    println("Mathtext fontset = ", rc[:get]("mathtext.fontset", "(default)"))

    println("=== End font inspection ===")
    return nothing
end

function rc_update_one( key, val )
    _rc=PyPlot.matplotlib["rcParams"]
    old_val=_rc._get(key)
    println("old value of \"$key\" was \"$(_rc._get(key))\"")
    _rc.update(PyDict(Dict( key => val )))
    return old_val
end


# """
#     pdf_to_svg(pdfpath::AbstractString)

# Convert a PDF to an SVG using `pdftocairo -svg`.
# The resulting SVG is written to the same directory as the PDF, with the same basename.
# Requires that `pdftocairo` (from Poppler) is installed and on PATH.
# """
# function pdf_to_svg(pdfpath::AbstractString)
#     # ensure file exists
#     isfile(pdfpath) || error("File not found: $pdfpath")

#     # split path into directory and filename
#     dir, base = splitdir(pdfpath)
#     stem, _ = splitext(base)

#     outpath = joinpath(dir, stem * ".svg")

#     run(`pdf2svg $pdfpath $outpath`)

#     return outpath
# end


