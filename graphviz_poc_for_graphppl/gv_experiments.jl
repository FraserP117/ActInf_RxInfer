using RxInfer, Distributions, Random, GraphPlot, Cairo, Graphs, MetaGraphsNext, GraphPPL, GraphViz, Dictionaries, Plots, StableRNGs, LinearAlgebra, StatsPlots, LaTeXStrings, DataFrames, CSV, GLM



# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(y, a, b)
    # We endow θ parameter of our model with some prior
    θ ~ Beta(a, b)
    # or, in this particular case, the `Uniform(0.0, 1.0)` prior also works:
    # θ ~ Uniform(0.0, 1.0)

    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

conditioned = coin_model(a = 2.0, b = 7.0) | (y = [ true, false, true ], )

# `Create` the actual graph of the model conditioned on the data
rxi_model = RxInfer.create_model(conditioned)

# get the undrelying GraphPPL.Model
gppl_model = RxInfer.getmodel(rxi_model)

# Extract the MetaGraphsNext graph
meta_graph = gppl_model.graph

# The existing plotting functionality:
g_plot = GraphPlot.gplot(
    meta_graph,
    layout=spring_layout,
    nodelabel=collect(labels(meta_graph)),
    nodelabelsize=1.0,
    NODESIZE=0.05, # diameter of the nodes,
    nodelabelc="blue",
    nodelabeldist=0.0,
    nodefillc="cyan",
)

display(g_plot)