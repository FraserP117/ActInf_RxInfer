@testitem "NodeIdPlugin: model with the plugin" begin
    using Distributions

    import GraphPPL:
        NodeIdPlugin,
        EmptyID,
        NodeCreationOptions,
        PluginsCollection,
        add_atomic_factor_node!,
        create_model,
        getcontext,
        hasextra,
        getextra,
        by_nodeid

    model = create_model(plugins = PluginsCollection(NodeIdPlugin()))
    ctx = getcontext(model)

    @testset begin
        label1, nodedata1, properties1 = add_atomic_factor_node!(model, ctx, NodeCreationOptions(id = 1), Normal)
        label2, nodedata2, properties2 = add_atomic_factor_node!(model, ctx, NodeCreationOptions(id = "2"), Normal)
        label3, nodedata3, properties3 = add_atomic_factor_node!(model, ctx, NodeCreationOptions(id = :id3), Normal)
        label4, nodedata4, properties4 = add_atomic_factor_node!(model, ctx, NodeCreationOptions(), Normal)
        label5, nodedata5, properties5 = add_atomic_factor_node!(model, ctx, NodeCreationOptions(id = 4), Normal)
        label6, nodedata6, properties6 = add_atomic_factor_node!(model, ctx, NodeCreationOptions(id = 4), Normal)

        @test length(collect(filter(as_node(Normal), model))) === 6
        # Not all have the `id` label associated with them
        @test !all(n -> hasextra(model[n], :id), collect(filter(as_node(Normal), model)))
        # But at least some should have the `id` label associated with it
        @test any(n -> hasextra(model[n], :id), collect(filter(as_node(Normal), model)))

        # id = 1
        @test length(collect(filter(by_nodeid(1), model))) === 1
        @test model[first(collect(filter(by_nodeid(1), model)))] === nodedata1

        # id = "2"
        @test length(collect(filter(by_nodeid("2"), model))) === 1
        @test model[first(collect(filter(by_nodeid("2"), model)))] === nodedata2

        # id = :id3
        @test length(collect(filter(by_nodeid(:id3), model))) === 1
        @test model[first(collect(filter(by_nodeid(:id3), model)))] === nodedata3

        # id = 4
        @test length(collect(filter(by_nodeid(4), model))) === 2
        @test model[collect(filter(by_nodeid(4), model))[1]] === nodedata5
        @test model[collect(filter(by_nodeid(4), model))[2]] === nodedata6
    end

end