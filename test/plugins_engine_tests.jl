
@testitem "Unknown types should have unknown plygin type" begin
    import GraphPPL: plugin_type, UnknownPluginType

    struct SomeArbitraryType end

    @test plugin_type(SomeArbitraryType) isa UnknownPluginType
end

@testitem "`GraphPlugin{T}` should have the same plugin type as `T`" begin
    import GraphPPL: plugin_type, GraphPlugin, GraphGlobalPlugin, FactorNodePlugin, VariableNodePlugin

    struct SomeArbitraryTypeGlobal end
    struct SomeArbitraryTypeFactor end
    struct SomeArbitraryTypeVariable end

    GraphPPL.plugin_type(::Type{SomeArbitraryTypeGlobal}) = GraphGlobalPlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryTypeFactor}) = FactorNodePlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryTypeVariable}) = VariableNodePlugin()

    @test plugin_type(GraphPlugin{SomeArbitraryTypeGlobal}()) === GraphGlobalPlugin()
    @test plugin_type(GraphPlugin{SomeArbitraryTypeFactor}()) === FactorNodePlugin()
    @test plugin_type(GraphPlugin{SomeArbitraryTypeVariable}()) === VariableNodePlugin()
end

@testitem "Test plugin creation" begin
    import GraphPPL: GraphPlugin, PluginCollection, materialize_plugins

    struct SomeArbitraryPluginGlobal end
    struct SomeArbitraryPluginFactor end
    struct SomeArbitraryPluginVariable end

    GraphPPL.plugin_type(::Type{SomeArbitraryPluginGlobal}) = GraphPPL.GraphGlobalPlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryPluginFactor}) = GraphPPL.FactorNodePlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryPluginVariable}) = GraphPPL.VariableNodePlugin()

    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginGlobal}) = SomeArbitraryPluginGlobal()
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginFactor}) = SomeArbitraryPluginFactor()
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginVariable}) = SomeArbitraryPluginVariable()

    @test materialize_plugins(GraphPlugin(SomeArbitraryPluginGlobal)) == PluginCollection((SomeArbitraryPluginGlobal(),))
    @test materialize_plugins(GraphPlugin(SomeArbitraryPluginFactor)) == PluginCollection((SomeArbitraryPluginFactor(),))
    @test materialize_plugins(GraphPlugin(SomeArbitraryPluginVariable)) == PluginCollection((SomeArbitraryPluginVariable(),))

    plugin_specification = GraphPlugin(SomeArbitraryPluginGlobal) + GraphPlugin(SomeArbitraryPluginFactor) + GraphPlugin(SomeArbitraryPluginVariable)

    @test materialize_plugins(plugin_specification) == PluginCollection((SomeArbitraryPluginGlobal(), SomeArbitraryPluginFactor(), SomeArbitraryPluginVariable()))

    @test materialize_plugins(GraphPPL.GraphGlobalPlugin(), plugin_specification) == PluginCollection((SomeArbitraryPluginGlobal(),))
    @test materialize_plugins(GraphPPL.FactorNodePlugin(), plugin_specification) == PluginCollection((SomeArbitraryPluginFactor(),))
    @test materialize_plugins(GraphPPL.VariableNodePlugin(), plugin_specification) == PluginCollection((SomeArbitraryPluginVariable(),))
end

@testitem "Test that the same plugin cannot be attached twice" begin
    import GraphPPL: plugin_type, attach_plugin, PluginCollection

    struct SomeArbitraryPlugin end

    GraphPPL.plugin_type(::Type{SomeArbitraryPlugin}) = GraphGlobalPlugin()
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPlugin}) = SomeArbitraryPlugin()

    collection = PluginCollection()
    new_collection = attach_plugin(collection, SomeArbitraryPlugin)

    @test new_collection == PluginCollection((SomeArbitraryPlugin(),))
    # The second time it should as the plugin is already attached
    @test_throws ErrorException attach_plugin(new_collection, SomeArbitraryPlugin)
end

@testitem "Filter `PluginCollection` by `plugin_type`" begin
    import GraphPPL: plugin_type, GraphGlobalPlugin, FactorNodePlugin, VariableNodePlugin, UnknownPluginType, PluginCollection, materialize_plugins, getplugin

    struct SomeArbitraryPluginGlobal1 end
    struct SomeArbitraryPluginGlobal2 end

    GraphPPL.plugin_type(::Type{SomeArbitraryPluginGlobal1}) = GraphGlobalPlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryPluginGlobal2}) = GraphGlobalPlugin()

    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginGlobal1}) = SomeArbitraryPluginGlobal1()
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginGlobal2}) = SomeArbitraryPluginGlobal2()

    struct SomeArbitraryPluginFactor1 end
    struct SomeArbitraryPluginFactor2 end

    GraphPPL.plugin_type(::Type{SomeArbitraryPluginFactor1}) = FactorNodePlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryPluginFactor2}) = FactorNodePlugin()

    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginFactor1}) = SomeArbitraryPluginFactor1()
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginFactor2}) = SomeArbitraryPluginFactor2()

    struct SomeArbitraryPluginVariable1 end
    struct SomeArbitraryPluginVariable2 end

    GraphPPL.plugin_type(::Type{SomeArbitraryPluginVariable1}) = VariableNodePlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryPluginVariable2}) = VariableNodePlugin()

    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginVariable1}) = SomeArbitraryPluginVariable1()
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginVariable2}) = SomeArbitraryPluginVariable2()

    specification =
        GraphPlugin(SomeArbitraryPluginGlobal1) +
        GraphPlugin(SomeArbitraryPluginGlobal2) +
        GraphPlugin(SomeArbitraryPluginFactor1) +
        GraphPlugin(SomeArbitraryPluginFactor2) +
        GraphPlugin(SomeArbitraryPluginVariable1) +
        GraphPlugin(SomeArbitraryPluginVariable2)

    collection = materialize_plugins(specification)

    @test getplugin(collection, SomeArbitraryPluginGlobal1) isa SomeArbitraryPluginGlobal1
    @test getplugin(collection, SomeArbitraryPluginGlobal2) isa SomeArbitraryPluginGlobal2
    @test getplugin(collection, SomeArbitraryPluginFactor1) isa SomeArbitraryPluginFactor1
    @test getplugin(collection, SomeArbitraryPluginFactor2) isa SomeArbitraryPluginFactor2
    @test getplugin(collection, SomeArbitraryPluginVariable1) isa SomeArbitraryPluginVariable1
    @test getplugin(collection, SomeArbitraryPluginVariable2) isa SomeArbitraryPluginVariable2
end

@testitem "Get and modify a particular plugin instance within the collection" begin
    import GraphPPL: GraphGlobalPlugin, GraphPlugin, materialize_plugins, is_plugin_present, getplugin, modify_plugin!

    mutable struct SomeArbitraryPluginGlobal1
        field::Int
    end

    mutable struct SomeArbitraryPluginGlobal2
        field::Float64
    end

    # We don't include this plugin in the specification
    # to test that the `modify_plugin!` throws an error
    struct SomeArbitraryPluginGlobal3 end

    GraphPPL.plugin_type(::Type{SomeArbitraryPluginGlobal1}) = GraphGlobalPlugin()
    GraphPPL.plugin_type(::Type{SomeArbitraryPluginGlobal2}) = GraphGlobalPlugin()

    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginGlobal1}) = SomeArbitraryPluginGlobal1(0)
    GraphPPL.materialize_plugin(::Type{SomeArbitraryPluginGlobal2}) = SomeArbitraryPluginGlobal2(0.0)

    specification = GraphPlugin(SomeArbitraryPluginGlobal1) | GraphPlugin(SomeArbitraryPluginGlobal2)
    collection = materialize_plugins(specification)

    @test is_plugin_present(collection, SomeArbitraryPluginGlobal1)
    @test is_plugin_present(collection, SomeArbitraryPluginGlobal2)
    @test !is_plugin_present(collection, SomeArbitraryPluginGlobal3)

    plugin1 = getplugin(collection, SomeArbitraryPluginGlobal1)
    plugin2 = getplugin(collection, SomeArbitraryPluginGlobal2)

    @test plugin1 isa SomeArbitraryPluginGlobal1
    @test plugin2 isa SomeArbitraryPluginGlobal2

    @test_throws ErrorException getplugin(collection, SomeArbitraryPluginGlobal3)
    @test getplugin(collection, SomeArbitraryPluginGlobal3, Val(false)) === GraphPPL.MissingPlugin()

    @test plugin1.field === 0
    @test plugin2.field === 0.0

    for value in 1:10
        modify_plugin!(collection, SomeArbitraryPluginGlobal1) do plugin
            plugin.field = value
        end
        @test plugin1.field === value
        @test plugin2.field === 0.0
    end

    for value in float.(1:10)
        modify_plugin!(collection, SomeArbitraryPluginGlobal2) do plugin
            plugin.field = value
        end
        @test plugin1.field === 10 # the last value in the previous test was 10
        @test plugin2.field === value
    end

    @test_throws ErrorException modify_plugin!(collection, SomeArbitraryPluginGlobal3) do plugin
        nothing
    end

    @test collection === modify_plugin!(collection, SomeArbitraryPluginGlobal3, Val(false)) do plugin
        nothing
    end
end