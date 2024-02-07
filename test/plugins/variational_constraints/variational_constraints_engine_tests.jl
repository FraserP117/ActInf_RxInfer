@testitem "FactorizationConstraintEntry" begin
    import GraphPPL: FactorizationConstraintEntry, IndexedVariable

    # Test 1: Test FactorisationConstraintEntry
    @test FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))) isa FactorizationConstraintEntry

    a = FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)))
    b = FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)))
    @test a == b
    c = FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing), IndexedVariable(:z, nothing)))
    @test a != c
    d = FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:p, nothing)))
    @test a != d

    # Test 2: Test FactorisationConstraintEntry with mixed IndexedVariable types
    a = FactorizationConstraintEntry((IndexedVariable(:x, 1), IndexedVariable(:y, nothing)))
end

@testitem "CombinedRange" begin
    import GraphPPL: CombinedRange, is_splitted, FunctionalIndex, IndexedVariable
    for left in 1:3, right in 5:8
        cr = CombinedRange(left, right)

        @test firstindex(cr) === left
        @test lastindex(cr) === right
        @test !is_splitted(cr)
        @test length(cr) === lastindex(cr) - firstindex(cr) + 1

        for i in left:right
            @test i ∈ cr
            @test !((i + lastindex(cr) + 1) ∈ cr)
        end
    end
    range = CombinedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))
    @test firstindex(range).f === firstindex
    @test lastindex(range).f === lastindex
    @test_throws MethodError length(range)

    # Test IndexedVariable with CombinedRange equality
    lhs = IndexedVariable(:x, CombinedRange(1, 2))
    rhs = IndexedVariable(:x, CombinedRange(1, 2))
    @test lhs == rhs
    @test lhs === rhs
    @test lhs != IndexedVariable(:x, CombinedRange(1, 3))
    @test lhs !== IndexedVariable(:x, CombinedRange(1, 3))
    @test lhs != IndexedVariable(:y, CombinedRange(1, 2))
    @test lhs !== IndexedVariable(:y, CombinedRange(1, 2))
end

@testitem "SplittedRange" begin
    import GraphPPL: SplittedRange, is_splitted, FunctionalIndex, IndexedVariable
    for left in 1:3, right in 5:8
        cr = SplittedRange(left, right)

        @test firstindex(cr) === left
        @test lastindex(cr) === right
        @test is_splitted(cr)
        @test length(cr) === lastindex(cr) - firstindex(cr) + 1

        for i in left:right
            @test i ∈ cr
            @test !((i + lastindex(cr) + 1) ∈ cr)
        end
    end
    range = SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))
    @test firstindex(range).f === firstindex
    @test lastindex(range).f === lastindex
    @test_throws MethodError length(range)

    # Test IndexedVariable with SplittedRange equality
    lhs = IndexedVariable(:x, SplittedRange(1, 2))
    rhs = IndexedVariable(:x, SplittedRange(1, 2))
    @test lhs == rhs
    @test lhs === rhs
    @test lhs != IndexedVariable(:x, SplittedRange(1, 3))
    @test lhs !== IndexedVariable(:x, SplittedRange(1, 3))
    @test lhs != IndexedVariable(:y, SplittedRange(1, 2))
    @test lhs !== IndexedVariable(:y, SplittedRange(1, 2))
end

@testitem "__factorization_specification_resolve_index" begin
    using GraphPPL
    import GraphPPL: __factorization_specification_resolve_index, FunctionalIndex, CombinedRange, SplittedRange, NodeLabel, ResizableArray

    collection = ResizableArray(NodeLabel, Val(1))
    for i in 1:10
        collection[i] = NodeLabel(:x, i)
    end

    # Test 1: Test __factorization_specification_resolve_index with FunctionalIndex
    index = FunctionalIndex{:begin}(firstindex)
    @test __factorization_specification_resolve_index(index, collection) === firstindex(collection)

    @test_throws ErrorException __factorization_specification_resolve_index(index, collection[1])

    # Test 2: Test __factorization_specification_resolve_index with CombinedRange
    index = CombinedRange(1, 5)
    @test __factorization_specification_resolve_index(index, collection) === index
    index = CombinedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))
    @test __factorization_specification_resolve_index(index, collection) === CombinedRange(1, 10)
    index = CombinedRange(5, FunctionalIndex{:end}(lastindex))
    @test __factorization_specification_resolve_index(index, collection) === CombinedRange(5, 10)
    index = CombinedRange(1, 20)
    @test_throws ErrorException __factorization_specification_resolve_index(index, collection)

    @test_throws ErrorException __factorization_specification_resolve_index(index, collection[1])

    # Test 3: Test __factorization_specification_resolve_index with SplittedRange
    index = SplittedRange(1, 5)
    @test __factorization_specification_resolve_index(index, collection) === index
    index = SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))
    @test __factorization_specification_resolve_index(index, collection) === SplittedRange(1, 10)
    index = SplittedRange(5, FunctionalIndex{:end}(lastindex))
    @test __factorization_specification_resolve_index(index, collection) === SplittedRange(5, 10)
    index = SplittedRange(1, 20)
    @test_throws ErrorException __factorization_specification_resolve_index(index, collection)

    @test_throws ErrorException __factorization_specification_resolve_index(index, collection[1])

    # Test 4: Test __factorization_specification_resolve_index with Array of indices
    index = SplittedRange(
        [FunctionalIndex{:begin}(firstindex), FunctionalIndex{:begin}(firstindex)],
        [FunctionalIndex{:end}(lastindex), FunctionalIndex{:end}(lastindex)]
    )
    collection = GraphPPL.ResizableArray(GraphPPL.NodeLabel, Val(2))
    for i in 1:3
        for j in 1:5
            collection[i, j] = GraphPPL.NodeLabel(:x, i * j)
        end
    end
end

@testitem "factorization_split" begin
    import GraphPPL: factorization_split, FactorizationConstraintEntry, IndexedVariable, FunctionalIndex, CombinedRange, SplittedRange

    # Test 1: Test factorization_split with single split
    @test factorization_split(
        (FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:begin}(firstindex)),)),),
        (FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:end}(lastindex)),)),)
    ) == (
        FactorizationConstraintEntry((
            IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
        ),),
    )

    @test factorization_split(
        (
            FactorizationConstraintEntry((IndexedVariable(:y, nothing),)),
            FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:begin}(firstindex)),))
        ),
        (
            FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:end}(lastindex)),)),
            FactorizationConstraintEntry((IndexedVariable(:z, nothing),))
        )
    ) == (
        FactorizationConstraintEntry((IndexedVariable(:y, nothing),)),
        FactorizationConstraintEntry((
            IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
        )),
        FactorizationConstraintEntry((IndexedVariable(:z, nothing),))
    )

    @test factorization_split(
        (
            FactorizationConstraintEntry((
                IndexedVariable(:x, FunctionalIndex{:begin}(firstindex)), IndexedVariable(:y, FunctionalIndex{:begin}(firstindex))
            )),
        ),
        (
            FactorizationConstraintEntry((
                IndexedVariable(:x, FunctionalIndex{:end}(lastindex)), IndexedVariable(:y, FunctionalIndex{:end}(lastindex))
            )),
        )
    ) == (
        FactorizationConstraintEntry((
            IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
            IndexedVariable(:y, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex)))
        )),
    )

    # Test factorization_split with only FactorizationConstraintEntrys
    @test factorization_split(
        FactorizationConstraintEntry((
            IndexedVariable(:x, FunctionalIndex{:begin}(firstindex)), IndexedVariable(:y, FunctionalIndex{:begin}(firstindex))
        )),
        FactorizationConstraintEntry((
            IndexedVariable(:x, FunctionalIndex{:end}(lastindex)), IndexedVariable(:y, FunctionalIndex{:end}(lastindex))
        ))
    ) == FactorizationConstraintEntry((
        IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
        IndexedVariable(:y, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex)))
    ))

    # Test mixed behaviour 
    @test factorization_split(
        (
            FactorizationConstraintEntry((IndexedVariable(:y, nothing),)),
            FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:begin}(firstindex)),))
        ),
        FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:end}(lastindex)),))
    ) == (
        FactorizationConstraintEntry((IndexedVariable(:y, nothing),)),
        FactorizationConstraintEntry((
            IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
        ))
    )

    @test factorization_split(
        FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:begin}(firstindex)),)),
        (
            FactorizationConstraintEntry((IndexedVariable(:x, FunctionalIndex{:end}(lastindex)),)),
            FactorizationConstraintEntry((IndexedVariable(:z, nothing),),)
        )
    ) == (
        FactorizationConstraintEntry((
            IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
        )),
        FactorizationConstraintEntry((IndexedVariable(:z, nothing),))
    )
end

@testitem "FactorizationConstraint" begin
    import GraphPPL: FactorizationConstraint, FactorizationConstraintEntry, IndexedVariable, FunctionalIndex, CombinedRange, SplittedRange

    # Test 1: Test FactorizationConstraint with single variables
    @test FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))),)
    ) isa Any
    @test FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing),)), FactorizationConstraintEntry((IndexedVariable(:y, nothing),)))
    ) isa Any
    @test_throws ErrorException FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)), (FactorizationConstraintEntry((IndexedVariable(:x, nothing),)),)
    )
    @test_throws ErrorException FactorizationConstraint(
        (IndexedVariable(:x, nothing),), (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))),)
    )

    # Test 2: Test FactorizationConstraint with indexed variables
    @test FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, 1), IndexedVariable(:y, 1))),)
    ) isa Any
    @test FactorizationConstraint(
        (IndexedVariable(:x, 1), IndexedVariable(:y, 1)),
        (FactorizationConstraintEntry((IndexedVariable(:x, 1),)), FactorizationConstraintEntry((IndexedVariable(:y, 1),)))
    ) isa FactorizationConstraint
    @test_throws ErrorException FactorizationConstraint(
        (IndexedVariable(:x, 1), IndexedVariable(:y, 1)), (FactorizationConstraintEntry((IndexedVariable(:x, 1),)),)
    )
    @test_throws ErrorException FactorizationConstraint(
        (IndexedVariable(:x, 1),), (FactorizationConstraintEntry((IndexedVariable(:x, 1), IndexedVariable(:y, 1))),)
    )

    # Test 3: Test FactorizationConstraint with SplittedRanges
    @test FactorizationConstraint(
        (IndexedVariable(:x, nothing),),
        (
            FactorizationConstraintEntry((
                IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
            )),
        )
    ) isa FactorizationConstraint
    @test_throws ErrorException FactorizationConstraint(
        (IndexedVariable(:x, nothing),),
        (
            FactorizationConstraintEntry((
                IndexedVariable(:x, SplittedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
                IndexedVariable(:y, nothing)
            )),
        )
    )

    # Test 4: Test FactorizationConstraint with CombinedRanges
    @test FactorizationConstraint(
        (IndexedVariable(:x, nothing),),
        (
            FactorizationConstraintEntry((
                IndexedVariable(:x, CombinedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
            )),
        )
    ) isa FactorizationConstraint
    @test_throws ErrorException FactorizationConstraint(
        (IndexedVariable(:x, nothing)),
        (
            FactorizationConstraintEntry((
                IndexedVariable(:x, CombinedRange(FunctionalIndex{:begin}(firstindex), FunctionalIndex{:end}(lastindex))),
                IndexedVariable(:y, nothing)
            )),
        )
    )

    # Test 5: Test FactorizationConstraint  with duplicate entries
    @test_throws ErrorException constraint = FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing), IndexedVariable(:out, nothing)),
        (
            FactorizationConstraintEntry((IndexedVariable(:x, nothing),)),
            FactorizationConstraintEntry((IndexedVariable(:x, nothing),)),
            FactorizationConstraintEntry((IndexedVariable(:y, nothing),)),
            FactorizationConstraintEntry((IndexedVariable(:out, nothing),))
        )
    )
end

@testitem "multiply(::FactorizationConstraintEntry, ::FactorizationConstraintEntry)" begin
    import GraphPPL: FactorizationConstraintEntry, IndexedVariable

    entry = FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)))
    global x = entry
    for i in 1:3
        global x = x * x
        @test x == Tuple([entry for _ in 1:(2^i)])
    end
end

@testitem "push!(::Constraints, ::Constraint)" begin
    using Distributions
    import GraphPPL:
        Constraints,
        Constraint,
        FactorizationConstraint,
        FactorizationConstraintEntry,
        PosteriorFormConstraint,
        MessageFormConstraint,
        SpecificSubModelConstraints,
        GeneralSubModelConstraints,
        IndexedVariable,
        FactorID

    # Test 1: Test push! with FactorizationConstraint
    constraints = Constraints()
    constraint = FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),),)
    )
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
    constraint = FactorizationConstraint(
        (IndexedVariable(:x, 1), IndexedVariable(:y, 1)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))),)
    )
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
    constraint = FactorizationConstraint(
        (IndexedVariable(:y, nothing), IndexedVariable(:x, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))),)
    )
    @test_throws ErrorException push!(constraints, constraint)

    # Test 2: Test push! with PosteriorFormConstraint
    constraint = PosteriorFormConstraint(IndexedVariable(:x, nothing), Normal)
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
    constraint = PosteriorFormConstraint((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)), Normal)
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
    constraint = PosteriorFormConstraint(IndexedVariable(:x, 1), Normal)
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
    constraint = PosteriorFormConstraint([IndexedVariable(:x, 1), IndexedVariable(:y, 1)], Normal)
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)

    # Test 3: Test push! with MessageFormConstraint
    constraint = MessageFormConstraint(IndexedVariable(:x, nothing), Normal)
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
    constraint = MessageFormConstraint(IndexedVariable(:x, 2), Normal)
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)

    # Test 4: Test push! with SpecificSubModelConstraints
    constraint = SpecificSubModelConstraints(FactorID(sum, 3), Constraints())
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)

    # Test 5: Test push! with GeneralSubModelConstraints
    constraint = GeneralSubModelConstraints(sum, Constraints())
    push!(constraints, constraint)
    @test_throws ErrorException push!(constraints, constraint)
end

@testitem "push!(::SubModelConstraints, c::Constraint)" begin
    include("../../model_zoo.jl")
    using GraphPPL
    import GraphPPL:
        Constraint,
        GeneralSubModelConstraints,
        SpecificSubModelConstraints,
        FactorizationConstraint,
        FactorizationConstraintEntry,
        PosteriorFormConstraint,
        MessageFormConstraint,
        getconstraint,
        Constraints,
        IndexedVariable

    # Test 1: Test push! with FactorizationConstraint
    constraints = GeneralSubModelConstraints(gcv)
    constraint = FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))),)
    )
    push!(constraints, constraint)
    @test getconstraint(constraints) == Constraints([
        FactorizationConstraint(
            (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
            (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),),)
        )
    ],)
    @test_throws MethodError push!(constraints, "string")

    # Test 2: Test push! with PosteriorFormConstraint
    constraints = GeneralSubModelConstraints(gcv)
    constraint = PosteriorFormConstraint(IndexedVariable(:x, nothing), Normal)
    push!(constraints, constraint)
    @test getconstraint(constraints) == Constraints([PosteriorFormConstraint(IndexedVariable(:x, nothing), Normal)],)
    @test_throws MethodError push!(constraints, "string")

    # Test 3: Test push! with MessageFormConstraint
    constraints = GeneralSubModelConstraints(gcv)
    constraint = MessageFormConstraint(IndexedVariable(:x, nothing), Normal)
    push!(constraints, constraint)
    @test getconstraint(constraints) == Constraints([MessageFormConstraint(IndexedVariable(:x, nothing), Normal)],)
    @test_throws MethodError push!(constraints, "string")

    # Test 4: Test push! with SpecificSubModelConstraints
    constraints = SpecificSubModelConstraints(GraphPPL.FactorID(gcv, 3))
    constraint = FactorizationConstraint(
        (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
        (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing))),)
    )
    push!(constraints, constraint)
    @test getconstraint(constraints) == Constraints([
        FactorizationConstraint(
            (IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),
            (FactorizationConstraintEntry((IndexedVariable(:x, nothing), IndexedVariable(:y, nothing)),),)
        )
    ],)
    @test_throws MethodError push!(constraints, "string")

    # Test 5: Test push! with PosteriorFormConstraint
    constraints = GeneralSubModelConstraints(gcv)
    constraint = PosteriorFormConstraint(IndexedVariable(:x, nothing), Normal)
    push!(constraints, constraint)
    @test getconstraint(constraints) == Constraints([PosteriorFormConstraint(IndexedVariable(:x, nothing), Normal)],)
    @test_throws MethodError push!(constraints, "string")

    # Test 6: Test push! with MessageFormConstraint
    constraints = GeneralSubModelConstraints(gcv)
    constraint = MessageFormConstraint(IndexedVariable(:x, nothing), Normal)
    push!(constraints, constraint)
    @test getconstraint(constraints) == Constraints([MessageFormConstraint(IndexedVariable(:x, nothing), Normal)],)
    @test_throws MethodError push!(constraints, "string")
end

@testitem "is_valid_partition(::Set)" begin
    using GraphPPL
    import GraphPPL: is_valid_partition

    # Test 1: Test that is_valid_partition returns true for a valid partition
    @test is_valid_partition(Set([BitSet([1, 2]), BitSet([3, 4])])) == true

    # Test 2: Test that is_valid_partition returns false for an invalid partition
    @test is_valid_partition(Set([BitSet([1, 2]), BitSet([2, 3])])) == false

    # Test 3: Test that is_valid_partition returns false for an invalid partition
    @test is_valid_partition(Set([BitSet([1, 2]), BitSet([2, 3]), BitSet([3, 4])])) == false

    # Test 4: Test that is_valid_partition returns false for an invalid partition
    @test is_valid_partition(Set([BitSet([1, 2]), BitSet([4, 5])])) == false
end

@testitem "constant_constraint" begin
    using BitSetTuples
    import GraphPPL: constant_constraint

    @test constant_constraint(1, 1) == BitSetTuple(1)
    @test constant_constraint(5, 3) == BitSetTuple([[1, 2, 4, 5], [1, 2, 4, 5], [3], [1, 2, 4, 5], [1, 2, 4, 5]])
end

@testitem "Application of PosteriorFormConstraint" begin
    import GraphPPL: PosteriorFormConstraint, IndexedVariable, apply_constraints!, getextra, hasextra

    include("../../model_zoo.jl")

    struct ArbitraryFunctionalFormConstraint end

    # Test saving of PosteriorFormConstraint in single variable
    model = create_terminated_model(simple_model)
    context = GraphPPL.getcontext(model)
    constraint = PosteriorFormConstraint(IndexedVariable(:x, nothing), ArbitraryFunctionalFormConstraint())
    apply_constraints!(model, context, constraint)
    for node in filter(GraphPPL.as_variable(:x), model)
        @test getextra(model[node], :posterior_form_constraint) == ArbitraryFunctionalFormConstraint()
    end

    # Test saving of PosteriorFormConstraint in multiple variables
    model = create_terminated_model(vector_model)
    context = GraphPPL.getcontext(model)
    constraint = PosteriorFormConstraint(IndexedVariable(:x, nothing), ArbitraryFunctionalFormConstraint())
    apply_constraints!(model, context, constraint)
    for node in filter(GraphPPL.as_variable(:x), model)
        @test getextra(model[node], :posterior_form_constraint) == ArbitraryFunctionalFormConstraint()
    end
    for node in filter(GraphPPL.as_variable(:y), model)
        @test !hasextra(model[node], :posterior_form_constraint)
    end

    # Test saving of PosteriorFormConstraint in single variable in array
    model = create_terminated_model(vector_model)
    context = GraphPPL.getcontext(model)
    constraint = PosteriorFormConstraint(IndexedVariable(:x, 1), ArbitraryFunctionalFormConstraint())
    apply_constraints!(model, context, constraint)
    applied_node = context[:x][1]
    for node in filter(GraphPPL.as_variable(:x), model)
        if node == applied_node
            @test getextra(model[node], :posterior_form_constraint) == ArbitraryFunctionalFormConstraint()
        else
            @test !hasextra(model[node], :posterior_form_constraint)
        end
    end
end

@testitem "Application of MessageFormConstraint" begin
    import GraphPPL: MessageFormConstraint, IndexedVariable, apply_constraints!, hasextra, getextra

    include("../../model_zoo.jl")

    struct ArbitraryMessageFormConstraint end

    # Test saving of MessageFormConstraint in single variable
    model = create_terminated_model(simple_model)
    context = GraphPPL.getcontext(model)
    constraint = MessageFormConstraint(IndexedVariable(:x, nothing), ArbitraryMessageFormConstraint())
    node = first(filter(GraphPPL.as_variable(:x), model))
    apply_constraints!(model, context, constraint)
    @test getextra(model[node], :message_form_constraint) == ArbitraryMessageFormConstraint()

    # Test saving of MessageFormConstraint in multiple variables
    model = create_terminated_model(vector_model)
    context = GraphPPL.getcontext(model)
    constraint = MessageFormConstraint(IndexedVariable(:x, nothing), ArbitraryMessageFormConstraint())
    apply_constraints!(model, context, constraint)
    for node in filter(GraphPPL.as_variable(:x), model)
        @test getextra(model[node], :message_form_constraint) == ArbitraryMessageFormConstraint()
    end
    for node in filter(GraphPPL.as_variable(:y), model)
        @test !hasextra(model[node], :message_form_constraint)
    end

    # Test saving of MessageFormConstraint in single variable in array
    model = create_terminated_model(vector_model)
    context = GraphPPL.getcontext(model)
    constraint = MessageFormConstraint(IndexedVariable(:x, 1), ArbitraryMessageFormConstraint())
    apply_constraints!(model, context, constraint)
    applied_node = context[:x][1]
    for node in filter(GraphPPL.as_variable(:x), model)
        if node == applied_node
            @test getextra(model[node], :message_form_constraint) == ArbitraryMessageFormConstraint()
        else
            @test !hasextra(model[node], :message_form_constraint)
        end
    end
end

@testitem "save constraints with constants" begin
    include("../../model_zoo.jl")
    using BitSetTuples
    using GraphPPL
    import GraphPPL:
        intersect_constraint_bitset!,
        constant_constraint,
        factorization_constraint,
        getproperties,
        VariationalConstraintsPlugin,
        PluginsCollection

    model = create_terminated_model(simple_model; plugins = GraphPPL.PluginsCollection(VariationalConstraintsPlugin()))
    ctx = GraphPPL.getcontext(model)

    @test constant_constraint(3, 1) == BitSetTuple([[1], [2, 3], [2, 3]])
    @test constant_constraint(3, 2) == BitSetTuple([[1, 3], [2], [1, 3]])
    @test constant_constraint(3, 3) == BitSetTuple([[1, 2], [1, 2], [3]])

    node = ctx[NormalMeanVariance, 2]
    @test intersect_constraint_bitset!(model[node], constant_constraint(3, 1)) == BitSetTuple([[1], [2, 3], [2, 3]])
    @test intersect_constraint_bitset!(model[node], constant_constraint(3, 2)) == BitSetTuple([[1], [2], [3]])

    node = ctx[NormalMeanVariance, 1]
    # Here it is the mean field because the original model has `x ~ Normal(0, 1)` and `0` and `1` are constants 
    @test intersect_constraint_bitset!(model[node], constant_constraint(3, 1)) == BitSetTuple([[1], [2], [3]])
end

@testitem "materialize_constraints!(:Model, ::NodeLabel, ::FactorNodeData)" begin
    include("../../model_zoo.jl")

    using BitSetTuples
    using GraphPPL

    import GraphPPL: materialize_constraints!, EdgeLabel, get_constraint_names, getproperties, getextra, setextra!

    # Test 1: Test materialize with a Full Factorization constraint
    model = create_terminated_model(simple_model)
    ctx = GraphPPL.getcontext(model)
    node = ctx[NormalMeanVariance, 2]

    # Force overwrite the bitset and the constraints
    setextra!(model[node], :factorization_constraint_bitset, BitSetTuple([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    materialize_constraints!(model, node)
    @test get_constraint_names(getextra(model[node], :factorization_constraint)) == ((:out, :μ, :σ),)

    node = ctx[NormalMeanVariance, 1]
    setextra!(model[node], :factorization_constraint_bitset, BitSetTuple([[1], [2], [3]]))
    materialize_constraints!(model, node)
    @test get_constraint_names(getextra(model[node], :factorization_constraint)) == ((:out,), (:μ,), (:σ,))

    # Test 2: Test materialize with an applied constraint
    model = create_terminated_model(simple_model)
    ctx = GraphPPL.getcontext(model)
    node = ctx[NormalMeanVariance, 2]

    setextra!(model[node], :factorization_constraint_bitset, BitSetTuple([[1], [2, 3], [2, 3]]))
    materialize_constraints!(model, node)
    @test get_constraint_names(getextra(model[node], :factorization_constraint)) == ((:out,), (:μ, :σ))

    # # Test 3: Check that materialize_constraints! throws if the constraint is not a valid partition
    model = create_terminated_model(simple_model)
    ctx = GraphPPL.getcontext(model)
    node = ctx[NormalMeanVariance, 2]

    setextra!(model[node], :factorization_constraint_bitset, BitSetTuple([[1], [3], [2, 3]]))
    @test_throws ErrorException materialize_constraints!(model, node)

    # Test 4: Check that materialize_constraints! throws if the constraint is not a valid partition
    model = create_terminated_model(simple_model)
    ctx = GraphPPL.getcontext(model)
    node = ctx[NormalMeanVariance, 2]

    setextra!(model[node], :factorization_constraint_bitset, BitSetTuple([[1], [1], [3]]))
    @test_throws ErrorException materialize_constraints!(model, node)
end

@testitem "Resolve Factorization Constraints" begin
    include("../../model_zoo.jl")
    using GraphPPL
    import GraphPPL:
        FactorizationConstraint,
        FactorizationConstraintEntry,
        IndexedVariable,
        resolve,
        ResolvedFactorizationConstraint,
        ResolvedConstraintLHS,
        ResolvedFactorizationConstraintEntry,
        ResolvedIndexedVariable,
        CombinedRange,
        SplittedRange

    model = create_terminated_model(outer)
    ctx = GraphPPL.getcontext(model)
    inner_context = ctx[inner, 1]

    # Test resolve constraint in child model

    let constraint = FactorizationConstraint(
            (IndexedVariable(:α, nothing), IndexedVariable(:θ, nothing)),
            (FactorizationConstraintEntry((IndexedVariable(:α, nothing),)), FactorizationConstraintEntry((IndexedVariable(:θ, nothing),)))
        )
        result = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:y, nothing, ctx), ResolvedIndexedVariable(:w, CombinedRange(2, 3), ctx)),),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, nothing, ctx),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, CombinedRange(2, 3), ctx),))
            )
        )
        @test resolve(model, inner_context, constraint) == result
    end

    # Test constraint in top level model

    let constraint = FactorizationConstraint(
            (IndexedVariable(:y, nothing), IndexedVariable(:w, nothing)),
            (FactorizationConstraintEntry((IndexedVariable(:y, nothing),)), FactorizationConstraintEntry((IndexedVariable(:w, nothing),)))
        )
        result = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:y, nothing, ctx), ResolvedIndexedVariable(:w, CombinedRange(1, 5), ctx)),),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, nothing, ctx),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, CombinedRange(1, 5), ctx),))
            )
        )
        @test resolve(model, ctx, constraint) == result
    end

    # Test a constraint that is not applicable at all

    let constraint = FactorizationConstraint(
            (IndexedVariable(:i, nothing), IndexedVariable(:dont, nothing), IndexedVariable(:apply, nothing)),
            (
                FactorizationConstraintEntry((IndexedVariable(:i, nothing),)),
                FactorizationConstraintEntry((IndexedVariable(:dont, nothing),)),
                FactorizationConstraintEntry((IndexedVariable(:apply, nothing),))
            )
        )
        result = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((),),
            (ResolvedFactorizationConstraintEntry(()), ResolvedFactorizationConstraintEntry(()), ResolvedFactorizationConstraintEntry(()))
        )
        @test resolve(model, ctx, constraint) == result
    end
end

@testitem "Resolved Constraints in" begin
    using GraphPPL
    import GraphPPL:
        ResolvedFactorizationConstraint,
        ResolvedConstraintLHS,
        ResolvedFactorizationConstraintEntry,
        ResolvedIndexedVariable,
        SplittedRange,
        getname,
        index,
        VariableNodeProperties

    context = GraphPPL.Context()
    variable = ResolvedIndexedVariable(:w, 2:3, context)
    node_data = GraphPPL.NodeData(context, VariableNodeProperties(name = :w, index = 2))
    @test node_data ∈ variable

    variable = ResolvedIndexedVariable(:w, 2:3, context)
    node_data = GraphPPL.NodeData(GraphPPL.Context(), VariableNodeProperties(name = :w, index = 2))
    @test !(node_data ∈ variable)

    variable = ResolvedIndexedVariable(:w, 2, context)
    node_data = GraphPPL.NodeData(context, VariableNodeProperties(name = :w, index = 2))
    @test node_data ∈ variable

    variable = ResolvedIndexedVariable(:w, SplittedRange(2, 3), context)
    node_data = GraphPPL.NodeData(context, VariableNodeProperties(name = :w, index = 2))
    @test node_data ∈ variable

    variable = ResolvedIndexedVariable(:w, SplittedRange(10, 15), context)
    node_data = GraphPPL.NodeData(context, VariableNodeProperties(name = :w, index = 2))
    @test !(node_data ∈ variable)

    variable = ResolvedIndexedVariable(:x, nothing, context)
    node_data = GraphPPL.NodeData(context, VariableNodeProperties(name = :x, index = 2))
    @test node_data ∈ variable

    variable = ResolvedIndexedVariable(:x, nothing, context)
    node_data = GraphPPL.NodeData(context, VariableNodeProperties(name = :x, index = nothing))
    @test node_data ∈ variable
end

@testitem "convert_to_bitsets" begin
    import GraphPPL:
        ResolvedFactorizationConstraint,
        ResolvedConstraintLHS,
        ResolvedFactorizationConstraintEntry,
        ResolvedIndexedVariable,
        SplittedRange,
        CombinedRange,
        apply_constraints!,
        getproperties

    using BitSetTuples

    include("../../model_zoo.jl")

    model = create_terminated_model(outer; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin()))
    context = GraphPPL.getcontext(model)
    inner_context = context[inner, 1]
    inner_inner_context = inner_context[inner_inner, 1]

    normal_node = inner_inner_context[NormalMeanVariance, 1]
    neighbors = model[GraphPPL.neighbors(model, normal_node)]

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:w, 2:3, context),)),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, 2, context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, 3, context),))
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1, 2, 3], [1, 2], [1, 3]])
    end

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:w, 4:5, context),)),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, 4, context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, 5, context),))
            )
        )
        @test !GraphPPL.is_applicable(neighbors, constraint)
    end

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:w, 2:3, context),)),
            (ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, SplittedRange(2, 3), context),)),)
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1, 2, 3], [1, 2], [1, 3]])
    end

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:w, 2:3, context), ResolvedIndexedVariable(:y, nothing, context))),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, SplittedRange(2, 3), context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, nothing, context),))
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1], [2], [3]])
    end

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:w, 2:3, context), ResolvedIndexedVariable(:y, nothing, context))),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, 2, context),)),
                ResolvedFactorizationConstraintEntry((
                    ResolvedIndexedVariable(:w, 3, context), ResolvedIndexedVariable(:y, nothing, context)
                ))
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1, 3], [2], [1, 3]])
    end

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:w, 2:3, context), ResolvedIndexedVariable(:y, nothing, context))),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:w, CombinedRange(2, 3), context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, nothing, context),))
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1], [2, 3], [2, 3]])
    end

    model = create_terminated_model(multidim_array; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin()))
    context = GraphPPL.getcontext(model)
    normal_node = context[NormalMeanVariance, 5]
    neighbors = model[GraphPPL.neighbors(model, normal_node)]

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:x, nothing, context),),),
            (
                ResolvedFactorizationConstraintEntry((
                    ResolvedIndexedVariable(:x, SplittedRange(CartesianIndex(1, 1), CartesianIndex(3, 3)), context),
                )),
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1, 3], [2, 3], [1, 2, 3]])
    end

    model = create_terminated_model(multidim_array; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin()))
    context = GraphPPL.getcontext(model)
    normal_node = context[NormalMeanVariance, 5]
    neighbors = model[GraphPPL.neighbors(model, normal_node)]

    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:x, nothing, context),),),
            (
                ResolvedFactorizationConstraintEntry((
                    ResolvedIndexedVariable(:x, CombinedRange(CartesianIndex(1, 1), CartesianIndex(3, 3)), context),
                )),
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    end

    # Test ResolvedFactorizationConstraints over anonymous variables

    model = create_terminated_model(node_with_only_anonymous; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin()))
    context = GraphPPL.getcontext(model)
    normal_node = context[NormalMeanVariance, 6]
    neighbors = model[GraphPPL.neighbors(model, normal_node)]
    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:y, nothing, context),),),
            (ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, SplittedRange(1, 10), context),)),)
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
    end

    # Test ResolvedFactorizationConstraints over multiple anonymous variables
    model = create_terminated_model(node_with_two_anonymous; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin()))
    context = GraphPPL.getcontext(model)
    normal_node = context[NormalMeanVariance, 6]
    neighbors = model[GraphPPL.neighbors(model, normal_node)]
    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:y, nothing, context),),),
            (ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, SplittedRange(1, 10), context),)),)
        )
        @test GraphPPL.is_applicable(neighbors, constraint)

        # This shouldn't throw and resolve because both anonymous variables are 1-to-1 and referenced by constraint.
        @test GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint) == BitSetTuple([[1, 2, 3], [1, 2], [1, 3]])
    end

    # Test ResolvedFactorizationConstraints over ambiguous anonymouys variables
    model = create_terminated_model(
        node_with_ambiguous_anonymous; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin())
    )
    context = GraphPPL.getcontext(model)
    normal_node = last(filter(GraphPPL.as_node(NormalMeanVariance), model))
    neighbors = model[GraphPPL.neighbors(model, normal_node)]
    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((ResolvedIndexedVariable(:y, nothing, context),),),
            (ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:y, SplittedRange(1, 10), context),)),)
        )
        @test GraphPPL.is_applicable(neighbors, constraint)

        # This test should throw since we cannot resolve the constraint
        @test_throws ErrorException GraphPPL.convert_to_bitsets(model, normal_node, neighbors, constraint)
    end

    # Test ResolvedFactorizationConstraint with a Mixture node
    model = create_terminated_model(mixture; plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin()))
    context = GraphPPL.getcontext(model)
    mixture_node = first(filter(GraphPPL.as_node(Mixture), model))
    neighbors = model[GraphPPL.neighbors(model, mixture_node)]
    let constraint = ResolvedFactorizationConstraint(
            ResolvedConstraintLHS((
                ResolvedIndexedVariable(:m1, nothing, context),
                ResolvedIndexedVariable(:m2, nothing, context),
                ResolvedIndexedVariable(:m3, nothing, context),
                ResolvedIndexedVariable(:m4, nothing, context)
            ),),
            (
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:m1, nothing, context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:m2, nothing, context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:m3, nothing, context),)),
                ResolvedFactorizationConstraintEntry((ResolvedIndexedVariable(:m4, nothing, context),))
            )
        )
        @test GraphPPL.is_applicable(neighbors, constraint)
        @test GraphPPL.convert_to_bitsets(model, mixture_node, neighbors, constraint) == BitSetTuple([
            collect(1:9),
            [1, 2, 6, 7, 8, 9],
            [1, 3, 6, 7, 8, 9],
            [1, 4, 6, 7, 8, 9],
            [1, 5, 6, 7, 8, 9],
            collect(1:9),
            collect(1:9),
            collect(1:9),
            collect(1:9)
        ])
    end
end

@testitem "lazy_bool_allequal" begin
    import GraphPPL: lazy_bool_allequal

    @testset begin
        itr = [1, 2, 3, 4]

        outcome, value = lazy_bool_allequal(x -> x > 0, itr)
        @test outcome === true
        @test value === true

        outcome, value = lazy_bool_allequal(x -> x < 0, itr)
        @test outcome === true
        @test value === false
    end

    @testset begin
        itr = [1, 2, -1, -2]

        outcome, value = lazy_bool_allequal(x -> x > 0, itr)
        @test outcome === false
        @test value === true

        outcome, value = lazy_bool_allequal(x -> x < 0, itr)
        @test outcome === false
        @test value === false
    end

    @testset begin
        # We do not support it for now, but we can add it in the future
        @test_throws ErrorException lazy_bool_allequal(x -> x > 0, [])
    end
end

@testitem "default_constraints" begin
    import GraphPPL:
        default_constraints, factorization_constraint, getproperties, PluginsCollection, VariationalConstraintsPlugin, hasextra, getextra

    include("../../model_zoo.jl")

    @test default_constraints(simple_model) == GraphPPL.Constraints()
    @test default_constraints(model_with_default_constraints) == @constraints(
        begin
            q(a, d) = q(a)q(d)
        end
    )

    model = create_terminated_model(contains_default_constraints; plugins = PluginsCollection(VariationalConstraintsPlugin()))
    ctx = GraphPPL.getcontext(model)
    # Test that default constraints are applied
    for i in 1:10
        node = model[ctx[model_with_default_constraints, i][NormalMeanVariance, 1]]
        @test hasextra(node, :factorization_constraint)
        @test GraphPPL.getname(getextra(node, :factorization_constraint)) == ((:out,), (:μ,), (:σ,))
    end

    # Test that default constraints are not applied if we specify constraints in the context
    c = @constraints begin
        for q in model_with_default_constraints
            q(a, d) = q(a, d)
        end
    end
    model = create_terminated_model(contains_default_constraints; plugins = PluginsCollection(VariationalConstraintsPlugin(c)))
    ctx = GraphPPL.getcontext(model)
    for i in 1:10
        node = model[ctx[model_with_default_constraints, i][NormalMeanVariance, 1]]
        @test hasextra(node, :factorization_constraint)
        @test GraphPPL.getname(getextra(node, :factorization_constraint)) == ((:out, :μ), (:σ,))
    end

    # Test that default constraints are not applied if we specify constraints for a specific instance of the submodel
    c = @constraints begin
        for q in (model_with_default_constraints, 1)
            q(a, d) = q(a, d)
        end
    end
    model = create_terminated_model(contains_default_constraints; plugins = PluginsCollection(VariationalConstraintsPlugin(c)))
    ctx = GraphPPL.getcontext(model)
    for i in 1:10
        node = model[ctx[model_with_default_constraints, i][NormalMeanVariance, 1]]
        @test hasextra(node, :factorization_constraint)
        if i == 1
            @test GraphPPL.getname(getextra(node, :factorization_constraint)) == ((:out, :μ), (:σ,))
        else
            @test GraphPPL.getname(getextra(node, :factorization_constraint)) == ((:out,), (:μ,), (:σ,))
        end
    end
end

@testitem "mean_field_constraint" begin
    using BitSetTuples
    import GraphPPL: mean_field_constraint

    @test mean_field_constraint(5) == BitSetTuple([[1], [2], [3], [4], [5]])
    @test mean_field_constraint(10) == BitSetTuple([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

    @test mean_field_constraint(1, (1,)) == BitSetTuple(1)
    @test mean_field_constraint(2, (1,)) == BitSetTuple([[1], [2]])
    @test mean_field_constraint(2, (2,)) == BitSetTuple([[1], [2]])
    @test mean_field_constraint(5, (1, 3, 5)) == BitSetTuple([[1], [2, 4], [3], [2, 4], [5]])
    @test mean_field_constraint(5, (1, 2, 3, 4, 5)) == BitSetTuple([[1], [2], [3], [4], [5]])
    @test_throws BoundsError mean_field_constraint(5, (1, 2, 3, 4, 5, 6)) == BitSetTuple([[1], [2], [3], [4], [5]])
    @test mean_field_constraint(5, (1, 2)) == BitSetTuple([[1], [2], [3, 4, 5], [3, 4, 5], [3, 4, 5]])
end