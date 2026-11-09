using Devito, PyCall, Test

# Block-owned decoupler memory (`DEVITO_DECOUPLER_MEMORY=blocks`). The data
# lives in the workers' blocks; `blockdata` reaches it without materializing
# the global array in this process. Requires a running decoupler.

@testset "DevitoBlockArray basics" begin
    g = Grid(shape=(8, 8), extent=(7.0, 7.0))
    u = TimeFunction(name="ublk", grid=g, space_order=2)

    data(u) .= 1.0
    op = Operator([Eq(forward(u), u + 1)])
    apply(op, time_M=2, dt=1.0)

    b = blockdata(u)

    @test size(b) == size(data(u))
    @test eltype(b) == Float32

    # Routed element reads agree with what the master's array holds
    reference = convert(Array, data(u))
    @test b[1, 1, 1] ≈ reference[1, 1, 1]
    @test b[4, 5, 2] ≈ reference[4, 5, 2]

    # A full gather agrees too -- it warns on the Python side, by design
    @test convert(Array, b) ≈ reference

    # Reductions never gather
    @test minimum(b) ≈ minimum(reference)
    @test maximum(b) ≈ maximum(reference)

    # The coordinator owns nothing, so its block is empty but the handle on
    # the Python owner is retained -- that is what keeps the blocks alive
    @test length(parent(b)) == 0
    @test b.o isa PyObject
end

@testset "DevitoBlockArray write-through" begin
    g = Grid(shape=(8, 8), extent=(7.0, 7.0))
    u = TimeFunction(name="wblk", grid=g, space_order=2)

    data(u) .= 0.0
    op = Operator([Eq(forward(u), u + 1)])
    apply(op, time_M=1, dt=1.0)

    b = blockdata(u)
    fill!(b, 7.0f0)

    @test convert(Array, b) ≈ 7 .* ones(Float32, size(b))
end
