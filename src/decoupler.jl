#
# Block-owned decoupler memory
#
# Under `DEVITO_DECOUPLER_MEMORY=blocks` a devitopro decoupler worker owns its
# block outright and the master -- this process -- owns no payload. A Function's
# values are then reached through a Python `GlobalView`, which is collective
# over the workers: the master drives them into each call and takes part owning
# nothing.
#
# `DevitoBlockArray` presents that as an `AbstractArray`. It mirrors
# `DevitoMPIArray` in `ext/MPIExt.jl` field for field -- the two describe the
# same thing, a global array whose pieces live elsewhere -- but needs no MPI.jl
# at all: every exchange happens inside Python, over the decoupler's own data
# plane. That is also why this lives in `src/` rather than `ext/`: a Julia
# package extension has to be triggered by a weak dependency, and there is no
# new dependency to trigger on.
#
# Lifetime
# --------
# `p` is wrapped with `own=false` when the block is reachable through a
# shared-memory window, which means Julia holds no reference to the Python
# owner and will not keep it alive. The `o::PyObject` field exists precisely
# for that: it retains the `GlobalView`, which in turn retains the window.
# Dropping `o` while `p` is still in use would leave a dangling pointer, so
# never reconstruct a `DevitoBlockArray` around a bare `p`.

const decoupler_views = PyNULL()

"""
    DevitoBlockArray{T,N,A,D}

A global array whose data lives in the decoupler workers' blocks.

Indexing, conversion and assignment all go through the Python `GlobalView` in
`o`, which routes each element to the worker that owns it. `p` is this
process's own block -- empty on the master, which is the coordinator and owns
nothing.

See also [`blockdata`](@ref).
"""
struct DevitoBlockArray{T,N,A<:AbstractArray{T,N},D} <: AbstractArray{T,N}
    o::PyObject                   # the GlobalView -- keeps the blocks alive
    p::A                          # this process's block; empty on the master
    local_indices::NTuple{N,UnitRange{Int}}
    decomposition::D
    topology::NTuple{N,Int}
    size::NTuple{N,Int}           # GLOBAL
end

Base.size(x::DevitoBlockArray) = x.size
Base.parent(x::DevitoBlockArray) = x.p
Base.IndexStyle(::Type{<:DevitoBlockArray}) = IndexCartesian()

localindices(x::DevitoBlockArray) = x.local_indices
decomposition(x::DevitoBlockArray) = x.decomposition
topology(x::DevitoBlockArray) = x.topology

"""
    blockdata(x::DiscreteFunction)

The block-owned data of `x`, as a [`DevitoBlockArray`](@ref).

Requires a running decoupler under `DEVITO_DECOUPLER_MEMORY=blocks`; `x` must
have been passed to an `apply`, which is what splits it into blocks.

Unlike `data(x)`, this never materializes the global array in this process:
reads and writes are routed to the owning worker.
"""
function blockdata(x::DiscreteFunction{T,N,M}) where {T,N,M}
    if decoupler_views == PyNULL()
        copy!(decoupler_views, pyimport("devitopro.decoupler.views"))
    end
    DevitoBlockArray{T,N}(decoupler_views.block_view(x.o.name))
end

function DevitoBlockArray{T,N}(o) where {T,N}
    layout = o.layout
    n = NTuple{N,Int}(reverse(Int.(layout.global_shape)))
    topo = NTuple{N,Int}(reverse(Int.(layout.topology)))

    # The master owns nothing, so its block is empty and its local indices are
    # all empty ranges. A `ptr` is handed out only for a block reachable
    # through a shared-memory window; without one the values move over the
    # data plane and there is nothing local to wrap.
    idxs = ntuple(_->1:0, N)
    p = Array{T,N}(undef, ntuple(_->0, N))
    decomp = block_decomposition(layout, N)

    DevitoBlockArray{T,N,Array{T,N},typeof(decomp)}(o, p, idxs, decomp, topo, n)
end

"""
    block_decomposition(layout, N)

The per-axis one-based global index ranges owned by each block, slowest axis
last to match Julia's ordering.
"""
function block_decomposition(layout, N)
    ntuple(N) do i
        axis = N - i + 1                                 # Python axis order
        chunks = layout.chunks[axis]
        [Int.(c) .+ 1 for c in chunks]
    end
end

function Base.getindex(x::DevitoBlockArray{T,N}, I::Vararg{Int,N}) where {T,N}
    # Python indexes slowest axis first and counts from zero
    idx = reverse(ntuple(i->I[i] - 1, N))
    convert(T, x.o.__getitem__(idx))
end

function Base.convert(::Type{Array}, x::DevitoBlockArray{T,N}) where {T,N}
    # `__array__` warns on the Python side: it funnels every block through one
    # rank, which is exactly what block ownership is there to avoid. It stays
    # available because printing and plotting genuinely need it.
    permutedims(convert(Array{T,N}, PyArray(x.o.__array__())), N:-1:1)
end

function Base.copyto!(dst::DevitoBlockArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    dst.o.__setitem__(pybuiltin("Ellipsis"), permutedims(Array(src), N:-1:1))
    dst
end

function Base.copy!(dst::DevitoBlockArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    axes(dst) == axes(src) || throw(ArgumentError(
        "arrays must have the same axes for copy! (consider using `copyto!`)"))
    copyto!(dst, src)
end

Base.fill!(x::DevitoBlockArray, v) = (x.o.__setitem__(pybuiltin("Ellipsis"), v); x)

Base.minimum(x::DevitoBlockArray{T}) where {T} = convert(T, x.o.min())
Base.maximum(x::DevitoBlockArray{T}) where {T} = convert(T, x.o.max())
Base.sum(x::DevitoBlockArray{T}) where {T} = convert(T, x.o.sum())

export DevitoBlockArray, blockdata
