#=
1. setting earth model properties
   i. get size information (including) halo from grid.
   ii. get the size that includes the halo
   iii. get localindices ??
2. setting wavelets
3. integration showing MPI and serial make the same result
4. unit tests
5. type stability
6. use latest release of Devito or head of master
=#

module Devito

using MPI, PyCall

const numpy = PyNULL()
const devito = PyNULL()
const seismic = PyNULL()

function __init__()
    copy!(numpy, pyimport("numpy"))
    copy!(devito, pyimport("devito"))
    copy!(seismic, pyimport("examples.seismic"))
end

numpy_eltype(dtype) = dtype == numpy.float32 ? Float32 : Float64

PyCall.PyObject(::Type{Float32}) = numpy.float32
PyCall.PyObject(::Type{Float64}) = numpy.float64

struct DevitoArray{T,N} <: AbstractArray{T,N}
    o::PyObject
    p::Array{T,N}
end

function DevitoArray{T,N}(o) where {T,N}
    p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), reverse(o.shape); own=false)
    DevitoArray{T,N}(o, p)
end

function DevitoArray(o)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    DevitoArray{T,N}(o)
end

Base.size(x::DevitoArray{T,N}) where {T,N} = reverse(x.o.shape)::NTuple{N,Int}
Base.parent(x::DevitoArray) = x.p

Base.getindex(x::DevitoArray{T,N}, i) where {T,N} = getindex(parent(x), i)
Base.setindex!(x::DevitoArray{T,N}, v, i) where {T,N} = setindex!(parent(x), v, i)
Base.IndexStyle(::Type{<:DevitoArray}) = IndexLinear()

# Devito configuration methods
function configuration!(key, value)
    c = PyDict(devito."configuration")
    c[key] = value
    c[key]
end
configuration(key) = PyDict(devito."configuration")[key]
configuration() = PyDict(devito."configuration")

# Python <-> Julia quick-and-dirty type/struct mappings
for (M,F) in (
        (:devito,:Constant), (:devito,:Eq), (:devito,:Injection), (:devito,:Operator), (:devito,:SpaceDimension), (:devito,:SteppingDimension),
        (:seismic, :Receiver), (:seismic,:RickerSource), (:seismic,:TimeAxis))
    @eval begin
        struct $F
            o::PyObject
        end
        PyCall.PyObject(x::$F) = x.o
        Base.convert(::Type{$F}, x::PyObject) = $F(x)
        $F(args...; kwargs...) = pycall($M.$F, $F, args...; kwargs...)
        export $F
    end
end

#
# Grid
#
struct Grid{T,N}
    o::PyObject
end

function Grid(args...; kwargs...)
    o = pycall(devito.Grid, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    Grid{T,N}(o)
end

PyCall.PyObject(x::Grid) = x.o

Base.size(grid::Grid{T,N}) where {T,N} = reverse((grid.o.shape)::NTuple{N,Int})
Base.ndims(grid::Grid{T,N}) where {T,N} = N
Base.eltype(grid::Grid{T}) where {T} = T

spacing(x::Union{SpaceDimension,SteppingDimension}) = x.o.spacing
spacing(x::Grid{T,N}) where {T,N} = reverse(x.o.spacing)
spacing_map(x::Grid) = PyDict(x.o."spacing_map")

#
# Functions
#
abstract type DiscreteFunction{T,N} end

struct Function{T,N} <: DiscreteFunction{T,N}
    o::PyObject
end

function Function(args...; kwargs...)
    o = pycall(devito.Function, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    Function{T,N}(o)
end

struct TimeFunction{T,N} <: DiscreteFunction{T,N}
    o::PyObject
end

function TimeFunction(args...; kwargs...)
    o = pycall(devito.TimeFunction, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    TimeFunction{T,N}(o)
end

PyCall.PyObject(x::DiscreteFunction) = x.o

grid(x::Function{T,N}) where {T,N} = Grid{T,N}(x.o.grid)
grid(x::TimeFunction{T,N}) where {T,N} = Grid{T,N-1}(x.o.grid)
halo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o.halo)::NTuple{N,Tuple{Int,Int}}

forward(x::TimeFunction) = x.o.forward
backward(x::TimeFunction) = x.o.backward

data_with_halo(x::DiscreteFunction{T,N}) where {T,N} = DevitoArray{T,N}(x.o."data_with_halo")

function data(x::DiscreteFunction{T,N}) where {T,N}
    h = halo(x)
    y = data_with_halo(x)
    rng = ntuple(i->(h[i][1]+1):(size(y,i)-h[i][2]), N)
    @view y[rng...]
end

# TODO - make me type stable
function data(x::Receiver)
    T = numpy_eltype(x.o."data".dtype)
    N = length(x.o."data".shape)
    DevitoArray{T,N}(x.o."data")'
end

function Dimension(o)
    if o.is_Space
        return SpaceDimension(o)
    elseif o.is_Stepping
        return SteppingDimension(o)
    else
        error("not implemented")
    end
end

function dimensions(x::DiscreteFunction{T,N}) where {T,N}
    ntuple(i->Dimension(x.o.dimensions[N-i+1]), N)
end

inject(x::RickerSource, args...; kwargs...) = pycall(PyObject(x).inject, Injection, args...; kwargs...)

interpolate(x::Receiver; kwargs...) = pycall(PyObject(x).interpolate, PyObject; kwargs...)

Base.step(x::TimeAxis) = PyObject(x).step

apply(x::Operator, args...; kwargs...) = pycall(PyObject(x).apply, PyObject, args...; kwargs...)

dx(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dx, PyObject, args...; kwargs...)
dy(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dy, PyObject, args...; kwargs...)
dz(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dz, PyObject, args...; kwargs...)

Base.:*(x::DiscreteFunction, y::PyObject) = x.o*y
Base.:*(x::PyObject, y::DiscreteFunction) = x*y.o
Base.:/(x::DiscreteFunction, y::PyObject) = x.o/y
Base.:/(x::PyObject, y::DiscreteFunction) = x/y.o
Base.:^(x::Function, y) = x.o^y

export Grid, Function, SpaceDimension, SteppingDimension, TimeFunction, apply, backward, configuration, configuration!, data, data_with_halo, dx, dy, dz, interpolate, dimensions, forward, grid, inject, spacing, spacing_map, step

# lindices(x::TimeFunction) = PyObject(x).local_indices

# data_nompi(timefunction::TimeFunction) = PyObject(timefunction).data

# function data_mpi_2D(timefunction::TimeFunction)
#     MPI.Initialized() || MPI.Init()

#     grd = grid(timefunction)
#     indices = lindices(timefunction)
#     x = PyObject(timefunction).data
#     y = zeros(eltype(grd), size(x,1), size(grd)...)

#     indices_1 = indices[1].start+1:indices[1].stop
#     indices_2 = indices[2].start+1:indices[2].stop
#     indices_3 = indices[3].start+1:indices[3].stop

#     y[indices_1,indices_2,indices_3] .= x
#     MPI.Reduce(y, +, 0, MPI.COMM_WORLD)
# end

# function data_mpi_3D(timefunction::TimeFunction)
#     MPI.Initialized() || MPI.Init()

#     grd = grid(timefunction)
#     indices = lindices(timefunction)
#     x = data(timefunction)
#     y = zeros(eltype(grd), size(x,1), size(grd)...)

#     indices_1 = indices[1].start+1:indices[1].stop
#     indices_2 = indices[2].start+1:indices[2].stop
#     indices_3 = indices[3].start+1:indices[3].stop
#     indices_4 = indices[4].start+1:indices[4].stop

#     y[indices_1,indices_2,indices_3,indices_4] .= x
#     MPI.Reduce(y, +, 0, MPI.COMM_WORLD)
# end

# # TODO: use parametric types for TimeFunction to make this type stable
# function data(timefunction::TimeFunction)
#     local d
#     if configuration("mpi") == false
#         d = data_nompi(timefunction)
#     else
#         grd = grid(timefunction)
#         if ndims(grd) == 2
#             d = data_mpi_2D(timefunction)
#         elseif ndims(grd) == 3
#             d = data_mpi_3D(timefunction)
#         else
#             error("grid with MPI and ndims=$(ndims(grd)) is not supported.")
#         end
#     end
#     PermutedDimsArray(d, ndims(d):-1:1)
# end



end
