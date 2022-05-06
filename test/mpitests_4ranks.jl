using Devito, LinearAlgebra, MPI, Random, Strided, Test

MPI.Init()
configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", true)

@testset "DevitoMPITimeArray coordinates check" begin
    ny,nx = 4,6

    grd = Grid(shape=(ny,nx), extent=(ny-1,nx-1), dtype=Float32)
    time_order = 1
    fx = TimeFunction(name="fx", grid=grd, time_order=time_order, save=time_order+1)
    fy = TimeFunction(name="fy", grid=grd, time_order=time_order, save=time_order+1)
    sx = SparseTimeFunction(name="sx", grid=grd, npoint=ny*nx, nt=time_order+1)
    sy = SparseTimeFunction(name="sy", grid=grd, npoint=ny*nx, nt=time_order+1)

    cx = [ix-1 for iy = 1:ny, ix=1:nx][:]
    cy = [iy-1 for iy = 1:ny, ix=1:nx][:]

    coords = zeros(Float32, 2, ny*nx)
    coords[1,:] .= cx
    coords[2,:] .= cy
    copy!(coordinates(sx), coords)
    copy!(coordinates(sy), coords)

    datx = reshape(Float32[ix for iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny, time_order+1)
    daty = reshape(Float32[iy for iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny, time_order+1)

    copy!(data(sx), datx)
    copy!(data(sy), daty)

    eqx = inject(sx, field=forward(fx), expr=sx)
    eqy = inject(sy, field=forward(fy), expr=sy)
    op = Operator([eqx, eqy], name="CoordOp")
    apply(op)

    x = convert(Array, data(fx))
    y = convert(Array, data(fy))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if VERSION >= v"1.7"
            @test x ≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0]
            @test y ≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 1.0 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0 4.0]
        else
            _x = zeros(Float32, ny, nx, 2)
            _x[:,:,1] .= [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
            _x[:,:,2] .= [1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0]
            @test x ≈ _x
            _y = zeros(Float32, ny, nx, 2)
            _y[:,:,1] .= [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
            _y[:,:,2] .= [1.0 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0 4.0]
            @test y ≈ _y
        end
    end
end

@testset "DevitoMPITimeArray coordinates check, 3D" begin
    nz,ny,nx = 4,5,6

    grd = Grid(shape=(nz,ny,nx), extent=(nz-1,ny-1,nx-1), dtype=Float32)
    time_order = 1
    fx = TimeFunction(name="fx", grid=grd, time_order=time_order, save=time_order+1)
    fy = TimeFunction(name="fy", grid=grd, time_order=time_order, save=time_order+1)
    fz = TimeFunction(name="fz", grid=grd, time_order=time_order, save=time_order+1)
    sx = SparseTimeFunction(name="sx", grid=grd, npoint=nz*ny*nx, nt=time_order+1)
    sy = SparseTimeFunction(name="sy", grid=grd, npoint=nz*ny*nx, nt=time_order+1)
    sz = SparseTimeFunction(name="sz", grid=grd, npoint=nz*ny*nx, nt=time_order+1)

    cx = [ix-1 for iz = 1:nz, iy = 1:ny, ix=1:nx][:]
    cy = [iy-1 for iz = 1:nz, iy = 1:ny, ix=1:nx][:]
    cz = [iz-1 for iz = 1:nz, iy = 1:ny, ix=1:nx][:]

    coords = zeros(Float32, 3, nz*ny*nx)
    coords[1,:] .= cx
    coords[2,:] .= cy
    coords[3,:] .= cz
    copy!(coordinates(sx), coords)
    copy!(coordinates(sy), coords)
    copy!(coordinates(sz), coords)

    datx = reshape(Float32[ix for iz = 1:nz, iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny*nz, time_order+1)
    daty = reshape(Float32[iy for iz = 1:nz, iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny*nz, time_order+1)
    datz = reshape(Float32[iz for iz = 1:nz, iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny*nz, time_order+1)

    copy!(data(sx), datx)
    copy!(data(sy), daty)
    copy!(data(sz), datz)

    eqx = inject(sx, field=forward(fx), expr=sx)
    eqy = inject(sy, field=forward(fy), expr=sy)
    eqz = inject(sz, field=forward(fz), expr=sz)
    op = Operator([eqx, eqy, eqz], name="CoordOp")
    apply(op)

    x = convert(Array, data(fx))
    y = convert(Array, data(fy))
    z = convert(Array, data(fz))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if VERSION >= v"1.7"
            @test x ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;;; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0;;; 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 2.0;;; 3.0 3.0 3.0 3.0 3.0; 3.0 3.0 3.0 3.0 3.0; 3.0 3.0 3.0 3.0 3.0; 3.0 3.0 3.0 3.0 3.0;;; 4.0 4.0 4.0 4.0 4.0; 4.0 4.0 4.0 4.0 4.0; 4.0 4.0 4.0 4.0 4.0; 4.0 4.0 4.0 4.0 4.0;;; 5.0 5.0 5.0 5.0 5.0; 5.0 5.0 5.0 5.0 5.0; 5.0 5.0 5.0 5.0 5.0; 5.0 5.0 5.0 5.0 5.0;;; 6.0 6.0 6.0 6.0 6.0; 6.0 6.0 6.0 6.0 6.0; 6.0 6.0 6.0 6.0 6.0; 6.0 6.0 6.0 6.0 6.0]
            @test y ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0]
            @test z ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0]
        end
    end
end