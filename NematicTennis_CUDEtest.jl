#Trying to use CUDA to put the game on the GPU. Not very successful so far

using GLMakie
using Random
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Colors
using CUDA

CUDA.allowscalar(false) # Disallow scalar indexing on CuArrays from CPU

# --- Constants for Green Color from Activity ---
const ACTIVITY_GREEN_DIV_STRESS_SQ_THRESHOLD = 0.05f0 # Squared magnitude of div(active_stress) to trigger green
const ACTIVITY_GREEN_SCALAR_ADD = 5.0f0            # Amount green to add if threshold met

# --- Constants ---
const PADDLE_SPEED = 350.0f0
const BALL_SPEED_INIT = 250.0f0
const BALL_SPEED_INCREASE = 20.0f0
const PADDLE_HEIGHT = 100.0f0
const PADDLE_WIDTH = 20.0f0
const BALL_SIZE = 20.0f0
const COURT_WIDTH = 800.0f0
const COURT_HEIGHT = 600.0f0
const SCORE_LIMIT = 5

# --- Fluid Simulation Setup --- Todd C Pomberg
const NX = 320 # Resolution in X
const NY = 240 # Resolution in Y
const DT = 0.001f0 # Time step for Q evolution
const SOLVER_ITER = 20 # Iterations for linear solver (Helmholtz and Poisson)

# --- Active Nematic Dimensionless Parameters ---
const μ = 0.01f0 #viscosity μ
const ζ_friction = 0.0000001f0 #substrate friction ζ
const λ= 0.7f0 #flow alignment λ
const γ = 1.0f0
const α = 0.6f0 # α (activity strength, >0 for extensile)
const NEMATIC_A_PAPER = 1.0f0
const NEMATIC_B_PAPER = -1.0f0
const NEMATIC_C_PAPER = 1.0f0
const NEMATIC_K_ELASTIC_PAPER = 0.01f0

# --- Paddle Jet Constants ---
const PADDLE_JET_FORCE_STRENGTH = 0.2f0 # Strength of the body force applied by the paddle jet
const TARGET_JET_S0 = 0.65f0      # Target nematic scalar order parameter (S₀) in the jet
const JET_CONE_LENGTH_CELLS = 10      # How many cells deep the direct paddle influence is
const JET_MAX_WIDTH_FRAC = 1.0f0 / 2.0f0 # Fraction of paddle height for max jet width at cone tip

# --- Ball/Fluid Interaction ---
const BALL_DRAG_COEFF = 0.2f0
const vel_scale = 1.0f0 # Scales fluid velocity for ball interaction

# --- Pull Constants (Push constants are superseded by Jet constants for the push effect) ---
const PULL_FORCE_STRENGTH = 1000.0f0

# --- Fluid Grid Constants ---
const FLUID_DX = COURT_WIDTH / NX
const FLUID_DY = COURT_HEIGHT / NY
const FLUID_SIZE = (NX + 2) * (NY + 2) # Total size including ghost cells
const INV_DX = 1.0f0 / FLUID_DX
const INV_DY = 1.0f0 / FLUID_DY
const INV_DX2 = INV_DX * INV_DX
const INV_DY2 = INV_DY * INV_DY

# --- Visualization Constants ---
const ARROW_SUBSAMPLE = 1 # Note: If this is >1, update_visualization_data logic for arrows needs care
const ARROW_LENGTH_SCALE = 2.0f0 * min(FLUID_DX, FLUID_DY) * ARROW_SUBSAMPLE

# --- GPU Helper Functions ---
@inline function IX_device(i::Int, j::Int, __NX::Int, __NY::Int) # For use in GPU kernels
    _i = clamp(i, 1, __NX + 2)
    _j = clamp(j, 1, __NY + 2)
    return _i + (_j - 1) * (__NX + 2)
end

@inline function IX_cpu(i, j) # Original for CPU-side logic if needed
    clamp(i, 1, NX + 2) + (clamp(j, 1, NY + 2) - 1) * (NX + 2)
end

@inline function grad_x_centered_device(f::CuDeviceArray{Float32}, i::Int, j::Int, __NX::Int, __NY::Int, _INV_DX::Float32)
    return (f[IX_device(i+1, j, __NX, __NY)] - f[IX_device(i-1, j, __NX, __NY)]) * 0.5f0 * _INV_DX
end

@inline function grad_y_centered_device(f::CuDeviceArray{Float32}, i::Int, j::Int, __NX::Int, __NY::Int, _INV_DY::Float32)
    return (f[IX_device(i, j+1, __NX, __NY)] - f[IX_device(i, j-1, __NX, __NY)]) * 0.5f0 * _INV_DY
end

@inline function laplacian_scalar_device(f::CuDeviceArray{Float32}, i::Int, j::Int, __NX::Int, __NY::Int, _INV_DX2::Float32, _INV_DY2::Float32)
    val_center = f[IX_device(i, j, __NX, __NY)]
    term_x = (f[IX_device(i+1, j, __NX, __NY)] - 2.0f0*val_center + f[IX_device(i-1, j, __NX, __NY)]) * _INV_DX2
    term_y = (f[IX_device(i, j+1, __NX, __NY)] - 2.0f0*val_center + f[IX_device(i, j-1, __NX, __NY)]) * _INV_DY2
    return term_x + term_y
end


# --- Global Fluid Arrays (on GPU) ---
global fluid_vx::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_vy::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_vx0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_vy0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_p::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_div::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)

global fluid_Qxx::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_Qxy::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_Qxx0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_Qxy0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)

global fluid_Hxx::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_Hxy::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)

global fluid_div_sigma_active_x::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_active_y::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_elastic_x::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_elastic_y::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)

# --- Color Tracer Fields (on GPU) ---
global fluid_color_R::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_color_G::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_color_B::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_color_R0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_color_G0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)
global fluid_color_B0::CuArray{Float32} = CUDA.zeros(Float32, FLUID_SIZE)

# Observables for Visualization (remain on CPU, data copied from GPU)
global fluid_color_image_obs = Observable(zeros(RGB{Float32}, NX, NY))
global arrow_pos_obs = Observable(Point2f[])
global arrow_dir_obs = Observable(Vec2f[])


# --- Boundary Condition Kernels (Simplified examples) ---
function set_boundary_scalar_kernel_tb!(x_gpu, __NX, __NY)
    # Kernel for top/bottom boundaries (i from 1 to NX+2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 1 <= i <= __NX + 2
        x_gpu[IX_device(i, 1, __NX, __NY)] = x_gpu[IX_device(i, 2, __NX, __NY)]
        x_gpu[IX_device(i, __NY + 2, __NX, __NY)] = x_gpu[IX_device(i, __NY + 1, __NX, __NY)]
    end
    return
end

function set_boundary_scalar_kernel_lr!(x_gpu, __NX, __NY)
    # Kernel for left/right boundaries (j from 1 to NY+2)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 1 <= j <= __NY + 2
        x_gpu[IX_device(1, j, __NX, __NY)] = x_gpu[IX_device(2, j, __NX, __NY)]
        x_gpu[IX_device(__NX + 2, j, __NX, __NY)] = x_gpu[IX_device(__NX + 1, j, __NX, __NY)]
    end
    return
end

function set_boundary_scalar_kernel_corners!(x_gpu, __NX, __NY)
    # This could be a tiny kernel or part of the above, handling corners carefully.
    # For simplicity, let's make it explicit (though less efficient for just 4 points)
    # Or, handle in the general kernel with if checks, but this is direct.
    # A 2x2 kernel launch could do this.
    tid_x = threadIdx().x
    tid_y = threadIdx().y

    if tid_x == 1 && tid_y == 1 # Bottom-left
        x_gpu[IX_device(1,1, __NX, __NY)] = 0.5f0 * (x_gpu[IX_device(2,1, __NX, __NY)] + x_gpu[IX_device(1,2, __NX, __NY)])
    elseif tid_x == 2 && tid_y == 1 # Bottom-right
        x_gpu[IX_device(__NX+2,1, __NX, __NY)] = 0.5f0 * (x_gpu[IX_device(__NX+1,1, __NX, __NY)] + x_gpu[IX_device(__NX+2,2, __NX, __NY)])
    elseif tid_x == 1 && tid_y == 2 # Top-left
        x_gpu[IX_device(1,__NY+2, __NX, __NY)] = 0.5f0 * (x_gpu[IX_device(2,__NY+2, __NX, __NY)] + x_gpu[IX_device(1,__NY+1, __NX, __NY)])
    elseif tid_x == 2 && tid_y == 2 # Top-right
        x_gpu[IX_device(__NX+2,__NY+2, __NX, __NY)] = 0.5f0 * (x_gpu[IX_device(__NX+1,__NY+2, __NX, __NY)] + x_gpu[IX_device(__NX+2,__NY+1, __NX, __NY)])
    end
    return
end

function set_boundary_scalar_gpu!(x_gpu::CuArray{Float32})
    threads_tb = min(256, NX+2)
    blocks_tb = cld(NX+2, threads_tb)
    @cuda threads=threads_tb blocks=blocks_tb set_boundary_scalar_kernel_tb!(x_gpu, NX, NY)

    threads_lr = min(256, NY+2)
    blocks_lr = cld(NY+2, threads_lr)
    @cuda threads=threads_lr blocks=blocks_lr set_boundary_scalar_kernel_lr!(x_gpu, NX, NY)
    
    @cuda threads=(2,2) blocks=1 set_boundary_scalar_kernel_corners!(x_gpu, NX, NY)
    # CUDA.synchronize() # Often good after boundary conditions if next step depends on it.
end

# TODO GPU: set_boundary_velocity_gpu! is more complex due to 'b' parameter and specific conditions.
# For now, we'll call a CPU version with data copy, or simplify.
# Placeholder - this would be slow due to copying.
function set_boundary_velocity!(b::Int, x_cpu::Vector{Float32}) # Original CPU version
    _NX = NX; _NY = NY # Use global constants
    for i in 1:(_NX+2)
        x_cpu[IX_cpu(i, 1)]   = b == 2 ? 0.0f0 : x_cpu[IX_cpu(i,2)]
        x_cpu[IX_cpu(i, _NY+2)] = b == 2 ? 0.0f0 : x_cpu[IX_cpu(i,_NY+1)]
    end
    for j in 1:(_NY+2)
        x_cpu[IX_cpu(1, j)]   = b == 1 ? 0.0f0 : x_cpu[IX_cpu(2,j)]
        x_cpu[IX_cpu(_NX+2, j)] = b == 1 ? 0.0f0 : x_cpu[IX_cpu(_NX+1,j)]
    end
    # Ensure tangential flow for no-slip, or free-slip based on 'b'
    for i in 1:(_NX+2) # Top/Bottom boundaries
        x_cpu[IX_cpu(i,1)] = b==1 ? x_cpu[IX_cpu(i,2)] : -x_cpu[IX_cpu(i,2)]
        x_cpu[IX_cpu(i,_NY+2)] = b==1 ? x_cpu[IX_cpu(i,_NY+1)] : -x_cpu[IX_cpu(i,_NY+1)]
    end
    for j in 1:(_NY+2) # Left/Right boundaries
        x_cpu[IX_cpu(1,j)] = b==2 ? x_cpu[IX_cpu(2,j)] : -x_cpu[IX_cpu(2,j)]
        x_cpu[IX_cpu(_NX+2,j)] = b==2 ? x_cpu[IX_cpu(_NX+1,j)] : -x_cpu[IX_cpu(_NX+1,j)]
    end

    x_cpu[IX_cpu(1,1)] = 0.5f0*(x_cpu[IX_cpu(1,2)]+x_cpu[IX_cpu(2,1)])
    x_cpu[IX_cpu(1,_NY+2)] = 0.5f0*(x_cpu[IX_cpu(1,_NY+1)]+x_cpu[IX_cpu(2,_NY+2)])
    x_cpu[IX_cpu(_NX+2,1)] = 0.5f0*(x_cpu[IX_cpu(_NX+2,2)]+x_cpu[IX_cpu(_NX+1,1)])
    x_cpu[IX_cpu(_NX+2,_NY+2)] = 0.5f0*(x_cpu[IX_cpu(_NX+2,_NY+1)]+x_cpu[IX_cpu(_NX+1,_NY+2)])
end

function set_boundary_velocity_gpu_stub!(b::Int, x_gpu::CuArray{Float32})
    # This is a STUB. For real GPU performance, specific kernels are needed.
    # This version copies to CPU, processes, then copies back. VERY SLOW.
    x_cpu = Array(x_gpu)
    set_boundary_velocity!(b, x_cpu) # Call original CPU version
    copyto!(x_gpu, x_cpu)
    # CUDA.synchronize()
end


# --- Advection Kernel ---
function advect_kernel!(d_out, d_in, velX, velY, dt, __NX, __NY, _INV_DX, _INV_DY)
    # Kernel indices for interior cells (1 to NX, 1 to NY)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1 
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)

        x_particle_idx = Float32(i_grid) - velX[idx] * dt * _INV_DX
        y_particle_idx = Float32(j_grid) - velY[idx] * dt * _INV_DY
        
        x_particle_idx = clamp(x_particle_idx, 1.5f0, Float32(__NX) + 0.5f0)
        y_particle_idx = clamp(y_particle_idx, 1.5f0, Float32(__NY) + 0.5f0)

        i0 = floor(Int, x_particle_idx); i1 = i0 + 1
        j0 = floor(Int, y_particle_idx); j1 = j0 + 1

        s1 = x_particle_idx - i0; s0 = 1.0f0 - s1
        t1 = y_particle_idx - j0; t0 = 1.0f0 - t1

        val = s0 * (t0 * d_in[IX_device(i0, j0, __NX, __NY)] + t1 * d_in[IX_device(i0, j1, __NX, __NY)]) +
              s1 * (t0 * d_in[IX_device(i1, j0, __NX, __NY)] + t1 * d_in[IX_device(i1, j1, __NX, __NY)])
        d_out[idx] = val
    end
    return
end

function advect_gpu!(b_type::Int, d::CuArray{Float32}, d0::CuArray{Float32}, 
                  velX_gpu::CuArray{Float32}, velY_gpu::CuArray{Float32}, dt_val::Float32)
    
    threads = (16, 16) 
    blocks = (cld(NX, threads[1]), cld(NY, threads[2]))

    @cuda threads=threads blocks=blocks advect_kernel!(d, d0, velX_gpu, velY_gpu, dt_val, NX, NY, INV_DX, INV_DY)

    if b_type == 1 || b_type == 2
        set_boundary_velocity_gpu_stub!(b_type, d) # Placeholder
    else
        set_boundary_scalar_gpu!(d)
    end
end

# --- General Linear Solver Kernel ---
function general_linear_solve_iteration_kernel!(x_out, x_prev_iter, x0_rhs,
                                                coeff_I, coeff_lap_dx, coeff_lap_dy,
                                                __NX, __NY)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)
        
        sum_neighbors_scaled = coeff_lap_dx * (x_prev_iter[IX_device(i_grid-1, j_grid, __NX, __NY)] + x_prev_iter[IX_device(i_grid+1, j_grid, __NX, __NY)]) +
                               coeff_lap_dy * (x_prev_iter[IX_device(i_grid, j_grid-1, __NX, __NY)] + x_prev_iter[IX_device(i_grid, j_grid+1, __NX, __NY)])

        denominator = coeff_I + 2.0f0*coeff_lap_dx + 2.0f0*coeff_lap_dy
        if abs(denominator) < 1e-9 # Avoid division by zero
            x_out[idx] = 0.0f0
        else
            x_out[idx] = (x0_rhs[idx] + sum_neighbors_scaled) / denominator
        end
    end
    return
end

function general_linear_solve_gpu!(b_type::Int, x_out::CuArray{Float32}, x0_rhs::CuArray{Float32},
                                 coeff_I_term::Float32, coeff_lap_term_dx::Float32, coeff_lap_term_dy::Float32,
                                 iterations::Int)
    
    x_prev_iter_gpu = similar(x_out) # Temporary CuArray

    threads = (16, 16)
    blocks = (cld(NX, threads[1]), cld(NY, threads[2]))

    for _ in 1:iterations # k not used
        copyto!(x_prev_iter_gpu, x_out) 
        
        @cuda threads=threads blocks=blocks general_linear_solve_iteration_kernel!(
            x_out, x_prev_iter_gpu, x0_rhs,
            coeff_I_term, coeff_lap_term_dx, coeff_lap_term_dy,
            NX, NY)

        if b_type == 1 || b_type == 2
            set_boundary_velocity_gpu_stub!(b_type, x_out) # Placeholder
        else
            set_boundary_scalar_gpu!(x_out)
        end
    end
end

# --- Project Step Kernels ---
function project_calc_div_kernel!(div_field, velX, velY, __NX, __NY, _INV_DX, _INV_DY)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)
        
        div_field[idx] = (velX[IX_device(i_grid+1, j_grid, __NX, __NY)] - velX[IX_device(i_grid-1, j_grid, __NX, __NY)]) * 0.5f0 * _INV_DX +
                         (velY[IX_device(i_grid, j_grid+1, __NX, __NY)] - velY[IX_device(i_grid, j_grid-1, __NX, __NY)]) * 0.5f0 * _INV_DY
    end
    return
end

function project_poisson_iter_kernel!(p_out, p_old_iter, div_field, __NX, __NY, _INV_DX2, _INV_DY2)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)
        
        sum_lap_neighbors = (p_old_iter[IX_device(i_grid+1,j_grid,__NX,__NY)] + p_old_iter[IX_device(i_grid-1,j_grid,__NX,__NY)]) * _INV_DX2 +
                            (p_old_iter[IX_device(i_grid,j_grid+1,__NX,__NY)] + p_old_iter[IX_device(i_grid,j_grid-1,__NX,__NY)]) * _INV_DY2
        
        denominator = 2.0f0 * (_INV_DX2 + _INV_DY2)
        p_out[idx] = (sum_lap_neighbors - div_field[idx]) / denominator # Make sure div_field sign is correct as per original
    end
    return
end

function project_update_vel_kernel!(velX, velY, p, __NX, __NY, _INV_DX, _INV_DY)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)
        
        velX[idx] -= (p[IX_device(i_grid+1, j_grid, __NX, __NY)] - p[IX_device(i_grid-1, j_grid, __NX, __NY)]) * 0.5f0 * _INV_DX
        velY[idx] -= (p[IX_device(i_grid, j_grid+1, __NX, __NY)] - p[IX_device(i_grid, j_grid-1, __NX, __NY)]) * 0.5f0 * _INV_DY
    end
    return
end

function project_gpu!(velX::CuArray{Float32}, velY::CuArray{Float32}, p::CuArray{Float32}, div_field::CuArray{Float32})
    threads = (16,16)
    blocks = (cld(NX, threads[1]), cld(NY, threads[2]))

    CUDA.fill!(p, 0.0f0) # Zero out pressure field initially
    @cuda threads=threads blocks=blocks project_calc_div_kernel!(div_field, velX, velY, NX, NY, INV_DX, INV_DY)
    set_boundary_scalar_gpu!(div_field)
    set_boundary_scalar_gpu!(p) # p is zero but boundaries need to reflect that based on scalar rules

    p_old_iter_gpu = similar(p)
    for _ in 1:SOLVER_ITER
        copyto!(p_old_iter_gpu, p)
        @cuda threads=threads blocks=blocks project_poisson_iter_kernel!(p, p_old_iter_gpu, div_field, NX, NY, INV_DX2, INV_DY2)
        # CUDA.synchronize()
        set_boundary_scalar_gpu!(p)
    end

    @cuda threads=threads blocks=blocks project_update_vel_kernel!(velX, velY, p, NX, NY, INV_DX, INV_DY)
    set_boundary_velocity_gpu_stub!(1, velX) # Placeholder
    set_boundary_velocity_gpu_stub!(2, velY) # Placeholder
end


# --- Nematic Calculation Kernels ---
function calculate_H_kernel!(Hxx_out, Hxy_out, Qxx_in, Qxy_in,
                             K_elastic, A_coeff, C_coeff, __NX, __NY, _INV_DX2, _INV_DY2)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)

        qxx_val = Qxx_in[idx]
        qxy_val = Qxy_in[idx]

        lap_qxx = laplacian_scalar_device(Qxx_in, i_grid, j_grid, __NX, __NY, _INV_DX2, _INV_DY2)
        lap_qxy = laplacian_scalar_device(Qxy_in, i_grid, j_grid, __NX, __NY, _INV_DX2, _INV_DY2)
        trQ2_val = 2.0f0 * (qxx_val*qxx_val + qxy_val*qxy_val)

        Hxx_out[idx] = -(A_coeff * qxx_val + 2.0f0 * C_coeff * trQ2_val * qxx_val - K_elastic * lap_qxx)
        Hxy_out[idx] = -(A_coeff * qxy_val + 2.0f0 * C_coeff * trQ2_val * qxy_val - K_elastic * lap_qxy)
    end
    return
end

function calculate_H_gpu!(Hxx_out, Hxy_out, Qxx_in, Qxy_in, K_elastic, A_coeff, C_coeff)
    threads = (16,16)
    blocks = (cld(NX, threads[1]), cld(NY, threads[2]))
    @cuda threads=threads blocks=blocks calculate_H_kernel!(Hxx_out, Hxy_out, Qxx_in, Qxy_in,
                                                        K_elastic, A_coeff, C_coeff, NX, NY, INV_DX2, INV_DY2)
    set_boundary_scalar_gpu!(Hxx_out)
    set_boundary_scalar_gpu!(Hxy_out)
end


function calculate_S_flow_term_kernel!(Sxx_out, Sxy_out, Qxx_in, Qxy_in, vx, vy, lambda_align,
                                       __NX, __NY, _INV_DX, _INV_DY)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)

        qxx = Qxx_in[idx]
        qxy = Qxy_in[idx]

        dvx_dx = grad_x_centered_device(vx,i_grid,j_grid, __NX, __NY, _INV_DX)
        dvx_dy = grad_y_centered_device(vx,i_grid,j_grid, __NX, __NY, _INV_DY)
        dvy_dx = grad_x_centered_device(vy,i_grid,j_grid, __NX, __NY, _INV_DX)
        dvy_dy = grad_y_centered_device(vy,i_grid,j_grid, __NX, __NY, _INV_DY)
        
        Dxx = dvx_dx 
        Dxy = 0.5f0 * (dvx_dy + dvy_dx)
        omega_xy_val = 0.5f0 * (dvy_dx - dvx_dy) 

        comm_Omega_Q_xx = 2.0f0 * omega_xy_val * qxy
        comm_Omega_Q_xy = -2.0f0 * omega_xy_val * qxx 

        Sxx_out[idx] = lambda_align * Dxx - comm_Omega_Q_xx
        Sxy_out[idx] = lambda_align * Dxy - comm_Omega_Q_xy
    end
    return
end

function calculate_S_flow_term_gpu!(Sxx_out, Sxy_out, Qxx_in, Qxy_in, vx, vy, lambda_align)
    threads = (16,16)
    blocks = (cld(NX, threads[1]), cld(NY, threads[2]))
    @cuda threads=threads blocks=blocks calculate_S_flow_term_kernel!(Sxx_out, Sxy_out, Qxx_in, Qxy_in,
                                                                    vx, vy, lambda_align, NX, NY, INV_DX, INV_DY)
end


function calculate_div_active_stress_kernel!(div_sx, div_sy, Qxx_in, Qxy_in, alpha_activity,
                                              __NX, __NY, _INV_DX, _INV_DY)
    ix_kern = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy_kern = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (1 <= ix_kern <= __NX) && (1 <= iy_kern <= __NY)
        i_grid = ix_kern + 1
        j_grid = iy_kern + 1
        idx = IX_device(i_grid, j_grid, __NX, __NY)

        dQxx_dx = grad_x_centered_device(Qxx_in,i_grid,j_grid,__NX,__NY,_INV_DX)
        dQxx_dy = grad_y_centered_device(Qxx_in,i_grid,j_grid,__NX,__NY,_INV_DY)
        dQxy_dx = grad_x_centered_device(Qxy_in,i_grid,j_grid,__NX,__NY,_INV_DX)
        dQxy_dy = grad_y_centered_device(Qxy_in,i_grid,j_grid,__NX,__NY,_INV_DY)

        div_sx[idx] = -alpha_activity * (dQxx_dx + dQxy_dy)
        div_sy[idx] = -alpha_activity * (dQxy_dx - dQxx_dy) 
    end
    return
end
function calculate_div_active_stress_gpu!(div_sx, div_sy, Qxx_in, Qxy_in, alpha_activity)
    threads = (16,16)
    blocks = (cld(NX, threads[1]), cld(NY, threads[2]))
    @cuda threads=threads blocks=blocks calculate_div_active_stress_kernel!(div_sx, div_sy, Qxx_in, Qxy_in, alpha_activity, NX, NY, INV_DX, INV_DY)
    # CUDA.synchronize()
end

function calculate_div_elastic_stress!(div_out_x_cpu::Vector{Float32}, div_out_y_cpu::Vector{Float32},
                                       Qxx_vec_cpu::Vector{Float32}, Qxy_vec_cpu::Vector{Float32},
                                       Hxx_vec_cpu::Vector{Float32}, Hxy_vec_cpu::Vector{Float32},
                                       K_elastic::Float32, lambda_align::Float32)
    
    _NX = NX; _NY = NY; _FLUID_SIZE = FLUID_SIZE
    _INV_DX = INV_DX; _INV_DY = INV_DY; _INV_DX2 = INV_DX2; _INV_DY2 = INV_DY2

    grad_x_c = (f,i,j) -> (f[IX_cpu(i+1,j)] - f[IX_cpu(i-1,j)]) * 0.5f0 * _INV_DX
    grad_y_c = (f,i,j) -> (f[IX_cpu(i,j+1)] - f[IX_cpu(i,j-1)]) * 0.5f0 * _INV_DY
    
    sigma_el_xx_temp = zeros(Float32, _FLUID_SIZE)
    sigma_el_xy_temp = zeros(Float32, _FLUID_SIZE)
    sigma_el_yy_temp = zeros(Float32, _FLUID_SIZE)

    for j_idx in 2:(_NY+1) # Using @threads here is fine for CPU version
        for i_idx in 2:(_NX+1)
            idx = IX_cpu(i_idx,j_idx)
            q_xx = Qxx_vec_cpu[idx]; q_xy = Qxy_vec_cpu[idx]
            h_xx = Hxx_vec_cpu[idx]; h_xy = Hxy_vec_cpu[idx]
            
            comm_QH_xy_val = 2.0f0 * (q_xx * h_xy - q_xy * h_xx)
            sigma_el_xx_temp[idx] = 0.0f0 # Original was -lambda_align * 0.0
            sigma_el_xy_temp[idx] = -lambda_align * comm_QH_xy_val
            sigma_el_yy_temp[idx] = 0.0f0 # Original was lambda_align * 0.0
        end
    end

    for j_idx in 2:(_NY+1)
        for i_idx in 2:(_NX+1)
            idx = IX_cpu(i_idx, j_idx)
            dqxx_dx = grad_x_c(Qxx_vec_cpu, i_idx, j_idx)
            dqxy_dx = grad_x_c(Qxy_vec_cpu, i_idx, j_idx)
            dqxx_dy = grad_y_c(Qxx_vec_cpu, i_idx, j_idx)
            dqxy_dy = grad_y_c(Qxy_vec_cpu, i_idx, j_idx)

            sigma_K_xx_val = -K_elastic * 2.0f0 * (dqxx_dx^2 + dqxy_dx^2)
            sigma_K_xy_val = -K_elastic * 2.0f0 * (dqxx_dx * dqxx_dy + dqxy_dx * dqxy_dy)
            sigma_K_yy_val = -K_elastic * 2.0f0 * (dqxx_dy^2 + dqxy_dy^2)

            sigma_el_xx_temp[idx] += sigma_K_xx_val
            sigma_el_xy_temp[idx] += sigma_K_xy_val
            sigma_el_yy_temp[idx] += sigma_K_yy_val
        end
    end
    
    set_b_scalar_cpu = (x_arr) -> begin
        for i in 1:(_NX+2); x_arr[IX_cpu(i, 1)] = x_arr[IX_cpu(i, 2)]; x_arr[IX_cpu(i, _NY + 2)] = x_arr[IX_cpu(i, _NY + 1)]; end
        for j in 1:(_NY+2); x_arr[IX_cpu(1, j)] = x_arr[IX_cpu(2, j)]; x_arr[IX_cpu(_NX + 2, j)] = x_arr[IX_cpu(_NX + 1, j)]; end
        x_arr[IX_cpu(1,1)] = 0.5f0*(x_arr[IX_cpu(2,1)]+x_arr[IX_cpu(1,2)]); x_arr[IX_cpu(1,_NY+2)] = 0.5f0*(x_arr[IX_cpu(2,_NY+2)]+x_arr[IX_cpu(1,_NY+1)])
        x_arr[IX_cpu(_NX+2,1)] = 0.5f0*(x_arr[IX_cpu(_NX+1,1)]+x_arr[IX_cpu(_NX+2,2)]); x_arr[IX_cpu(_NX+2,_NY+2)] = 0.5f0*(x_arr[IX_cpu(_NX+1,_NY+2)]+x_arr[IX_cpu(_NX+2,_NY+1)])
    end
    set_b_scalar_cpu(sigma_el_xx_temp)
    set_b_scalar_cpu(sigma_el_xy_temp)
    set_b_scalar_cpu(sigma_el_yy_temp)

    for j_idx in 2:(_NY+1)
        for i_idx in 2:(_NX+1)
            idx = IX_cpu(i_idx, j_idx)
            div_out_x_cpu[idx] = grad_x_c(sigma_el_xx_temp, i_idx, j_idx) + grad_y_c(sigma_el_xy_temp, i_idx, j_idx)
            div_out_y_cpu[idx] = grad_x_c(sigma_el_xy_temp, i_idx, j_idx) + grad_y_c(sigma_el_yy_temp, i_idx, j_idx)
        end
    end
end

# --- Paddle Jet Kernels ---

function add_jet_force_kernel!(force_vx_array, paddle_id, paddle_center_gy_idx, force_strength,
                               start_gxi_jet_idx, base_half_width_cells, tip_half_width_cells,
                               __NX, __NY, _JET_CONE_LENGTH_CELLS, _JET_MAX_WIDTH_FRAC)

    i_dist_cell = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 0 to JET_CONE_LENGTH_CELLS-1
    gy_offset_loop_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y # Index for width loop
    max_possible_jet_half_width = max(1, round(Int, (PADDLE_HEIGHT / FLUID_DY * _JET_MAX_WIDTH_FRAC) / 2.0))

    if 0 <= i_dist_cell < _JET_CONE_LENGTH_CELLS
        current_gxi = (paddle_id == 1) ? (start_gxi_jet_idx + i_dist_cell) : (start_gxi_jet_idx - i_dist_cell)

        if !(2 <= current_gxi <= __NX + 1)
            return # Early exit is fine
        end

        width_frac = _JET_CONE_LENGTH_CELLS <= 1 ? 1.0f0 : Float32(i_dist_cell) / max(1.0f0, Float32(_JET_CONE_LENGTH_CELLS - 1))
        current_jet_half_width_cells = round(Int, base_half_width_cells + (tip_half_width_cells - base_half_width_cells) * width_frac)
        current_jet_half_width_cells = max(1, current_jet_half_width_cells)

        if gy_offset_loop_idx <= (2 * current_jet_half_width_cells - 1)
            gy_offset = gy_offset_loop_idx - current_jet_half_width_cells # Shift to be centered around 0

            current_gyi = clamp(paddle_center_gy_idx + gy_offset, 2, __NY + 1)
            idx_cone = IX_device(current_gxi, current_gyi, __NX, __NY)

            jet_force_x_actual = (paddle_id == 1) ? force_strength : -force_strength
            CUDA.@atomic force_vx_array[idx_cone] += jet_force_x_actual
        end
    end
    return 
end



function _add_jet_force_in_cone_gpu!(
    force_vx_array::CuArray{Float32}, paddle_id::Int, paddle_base_y_world::Float32, force_strength::Float32
)
    paddle_center_y_world = paddle_base_y_world + PADDLE_HEIGHT / 2.0f0
    _, paddle_center_gy_float = world_to_grid(Point2f(0f0, paddle_center_y_world)) # world_to_grid is CPU
    paddle_center_gy_idx = clamp(round(Int, paddle_center_gy_float), 2, NY + 1)

    paddle_height_cells = PADDLE_HEIGHT / FLUID_DY
    
    start_x_world_edge = (paddle_id == 1) ? PADDLE_WIDTH : (COURT_WIDTH - PADDLE_WIDTH)
    start_gxi_jet_float = if paddle_id == 1
        world_to_grid(Point2f(start_x_world_edge + FLUID_DX * 0.5f0, 0f0))[1]
    else
        world_to_grid(Point2f(start_x_world_edge - FLUID_DX * 0.5f0, 0f0))[1]
    end
    start_gxi_jet_idx = clamp(round(Int, start_gxi_jet_float), 2, NX + 1)

    base_half_width_cells = max(1, round(Int, (paddle_height_cells / 3.0) / 2.0))
    max_jet_width_at_tip_cells = max(1, round(Int, paddle_height_cells * JET_MAX_WIDTH_FRAC))
    tip_half_width_cells = max(1, round(Int, max_jet_width_at_tip_cells / 2.0))

    threads_y_dim = max(1, 2 * round(Int, max(base_half_width_cells, tip_half_width_cells)) -1)
    
    threads = (min(32, JET_CONE_LENGTH_CELLS), min(32, threads_y_dim) )
    blocks = (cld(JET_CONE_LENGTH_CELLS, threads[1]), cld(threads_y_dim, threads[2]))

    @cuda threads=threads blocks=blocks add_jet_force_kernel!(
        force_vx_array, paddle_id, paddle_center_gy_idx, force_strength,
        start_gxi_jet_idx, base_half_width_cells, tip_half_width_cells,
        NX, NY, JET_CONE_LENGTH_CELLS, JET_MAX_WIDTH_FRAC)
end

function apply_paddle_jet_effect_kernel!(fluid_R, fluid_G, fluid_B,
                                         paddle_id, paddle_center_gy_idx,
                                         start_gxi_jet_idx, base_half_width_cells, tip_half_width_cells,
                                         __NX, __NY, _JET_CONE_LENGTH_CELLS, _JET_MAX_WIDTH_FRAC)
    i_dist_cell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gy_offset_loop_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    max_possible_jet_half_width = max(1, round(Int, (PADDLE_HEIGHT / FLUID_DY * _JET_MAX_WIDTH_FRAC) / 2.0))

    if 0 <= i_dist_cell < _JET_CONE_LENGTH_CELLS
        current_gxi = (paddle_id == 1) ? (start_gxi_jet_idx + i_dist_cell) : (start_gxi_jet_idx - i_dist_cell)

        if !(2 <= current_gxi <= __NX + 1)
             return # Early exit
        end

        width_frac = _JET_CONE_LENGTH_CELLS <= 1 ? 1.0f0 : Float32(i_dist_cell) / max(1.0f0, Float32(_JET_CONE_LENGTH_CELLS - 1))
        current_jet_half_width_cells = round(Int, base_half_width_cells + (tip_half_width_cells - base_half_width_cells) * width_frac)
        current_jet_half_width_cells = max(1, current_jet_half_width_cells)

        if gy_offset_loop_idx <= (2 * current_jet_half_width_cells - 1)
            gy_offset = gy_offset_loop_idx - current_jet_half_width_cells

            current_gyi = clamp(paddle_center_gy_idx + gy_offset, 2, __NY + 1)
            idx_cone = IX_device(current_gxi, current_gyi, __NX, __NY)

            if paddle_id == 1
                fluid_R[idx_cone] = 1.0f0; fluid_G[idx_cone] = 0.0f0; fluid_B[idx_cone] = 0.0f0
            else
                fluid_R[idx_cone] = 0.0f0; fluid_G[idx_cone] = 0.0f0; fluid_B[idx_cone] = 1.0f0
            end
        end
    end
    return # Explicitly return nothing
end

function apply_paddle_jet_effect_gpu!(paddle_id::Int, paddle_base_y_world::Float32)
    paddle_center_y_world = paddle_base_y_world + PADDLE_HEIGHT / 2.0f0
    _, paddle_center_gy_float = world_to_grid(Point2f(0f0, paddle_center_y_world))
    paddle_center_gy_idx = clamp(round(Int, paddle_center_gy_float), 2, NY + 1)

    paddle_height_cells = PADDLE_HEIGHT / FLUID_DY
    start_x_world_edge = (paddle_id == 1) ? PADDLE_WIDTH : (COURT_WIDTH - PADDLE_WIDTH)
    start_gxi_jet_float = (paddle_id == 1) ? world_to_grid(Point2f(start_x_world_edge + FLUID_DX*0.5f0,0f0))[1] : world_to_grid(Point2f(start_x_world_edge - FLUID_DX*0.5f0,0f0))[1]
    start_gxi_jet_idx = clamp(round(Int, start_gxi_jet_float), 2, NX + 1)

    base_half_width_cells = max(1, round(Int, (paddle_height_cells / 3.0) / 2.0))
    max_jet_width_at_tip_cells = max(1, round(Int, paddle_height_cells * JET_MAX_WIDTH_FRAC))
    tip_half_width_cells = max(1, round(Int, max_jet_width_at_tip_cells / 2.0))
    
    threads_y_dim = max(1, 2 * round(Int, max(base_half_width_cells, tip_half_width_cells)) -1)
    threads = (min(32, JET_CONE_LENGTH_CELLS), min(32, threads_y_dim) )
    blocks = (cld(JET_CONE_LENGTH_CELLS, threads[1]), cld(threads_y_dim, threads[2]))

    @cuda threads=threads blocks=blocks apply_paddle_jet_effect_kernel!(
        fluid_color_R, fluid_color_G, fluid_color_B,
        paddle_id, paddle_center_gy_idx,
        start_gxi_jet_idx, base_half_width_cells, tip_half_width_cells,
        NX, NY, JET_CONE_LENGTH_CELLS, JET_MAX_WIDTH_FRAC
    )
end

# --- Fluid Step (GPU version) ---
function update_Q_kernel!(qxx, qxy, sxx_flow, sxy_flow, hxx, hxy, dt_q, gamma_val, len)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= len
        qxx[idx] += dt_q * (sxx_flow[idx] + gamma_val * hxx[idx])
        qxy[idx] += dt_q * (sxy_flow[idx] + gamma_val * hxy[idx])
    end
    return
end

function add_green_color_kernel!(color_g, div_active_x, div_active_y, threshold_sq, add_val, len)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= len
        active_force_sq_mag = div_active_x[idx]^2 + div_active_y[idx]^2
        if active_force_sq_mag > threshold_sq
            color_g[idx] = min(1.0f0, color_g[idx] + add_val)
        end
    end
    return
end

function sum_forces_kernel!(vx0_out, vy0_out, div_active_x, div_active_y, div_elastic_x, div_elastic_y, len)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= len
        vx0_out[idx] = div_active_x[idx] + div_elastic_x[idx]
        vy0_out[idx] = div_active_y[idx] + div_elastic_y[idx]
    end
    return
end


function fluid_step_gpu!(p1_is_pushing::Bool, p1_y_world::Float32, p2_is_pushing::Bool, p2_y_world::Float32)
    dt_q_evolution = DT

    copyto!(fluid_Qxx0, fluid_Qxx) # Qxx0 = Qxx_current (source for advection)
    copyto!(fluid_Qxy0, fluid_Qxy) # Qxy0 = Qxy_current (source for advection)
    advect_gpu!(0, fluid_Qxx, fluid_Qxx0, fluid_vx, fluid_vy, dt_q_evolution) # fluid_Qxx is now advected Qxx
    advect_gpu!(0, fluid_Qxy, fluid_Qxy0, fluid_vx, fluid_vy, dt_q_evolution) # fluid_Qxy is now advected Qxy

    copyto!(fluid_color_R0, fluid_color_R); advect_gpu!(0, fluid_color_R, fluid_color_R0, fluid_vx, fluid_vy, dt_q_evolution)
    copyto!(fluid_color_G0, fluid_color_G); advect_gpu!(0, fluid_color_G, fluid_color_G0, fluid_vx, fluid_vy, dt_q_evolution)
    copyto!(fluid_color_B0, fluid_color_B); advect_gpu!(0, fluid_color_B, fluid_color_B0, fluid_vx, fluid_vy, dt_q_evolution)

    calculate_H_gpu!(fluid_Hxx, fluid_Hxy, fluid_Qxx, fluid_Qxy,
                     NEMATIC_K_ELASTIC_PAPER, NEMATIC_A_PAPER, NEMATIC_C_PAPER)

    Sxx_flow_temp_gpu = CUDA.zeros(Float32, FLUID_SIZE)
    Sxy_flow_temp_gpu = CUDA.zeros(Float32, FLUID_SIZE)
    calculate_S_flow_term_gpu!(Sxx_flow_temp_gpu, Sxy_flow_temp_gpu, fluid_Qxx, fluid_Qxy,
                               fluid_vx, fluid_vy, λ) # Using current fluid_vx/vy

    threads_1d = 256
    blocks_1d = cld(FLUID_SIZE, threads_1d)
    @cuda threads=threads_1d blocks=blocks_1d update_Q_kernel!(
        fluid_Qxx, fluid_Qxy, Sxx_flow_temp_gpu, Sxy_flow_temp_gpu, 
        fluid_Hxx, fluid_Hxy, dt_q_evolution, γ, FLUID_SIZE)
    set_boundary_scalar_gpu!(fluid_Qxx)
    set_boundary_scalar_gpu!(fluid_Qxy)

    calculate_H_gpu!(fluid_Hxx, fluid_Hxy, fluid_Qxx, fluid_Qxy,
                     NEMATIC_K_ELASTIC_PAPER, NEMATIC_A_PAPER, NEMATIC_C_PAPER)

    calculate_div_active_stress_gpu!(fluid_div_sigma_active_x, fluid_div_sigma_active_y,
                                     fluid_Qxx, fluid_Qxy, α)
    
    div_elastic_x_cpu = zeros(Float32, FLUID_SIZE); div_elastic_y_cpu = zeros(Float32, FLUID_SIZE)
    Qxx_cpu = Array(fluid_Qxx); Qxy_cpu = Array(fluid_Qxy)
    Hxx_cpu = Array(fluid_Hxx); Hxy_cpu = Array(fluid_Hxy)
    calculate_div_elastic_stress!(div_elastic_x_cpu, div_elastic_y_cpu,
                                  Qxx_cpu, Qxy_cpu, Hxx_cpu, Hxy_cpu,
                                  NEMATIC_K_ELASTIC_PAPER, λ)
    copyto!(fluid_div_sigma_elastic_x, div_elastic_x_cpu)
    copyto!(fluid_div_sigma_elastic_y, div_elastic_y_cpu)

    @cuda threads=threads_1d blocks=blocks_1d add_green_color_kernel!(
        fluid_color_G, fluid_div_sigma_active_x, fluid_div_sigma_active_y,
        ACTIVITY_GREEN_DIV_STRESS_SQ_THRESHOLD, ACTIVITY_GREEN_SCALAR_ADD, FLUID_SIZE)

    @cuda threads=threads_1d blocks=blocks_1d sum_forces_kernel!(
        fluid_vx0, fluid_vy0, 
        fluid_div_sigma_active_x, fluid_div_sigma_active_y, 
        fluid_div_sigma_elastic_x, fluid_div_sigma_elastic_y, FLUID_SIZE)
    # CUDA.synchronize()

    if p1_is_pushing
        _add_jet_force_in_cone_gpu!(fluid_vx0, 1, p1_y_world, PADDLE_JET_FORCE_STRENGTH)
    end
    if p2_is_pushing
        _add_jet_force_in_cone_gpu!(fluid_vx0, 2, p2_y_world, PADDLE_JET_FORCE_STRENGTH)
    end

    general_linear_solve_gpu!(1, fluid_vx, fluid_vx0, # fluid_vx is output, fluid_vx0 is RHS
                              ζ_friction, μ * INV_DX2, μ * INV_DY2, SOLVER_ITER)
    general_linear_solve_gpu!(2, fluid_vy, fluid_vy0,
                              ζ_friction, μ * INV_DX2, μ * INV_DY2, SOLVER_ITER)

    project_gpu!(fluid_vx, fluid_vy, fluid_p, fluid_div)
end


# --- CPU Helper Functions (Unchanged unless they interact with GPU data directly) ---
function world_to_grid(pos::Point2f)
    gx = (pos[1] / COURT_WIDTH) * NX + 1.5f0; 
    gy = (pos[2] / COURT_HEIGHT) * NY + 1.5f0;
    return gx, gy
end

function get_fluid_velocity_at_cpu(pos::Point2f, vx_cpu::Vector{Float32}, vy_cpu::Vector{Float32})::Vec2f
    gx, gy = world_to_grid(pos)
    gx = clamp(gx, 1.5f0, NX + 0.5f0) 
    gy = clamp(gy, 1.5f0, NY + 0.5f0)

    i0 = floor(Int, gx); i1 = i0 + 1
    j0 = floor(Int, gy); j1 = j0 + 1

    s1 = gx - i0; s0 = 1.0f0 - s1
    t1 = gy - j0; t0 = 1.0f0 - t1

    # Use IX_cpu for CPU arrays
    vx = s0*(t0*vx_cpu[IX_cpu(i0, j0)] + t1*vx_cpu[IX_cpu(i0, j1)]) + s1*(t0*vx_cpu[IX_cpu(i1, j0)] + t1*vx_cpu[IX_cpu(i1, j1)])
    vy = s0*(t0*vy_cpu[IX_cpu(i0, j0)] + t1*vy_cpu[IX_cpu(i0, j1)]) + s1*(t0*vy_cpu[IX_cpu(i1, j0)] + t1*vy_cpu[IX_cpu(i1, j1)])
    return Vec2f(vx, vy) * vel_scale
end


function rect_overlaps(r1::Rect2f, r2::Rect2f) # CPU function, no change
    x_overlap = (r1.origin[1] < r2.origin[1] + r2.widths[1]) && (r1.origin[1] + r1.widths[1] > r2.origin[1])
    y_overlap = (r1.origin[2] < r2.origin[2] + r2.widths[2]) && (r1.origin[2] + r1.widths[2] > r2.origin[2])
    return x_overlap && y_overlap
end


# --- Run Function ---
function run()
    fig = Figure(size=(COURT_WIDTH, COURT_HEIGHT), backgroundcolor=:black, figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect(), limits=(0, COURT_WIDTH, 0, COURT_HEIGHT), backgroundcolor=:dimgrey)
    hidedecorations!(ax); hidespines!(ax)

    # Initialize fluid arrays on GPU
    CUDA.fill!(fluid_vx, 0.0f0); CUDA.fill!(fluid_vy, 0.0f0); CUDA.fill!(fluid_vx0, 0.0f0); CUDA.fill!(fluid_vy0, 0.0f0)
    CUDA.fill!(fluid_p, 0.0f0); CUDA.fill!(fluid_div, 0.0f0)

    rand_scale = 0.01f0
    # Initialize Qxx, Qxy on GPU
    rand_gpu_1 = CUDA.rand(Float32, FLUID_SIZE)
    rand_gpu_2 = CUDA.rand(Float32, FLUID_SIZE)
    
    function init_Q_kernel!(q_arr, rand_vals, scale, len)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if idx <= len
            q_arr[idx] = (rand_vals[idx] - 0.5f0) * 2.0f0 * scale
        end
        return
    end
    threads_1d_init = 256
    blocks_1d_init = cld(FLUID_SIZE, threads_1d_init)
    @cuda threads=threads_1d_init blocks=blocks_1d_init init_Q_kernel!(fluid_Qxx, rand_gpu_1, rand_scale, FLUID_SIZE)
    @cuda threads=threads_1d_init blocks=blocks_1d_init init_Q_kernel!(fluid_Qxy, rand_gpu_2, rand_scale, FLUID_SIZE)
    # CUDA.synchronize()
    set_boundary_scalar_gpu!(fluid_Qxx); set_boundary_scalar_gpu!(fluid_Qxy)
    
    CUDA.fill!(fluid_Qxx0, 0.0f0); CUDA.fill!(fluid_Qxy0, 0.0f0)
    CUDA.fill!(fluid_Hxx, 0.0f0); CUDA.fill!(fluid_Hxy, 0.0f0)
    CUDA.fill!(fluid_div_sigma_active_x, 0.0f0); CUDA.fill!(fluid_div_sigma_active_y, 0.0f0)
    CUDA.fill!(fluid_div_sigma_elastic_x, 0.0f0); CUDA.fill!(fluid_div_sigma_elastic_y, 0.0f0)

    CUDA.fill!(fluid_color_R, 0.0f0); CUDA.fill!(fluid_color_G, 0.0f0); CUDA.fill!(fluid_color_B, 0.0f0)
    CUDA.fill!(fluid_color_R0, 0.0f0); CUDA.fill!(fluid_color_G0, 0.0f0); CUDA.fill!(fluid_color_B0, 0.0f0)
    set_boundary_scalar_gpu!(fluid_color_R); set_boundary_scalar_gpu!(fluid_color_G); set_boundary_scalar_gpu!(fluid_color_B)

    # Arrow setup (CPU)
    arrow_positions = Point2f[];
    fluid_x_range_vis = LinRange(0+FLUID_DX/2, COURT_WIDTH-FLUID_DX/2, NX);
    fluid_y_range_vis = LinRange(0+FLUID_DY/2, COURT_HEIGHT-FLUID_DY/2, NY);
    for j_vis in 1:ARROW_SUBSAMPLE:NY, i_vis in 1:ARROW_SUBSAMPLE:NX # Corrected loop variable names
        push!(arrow_positions, Point2f(fluid_x_range_vis[i_vis], fluid_y_range_vis[j_vis]));
    end
    arrow_pos_obs[] = arrow_positions
    arrow_dir_obs[] = fill(Vec2f(1,0), length(arrow_positions))

    # Game state observables (CPU)
    ball_pos = Observable(Point2f(COURT_WIDTH / 2f0, COURT_HEIGHT / 2f0)); prev_ball_pos = Observable(Point2f(COURT_WIDTH / 2f0, COURT_HEIGHT / 2f0))
    ball_vel = Observable(Vec2f(0f0, 0f0)); current_ball_speed = Observable(BALL_SPEED_INIT)
    paddle_left_y = Observable(COURT_HEIGHT / 2f0 - PADDLE_HEIGHT / 2f0); paddle_right_y = Observable(COURT_HEIGHT / 2f0 - PADDLE_HEIGHT / 2f0)
    score_left = Observable(0); score_right = Observable(0); game_active = Observable(true)
    game_message = Observable("Press Serve Keys to Start"); serve_state = Observable(:p1_serve); served_by = Observable(:p1)
    last_update_time = Ref(time()); fluid_time_accumulator = Ref(0.0f0)

    image!(ax, (0.0f0, COURT_WIDTH), (0.0f0, COURT_HEIGHT), fluid_color_image_obs, interpolate=true) # fluid_color_image_obs updated from GPU data
    arrows!(ax, arrow_pos_obs, arrow_dir_obs, arrowsize=Vec2f(ARROW_LENGTH_SCALE*0.25, ARROW_LENGTH_SCALE*0.35), lengthscale=ARROW_LENGTH_SCALE, arrowcolor=:white, linecolor=:white, linewidth=0.5)
    ball_visual_size = BALL_SIZE * 0.9f0
    ball_rect_obs = @lift Rect2f($ball_pos[1]-ball_visual_size/2f0, $ball_pos[2]-ball_visual_size/2f0, ball_visual_size, ball_visual_size)
    poly!(ax, ball_rect_obs, color=:yellow, strokecolor=:orange, strokewidth=2)
    poly!(ax, @lift(Rect2f(0, $paddle_left_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color=:lightcyan)
    poly!(ax, @lift(Rect2f(COURT_WIDTH - PADDLE_WIDTH, $paddle_right_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color=:lightcyan)
    score_text_obs = @lift "$($score_left) - $($score_right)"
    text!(ax, score_text_obs, position=Point2f(COURT_WIDTH/2f0, COURT_HEIGHT-20f0), fontsize=40, color=:white, align=(:center, :top))
    text!(ax, game_message, position=Point2f(COURT_WIDTH/2f0, 30f0), fontsize=30, color=:yellow, align=(:center, :bottom))


    function reset_ball() # CPU function
        current_ball_speed[] = BALL_SPEED_INIT; ball_vel[] = Vec2f(0f0, 0f0)
        local_paddle_y = 0.0f0; ball_x = 0.0f0
        if served_by[] == :p1
            local_paddle_y = paddle_left_y[]; ball_x = PADDLE_WIDTH+BALL_SIZE/2f0; serve_state[] = :p1_serve; game_message[] = "P1: Use A/D or Space to Serve/Push/Pull"
        else
            local_paddle_y = paddle_right_y[]; ball_x = COURT_WIDTH-PADDLE_WIDTH-BALL_SIZE/2f0; serve_state[] = :p2_serve; game_message[] = "P2: Use ←/→ or Enter to Serve/Push/Pull"
        end
        ball_y = local_paddle_y + PADDLE_HEIGHT/2f0; new_pos = Point2f(ball_x, clamp(ball_y, BALL_SIZE/2f0, COURT_HEIGHT-BALL_SIZE/2f0))
        ball_pos[] = new_pos; prev_ball_pos[] = new_pos
    end
    reset_ball()

    local fluid_vx_cpu = zeros(Float32, FLUID_SIZE)
    local fluid_vy_cpu = zeros(Float32, FLUID_SIZE)
    local fluid_color_R_cpu = zeros(Float32, FLUID_SIZE)
    local fluid_color_G_cpu = zeros(Float32, FLUID_SIZE)
    local fluid_color_B_cpu = zeros(Float32, FLUID_SIZE)
    local fluid_Qxx_cpu = zeros(Float32, FLUID_SIZE)
    local fluid_Qxy_cpu = zeros(Float32, FLUID_SIZE)


    function update_visualization_data_from_gpu!()
        copyto!(fluid_color_R_cpu, fluid_color_R)
        copyto!(fluid_color_G_cpu, fluid_color_G)
        copyto!(fluid_color_B_cpu, fluid_color_B)

        color_image_data_local_cpu = fluid_color_image_obs[] # Get the underlying Array
        for j_vis in 1:NY
            for i_vis in 1:NX
                idx_sim_grid = IX_cpu(i_vis + 1, j_vis + 1) # Use CPU version of IX
                r = clamp(fluid_color_R_cpu[idx_sim_grid], 0.0f0, 1.0f0)
                g = clamp(fluid_color_G_cpu[idx_sim_grid], 0.0f0, 1.0f0)
                b = clamp(fluid_color_B_cpu[idx_sim_grid], 0.0f0, 1.0f0)
                color_image_data_local_cpu[i_vis, j_vis] = RGB{Float32}(r, g, b)
            end
        end
        fluid_color_image_obs[] = color_image_data_local_cpu # Trigger update

        copyto!(fluid_Qxx_cpu, fluid_Qxx)
        copyto!(fluid_Qxy_cpu, fluid_Qxy)

        num_arrows = length(arrow_pos_obs[])
        new_arrow_dirs_local = Vector{Vec2f}(undef, num_arrows)
        vis_idx = 1
        for j_glob_vis in 1:ARROW_SUBSAMPLE:NY # Loop over visualization grid points
            for i_glob_vis in 1:ARROW_SUBSAMPLE:NX
                if vis_idx > num_arrows; break; end
                idx_sim_grid = IX_cpu(i_glob_vis + 1, j_glob_vis + 1) # Map to simulation grid index
                qxx_val = fluid_Qxx_cpu[idx_sim_grid]
                qxy_val = fluid_Qxy_cpu[idx_sim_grid]
                
                angle_double = atan(qxy_val, qxx_val) 
                angle_director = 0.5f0 * angle_double
                director_x = cos(angle_director)
                director_y = sin(angle_director)
                
                norm_dir = sqrt(director_x^2 + director_y^2)
                if norm_dir > 1e-7
                    new_arrow_dirs_local[vis_idx] = Vec2f(director_x/norm_dir, director_y/norm_dir)
                else
                    new_arrow_dirs_local[vis_idx] = Vec2f(1.0f0, 0.0f0) 
                end
                vis_idx += 1
            end
            if vis_idx > num_arrows break end
        end
        if num_arrows > 0 && (vis_idx - 1 == num_arrows)
             arrow_dir_obs[] = new_arrow_dirs_local
        elseif num_arrows == 0
             arrow_dir_obs[] = Vec2f[] 
        end
    end

    # Main game loop
    on(events(fig).tick) do _
        if !game_active[]; return Consume(false); end
        current_time = time(); dt_game = Float32(clamp(current_time - last_update_time[], 0.001, 0.05)); last_update_time[] = current_time
        keys = events(fig).keyboardstate; bp = ball_pos[]; bv = ball_vel[]; 
        p_left_y_val = paddle_left_y[]; p_right_y_val = paddle_right_y[]; prev_ball_pos[] = bp 

        # Paddle movement (CPU)
        paddle_delta = PADDLE_SPEED * dt_game
        if Keyboard.w in keys; paddle_left_y[] = min(COURT_HEIGHT - PADDLE_HEIGHT, p_left_y_val + paddle_delta); end
        if Keyboard.s in keys; paddle_left_y[] = max(0.0f0, p_left_y_val - paddle_delta); end
        if Keyboard.up in keys; paddle_right_y[] = min(COURT_HEIGHT - PADDLE_HEIGHT, p_right_y_val + paddle_delta); end
        if Keyboard.down in keys; paddle_right_y[] = max(0.0f0, p_right_y_val - paddle_delta); end
        
        # Update local paddle y values after clamp
        p_left_y_val = paddle_left_y[]; p_right_y_val = paddle_right_y[];

        direct_pull_force = Vec2f(0.0f0) # CPU Vec
        is_p1_pushing_this_frame = (Keyboard.d in keys || (Keyboard.space in keys && served_by[] == :p1)) && serve_state[] == :playing
        is_p2_pushing_this_frame = (Keyboard.left in keys || (Keyboard.enter in keys && served_by[] == :p2)) && serve_state[] == :playing

        # Pull force logic (CPU)
        if Keyboard.a in keys && serve_state[] == :playing 
            paddle_center_y = p_left_y_val + PADDLE_HEIGHT / 2f0; paddle_center_world = Point2f(PADDLE_WIDTH / 2f0, paddle_center_y)
            dir_to_paddle = paddle_center_world - bp; dist_sq = sum(dir_to_paddle .^ 2)
            if dist_sq > 1e-4; direct_pull_force += normalize(dir_to_paddle) * PULL_FORCE_STRENGTH; end
        end
        if Keyboard.right in keys && serve_state[] == :playing 
            paddle_center_y = p_right_y_val + PADDLE_HEIGHT / 2f0; paddle_center_world = Point2f(COURT_WIDTH - PADDLE_WIDTH / 2f0, paddle_center_y)
            dir_to_paddle = paddle_center_world - bp; dist_sq = sum(dir_to_paddle .^ 2)
            if dist_sq > 1e-4; direct_pull_force += normalize(dir_to_paddle) * PULL_FORCE_STRENGTH; end
        end

        fluid_time_accumulator[] += dt_game
        steps_taken = 0; max_steps_per_frame = 5 
        while fluid_time_accumulator[] >= DT && steps_taken < max_steps_per_frame
            if is_p1_pushing_this_frame
                apply_paddle_jet_effect_gpu!(1, p_left_y_val); 
            end
            if is_p2_pushing_this_frame
                apply_paddle_jet_effect_gpu!(2, p_right_y_val); 
            end

            fluid_step_gpu!(is_p1_pushing_this_frame, p_left_y_val, is_p2_pushing_this_frame, p_right_y_val)
            
            fluid_time_accumulator[] -= DT
            steps_taken += 1
        end
        if steps_taken > 0; CUDA.synchronize(); update_visualization_data_from_gpu!(); end # Synchronize before vis data copy

        # Ball physics (CPU)
        if serve_state[] == :playing
            total_force_on_ball = Vec2f(0.0f0)
            
            copyto!(fluid_vx_cpu, fluid_vx)
            copyto!(fluid_vy_cpu, fluid_vy)

            fluid_vel_at_ball = get_fluid_velocity_at_cpu(bp, fluid_vx_cpu, fluid_vy_cpu)
            relative_vel = fluid_vel_at_ball - bv
            drag_force = BALL_DRAG_COEFF * relative_vel
            total_force_on_ball += drag_force
            total_force_on_ball += direct_pull_force

            new_bv = bv + total_force_on_ball * dt_game 
            new_bp = bp + new_bv * dt_game

            # Wall collision
            if new_bp[2] < BALL_SIZE/2f0 || new_bp[2] > COURT_HEIGHT-BALL_SIZE/2f0
                new_bp = Point2f(new_bp[1], clamp(new_bp[2], BALL_SIZE/2f0+0.1f0, COURT_HEIGHT-BALL_SIZE/2f0-0.1f0))
                new_bv = Vec2f(new_bv[1], -new_bv[2]) 
            end

            # Paddle collision
            paddle_left_rect=Rect2f(0f0, p_left_y_val, PADDLE_WIDTH, PADDLE_HEIGHT)
            paddle_right_rect=Rect2f(COURT_WIDTH-PADDLE_WIDTH, p_right_y_val, PADDLE_WIDTH, PADDLE_HEIGHT)
            ball_rect_predict = Rect2f(new_bp[1]-BALL_SIZE/2f0, new_bp[2]-BALL_SIZE/2f0, BALL_SIZE, BALL_SIZE)
            
            cbs_val = current_ball_speed[] # Get value from observable

            if new_bv[1]<0f0 && rect_overlaps(paddle_left_rect, ball_rect_predict)
                paddle_center_y_val=p_left_y_val+PADDLE_HEIGHT/2f0
                hit_offset=clamp((new_bp[2]-paddle_center_y_val)/(PADDLE_HEIGHT/2f0), -1.0f0, 1.0f0)
                bounce_angle=hit_offset*(pi/3f0) 
                cbs_val += BALL_SPEED_INCREASE
                new_bv=Vec2f(cos(bounce_angle), sin(bounce_angle))*cbs_val
                new_bp=Point2f(PADDLE_WIDTH+BALL_SIZE/2f0+0.1f0, new_bp[2])
            elseif new_bv[1]>0f0 && rect_overlaps(paddle_right_rect, ball_rect_predict)
                paddle_center_y_val=p_right_y_val+PADDLE_HEIGHT/2f0
                hit_offset=clamp((new_bp[2]-paddle_center_y_val)/(PADDLE_HEIGHT/2f0), -1.0f0, 1.0f0)
                bounce_angle=hit_offset*(pi/3f0)
                cbs_val += BALL_SPEED_INCREASE
                world_angle=pi-bounce_angle 
                new_bv=Vec2f(cos(world_angle), sin(world_angle))*cbs_val
                new_bp=Point2f(COURT_WIDTH-PADDLE_WIDTH-BALL_SIZE/2f0-0.1f0, new_bp[2])
            end
            ball_pos[] = new_bp; ball_vel[] = new_bv; current_ball_speed[] = norm(new_bv) 

            # Scoring
            scored = false; final_pos = ball_pos[] 
            if final_pos[1] < BALL_SIZE/2f0; score_right[]+=1; served_by[]=:p1; scored=true; end
            if final_pos[1] > COURT_WIDTH-BALL_SIZE/2f0; score_left[]+=1; served_by[]=:p2; scored=true; end
            if scored
                if score_left[] >= SCORE_LIMIT || score_right[] >= SCORE_LIMIT
                    game_active[] = false; winner = score_left[] >= SCORE_LIMIT ? "Left" : "Right"; game_message[] = "$winner Player Wins!\nPress R to Restart"
                else; reset_ball(); end
            end
        # Ball serving position update
        elseif serve_state[] == :p1_serve
            target_y=p_left_y_val+PADDLE_HEIGHT/2f0; new_pos=Point2f(bp[1], clamp(target_y, BALL_SIZE/2f0, COURT_HEIGHT-BALL_SIZE/2f0)); ball_pos[]=new_pos; prev_ball_pos[]=new_pos
        elseif serve_state[] == :p2_serve
            target_y=p_right_y_val+PADDLE_HEIGHT/2f0; new_pos=Point2f(bp[1], clamp(target_y, BALL_SIZE/2f0, COURT_HEIGHT-BALL_SIZE/2f0)); ball_pos[]=new_pos; prev_ball_pos[]=new_pos
        end
        return Consume(false)
    end

    # Keyboard input handling (CPU)
    on(events(fig).keyboardbutton) do event
        is_serve_key_p1 = event.key == Keyboard.space || event.key == Keyboard.d
        is_serve_key_p2 = event.key == Keyboard.enter || event.key == Keyboard.left
        current_state = serve_state[]

        if game_active[] && event.action == Keyboard.press && current_state != :playing
            serve_triggered = false; base_angle = 0f0
            if current_state == :p1_serve && is_serve_key_p1; serve_triggered=true; base_angle=0f0; served_by[]=:p1; end
            if current_state == :p2_serve && is_serve_key_p2; serve_triggered=true; base_angle=pi; served_by[]=:p2; end

            if serve_triggered
                angle_offset = (rand(Float32) * (pi/3f0)) - (pi/6f0) 
                angle=base_angle + angle_offset
                serve_speed=current_ball_speed[] 
                ball_vel[]=Vec2f(cos(angle), sin(angle))*serve_speed
                serve_state[] = :playing; game_message[] = ""; return Consume(true)
            end
        end
        if !game_active[] && event.key == Keyboard.r && event.action == Keyboard.press # Restart game
            score_left[]=0; score_right[]=0; served_by[]=rand([:p1, :p2]); reset_ball(); game_active[]=true
            
            # Reset fluid state on GPU
            CUDA.fill!(fluid_vx, 0.0f0); CUDA.fill!(fluid_vy, 0.0f0); CUDA.fill!(fluid_vx0, 0.0f0); CUDA.fill!(fluid_vy0, 0.0f0)
            CUDA.fill!(fluid_p, 0.0f0); CUDA.fill!(fluid_div, 0.0f0)
            rand_scale_init=0.01f0
            
            rand_gpu_1_rst = CUDA.rand(Float32, FLUID_SIZE) # New random numbers
            rand_gpu_2_rst = CUDA.rand(Float32, FLUID_SIZE)
            @cuda threads=threads_1d_init blocks=blocks_1d_init init_Q_kernel!(fluid_Qxx, rand_gpu_1_rst, rand_scale_init, FLUID_SIZE)
            @cuda threads=threads_1d_init blocks=blocks_1d_init init_Q_kernel!(fluid_Qxy, rand_gpu_2_rst, rand_scale_init, FLUID_SIZE)
            # CUDA.synchronize()
            set_boundary_scalar_gpu!(fluid_Qxx); set_boundary_scalar_gpu!(fluid_Qxy)
            
            CUDA.fill!(fluid_Hxx,0.0f0); CUDA.fill!(fluid_Hxy,0.0f0)
            CUDA.fill!(fluid_div_sigma_active_x,0.0f0); CUDA.fill!(fluid_div_sigma_active_y,0.0f0)
            CUDA.fill!(fluid_div_sigma_elastic_x,0.0f0); CUDA.fill!(fluid_div_sigma_elastic_y,0.0f0)
            CUDA.fill!(fluid_color_R, 0.0f0); CUDA.fill!(fluid_color_G, 0.0f0); CUDA.fill!(fluid_color_B, 0.0f0)
            CUDA.fill!(fluid_color_R0, 0.0f0); CUDA.fill!(fluid_color_G0, 0.0f0); CUDA.fill!(fluid_color_B0, 0.0f0)
            set_boundary_scalar_gpu!(fluid_color_R); set_boundary_scalar_gpu!(fluid_color_G); set_boundary_scalar_gpu!(fluid_color_B)
            
            CUDA.synchronize() # Ensure all GPU resets are done
            update_visualization_data_from_gpu!(); return Consume(true)
        end
        return Consume(false)
    end
    
    update_visualization_data_from_gpu!() # Initial draw
    display(fig); return fig
end

run()