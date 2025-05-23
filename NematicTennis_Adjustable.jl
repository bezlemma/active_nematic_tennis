using GLMakie
using Random
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Colors

# --- Constants for Green Color from Activity ---
const ACTIVITY_GREEN_DIV_STRESS_SQ_THRESHOLD = 0.02f0 # Squared magnitude of div(active_stress) to trigger green
const ACTIVITY_GREEN_SCALAR_ADD = 0.0f0             # Amount of green to add if threshold met (currently off)

# --- Game Constants ---
const PADDLE_SPEED = 350.0f0
const BALL_SPEED_INIT = 250.0f0
const BALL_SPEED_INCREASE = 20.0f0
const PADDLE_HEIGHT = 100.0f0
const PADDLE_WIDTH = 20.0f0
const BALL_SIZE = 20.0f0
const COURT_WIDTH = 800.0f0
const COURT_HEIGHT = 600.0f0
const SCORE_LIMIT = 5

# --- Fluid Simulation Setup ---
const NX = 160 # Resolution in X
const NY = 120 # Resolution in Y
const DT = 0.0001f0 # Time step for Q evolution & fluid solver
const SOLVER_ITER = 20 # Iterations for linear solver

# --- Active Nematic Parameters ---
const INITIAL_VISCOSITY = 0.01f0       # For viscosity_obs (μ)
const λ = 0.7f0                        # Flow alignment λ (fixed)
const γ = 1.0f0                        # Rotational/Relaxation viscosity γ
const INITIAL_ACTIVITY = 0.6f0       # For activity_obs (α)
const LANDAU_DEGENNES_A = 1.0f0        # Landau-de Gennes coefficient A
const LANDAU_DEGENNES_C = 1.0f0        # Landau-de Gennes coefficient C
const FRANK_ELASTIC_K_INITIAL = 0.01f0 # For elasticity_obs (K)

# --- Observables for Controllable Parameters ---
global activity_obs = Observable(INITIAL_ACTIVITY)
global viscosity_obs = Observable(INITIAL_VISCOSITY)
global elasticity_obs = Observable(FRANK_ELASTIC_K_INITIAL)

# --- Paddle to game constants ---
const PADDLE_JET_FORCE_STRENGTH = 0.2f0 # Strength of the body force applied by the paddle jet
const JET_CONE_LENGTH_CELLS = 10      # How many cells deep the direct paddle influence is
const JET_MAX_WIDTH_FRAC = 0.5f0        # Fraction of paddle height for max jet width at cone tip
const BALL_DRAG_COEFF = 0.2f0
const PULL_FORCE_STRENGTH = 1000.0f0

# --- Fluid Grid Constants ---
const FLUID_DX = COURT_WIDTH / NX       
const FLUID_DY = COURT_HEIGHT / NY      
const FLUID_SIZE = (NX + 2) * (NY + 2)
const INV_DX = 1.0f0 / FLUID_DX
const INV_DY = 1.0f0 / FLUID_DY
const INV_DX2 = INV_DX * INV_DX
const INV_DY2 = INV_DY * INV_DY

# --- Visualization Constants ---
const ARROW_SUBSAMPLE = 1
const ARROW_LENGTH_SCALE = 2.0f0 * min(FLUID_DX, FLUID_DY) * ARROW_SUBSAMPLE

# --- Global Fluid Arrays ---
global fluid_vx::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_vy::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_vx0::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Stores sum of forces for vx
global fluid_vy0::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Stores sum of forces for vy
global fluid_p::Vector{Float32} = zeros(Float32, FLUID_SIZE)   # Pressure field
global fluid_div::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Divergence of velocity field

global fluid_Qxx::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_Qxy::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_Qxx0::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Previous Qxx for advection
global fluid_Qxy0::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Previous Qxy for advection

global fluid_Hxx::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Molecular field Hxx
global fluid_Hxy::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Molecular field Hxy

global fluid_div_sigma_active_x::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_active_y::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_elastic_x::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_elastic_y::Vector{Float32} = zeros(Float32, FLUID_SIZE)

global Sxx_flow_temp::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Temporary storage for S_flow term
global Sxy_flow_temp::Vector{Float32} = zeros(Float32, FLUID_SIZE) # Temporary storage for S_flow term

# Color Tracer Fields
global fluid_color_R::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_color_G::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_color_B::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_color_R0::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_color_G0::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_color_B0::Vector{Float32} = zeros(Float32, FLUID_SIZE)

# Observables for Visualization
global fluid_color_image_obs = Observable(zeros(RGB{Float32}, NX, NY))
global arrow_pos_obs = Observable(Point2f[])
global arrow_dir_obs = Observable(Vec2f[])


@inline IX(i, j) = clamp(i, 1, NX + 2) + (clamp(j, 1, NY + 2) - 1) * (NX + 2) # Map 2D grid indices to 1D

function set_boundary_velocity!(b::Int, x::Vector{Float32})
    N = NX; M = NY
    # Implements free-slip boundary conditions.
    # Normal velocity component = 0 (v_ghost = -v_interior).
    # Tangential velocity component has zero normal gradient (v_ghost = v_interior).
    # 'b' is 1 for vx, 2 for vy.

    # Horizontal boundaries
    for i in 1:(N+2)
        x[IX(i,1)]   = (b == 1) ? x[IX(i,2)]   : -x[IX(i,2)]
        x[IX(i,M+2)] = (b == 1) ? x[IX(i,M+1)] : -x[IX(i,M+1)]
    end
    # Vertical boundaries
    for j in 1:(M+2)
        x[IX(1,j)]   = (b == 2) ? x[IX(2,j)]   : -x[IX(2,j)]
        x[IX(N+2,j)] = (b == 2) ? x[IX(N+1,j)] : -x[IX(N+1,j)]
    end

    x[IX(1,1)]       = 0.5f0*(x[IX(1,2)]+x[IX(2,1)])
    x[IX(1,M+2)]     = 0.5f0*(x[IX(1,M+1)]+x[IX(2,M+2)])
    x[IX(N+2,1)]     = 0.5f0*(x[IX(N+2,2)]+x[IX(N+1,1)])
    x[IX(N+2,M+2)]   = 0.5f0*(x[IX(N+2,M+1)]+x[IX(N+1,M+2)])
end

function set_boundary_scalar!(x::Vector{Float32})
    N = NX; M = NY
    # Neumann boundary conditions (zero gradient normal to the boundary).
    for i in 1:(N+2); x[IX(i, 1)] = x[IX(i, 2)]; x[IX(i, M + 2)] = x[IX(i, M + 1)]; end
    for j in 1:(M+2); x[IX(1, j)] = x[IX(2, j)]; x[IX(N + 2, j)] = x[IX(N + 1, j)]; end
    
    x[IX(1, 1)]       = 0.5f0 * (x[IX(2, 1)] + x[IX(1, 2)])
    x[IX(1, M + 2)]   = 0.5f0 * (x[IX(2, M + 2)] + x[IX(1, M + 1)])
    x[IX(N + 2, 1)]   = 0.5f0 * (x[IX(N + 1, 1)] + x[IX(N + 2, 2)])
    x[IX(N + 2, M + 2)] = 0.5f0 * (x[IX(N + 1, M + 2)] + x[IX(N + 2, M + 1)])
end

@inline function laplacian_scalar(f::Vector{Float32}, i::Int, j::Int)
    val_center = f[IX(i,j)]
    return (f[IX(i+1,j)] - 2.0f0*val_center + f[IX(i-1,j)]) * INV_DX2 +
           (f[IX(i,j+1)] - 2.0f0*val_center + f[IX(i,j-1)]) * INV_DY2
end
@inline function grad_x_centered(f::Vector{Float32}, i::Int, j::Int); return (f[IX(i+1,j)] - f[IX(i-1,j)]) * 0.5f0 * INV_DX; end
@inline function grad_y_centered(f::Vector{Float32}, i::Int, j::Int); return (f[IX(i,j+1)] - f[IX(i,j-1)]) * 0.5f0 * INV_DY; end

function general_linear_solve!(b_type::Int, x_out::Vector{Float32}, x0_rhs::Vector{Float32},
                                 coeff_I_term::Float32, coeff_lap_term_dx::Float32, coeff_lap_term_dy::Float32,
                                 iterations::Int)
    # Solves (coeff_I_term * I - (coeff_lap_dx * d²/dx² + coeff_lap_dy * d²/dy²)) * x_out = x0_rhs by Jacobi iteration.
    for _ in 1:iterations
        x_prev_iter = copy(x_out)
        for j in 2:(NY+1), i in 2:(NX+1)
            idx = IX(i,j)
            sum_neighbors_scaled = coeff_lap_term_dx * (x_prev_iter[IX(i-1,j)] + x_prev_iter[IX(i+1,j)]) +
                                 coeff_lap_term_dy * (x_prev_iter[IX(i,j-1)] + x_prev_iter[IX(i,j+1)])

            denominator = coeff_I_term + 2.0f0*coeff_lap_term_dx + 2.0f0*coeff_lap_term_dy
            if abs(denominator) < 1e-9 # Avoid division by zero (1e-9 is Float64)
                x_out[idx] = 0.0f0
            else
                x_out[idx] = (x0_rhs[idx] + sum_neighbors_scaled) / denominator
            end
        end
        b_type == 1 || b_type == 2 ? set_boundary_velocity!(b_type, x_out) : set_boundary_scalar!(x_out)
    end
end


function advect!(b_type::Int, d::Vector{Float32}, d0::Vector{Float32}, velX::Vector{Float32}, velY::Vector{Float32}, dt_adv::Float32)
    # Semi-Lagrangian advection
    for j in 2:(NY+1), i in 2:(NX+1)
        idx = IX(i, j)
        x_particle = Float32(i) - velX[idx] * dt_adv * INV_DX
        y_particle = Float32(j) - velY[idx] * dt_adv * INV_DY

        x_particle = clamp(x_particle, 1.5f0, NX + 0.5f0)
        y_particle = clamp(y_particle, 1.5f0, NY + 0.5f0)

        i0 = floor(Int, x_particle); i1 = i0 + 1
        j0 = floor(Int, y_particle); j1 = j0 + 1

        s1 = x_particle - Float32(i0); s0 = 1.0f0 - s1
        t1 = y_particle - Float32(j0); t0 = 1.0f0 - t1

        d[idx] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                 s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)])
    end
    b_type == 1 || b_type == 2 ? set_boundary_velocity!(b_type, d) : set_boundary_scalar!(d)
end

function project!(velX::Vector{Float32}, velY::Vector{Float32}, p::Vector{Float32}, div_field::Vector{Float32})
    # Enforces incompressibility (∇ ⋅ v = 0)
    for j in 2:(NY+1), i in 2:(NX+1)
        idx = IX(i,j)
        div_field[idx] = (velX[IX(i+1, j)] - velX[IX(i-1, j)]) * 0.5f0 * INV_DX +
                         (velY[IX(i, j+1)] - velY[IX(i, j-1)]) * 0.5f0 * INV_DY
        p[idx] = 0.0f0
    end
    set_boundary_scalar!(div_field); set_boundary_scalar!(p)

    # Solve Poisson equation: ∇²p = div_field
    for _ in 1:SOLVER_ITER
        p_old_iter = copy(p)
        for j in 2:(NY+1), i in 2:(NX+1)
            idx = IX(i,j)
            p[idx] = ((p_old_iter[IX(i+1,j)] + p_old_iter[IX(i-1,j)]) * INV_DX2 +
                      (p_old_iter[IX(i,j+1)] + p_old_iter[IX(i,j-1)]) * INV_DY2 - div_field[idx]) /
                     (2.0f0 * (INV_DX2 + INV_DY2))
        end
        set_boundary_scalar!(p)
    end

    # Correct velocities: v_new = v_old - ∇p
    for j in 2:(NY+1), i in 2:(NX+1)
        idx = IX(i,j)
        velX[idx] -= (p[IX(i+1, j)] - p[IX(i-1, j)]) * 0.5f0 * INV_DX
        velY[idx] -= (p[IX(i, j+1)] - p[IX(i, j-1)]) * 0.5f0 * INV_DY
    end
    set_boundary_velocity!(1, velX); set_boundary_velocity!(2, velY)
end

function calculate_H!(Hxx_out::Vector{Float32}, Hxy_out::Vector{Float32},
                        Qxx_in::Vector{Float32}, Qxy_in::Vector{Float32},
                        A_coeff::Float32, C_coeff::Float32)
    # Calculates molecular field H_ij using current elasticity from observable.
    current_elasticity = elasticity_obs[]
    Threads.@threads for j in 2:(NY+1)
        for i in 2:(NX+1)
            idx = IX(i,j)
            qxx_val = Qxx_in[idx]; qxy_val = Qxy_in[idx]
            lap_qxx = laplacian_scalar(Qxx_in, i, j)
            lap_qxy = laplacian_scalar(Qxy_in, i, j)
            trQ2_val = 2.0f0 * (qxx_val*qxx_val + qxy_val*qxy_val)

            Hxx_out[idx] = -(A_coeff * qxx_val + 2.0f0 * C_coeff * trQ2_val * qxx_val - current_elasticity * lap_qxx)
            Hxy_out[idx] = -(A_coeff * qxy_val + 2.0f0 * C_coeff * trQ2_val * qxy_val - current_elasticity * lap_qxy)
        end
    end
    set_boundary_scalar!(Hxx_out); set_boundary_scalar!(Hxy_out)
end

function calculate_S_flow_term!(Sxx_out::Vector{Float32}, Sxy_out::Vector{Float32},
                                  Qxx_in::Vector{Float32}, Qxy_in::Vector{Float32},
                                  vx::Vector{Float32}, vy::Vector{Float32}, lambda_align::Float32)
    # Calculates S_ij = λ D_ij - [Ω, Q]_ij (flow-coupling term).
    Threads.@threads for j in 2:(NY+1)
        for i in 2:(NX+1)
            idx = IX(i,j)
            qxx = Qxx_in[idx]; qxy = Qxy_in[idx]

            dvx_dx = grad_x_centered(vx,i,j); dvx_dy = grad_y_centered(vx,i,j)
            dvy_dx = grad_x_centered(vy,i,j); dvy_dy = grad_y_centered(vy,i,j)

            Dxx = dvx_dx
            Dxy = 0.5f0 * (dvx_dy + dvy_dx)
            omega_xy_val = 0.5f0 * (dvy_dx - dvx_dy) # Ω_yx component

            comm_Omega_Q_xx = 2.0f0 * omega_xy_val * qxy
            comm_Omega_Q_xy = -2.0f0 * omega_xy_val * qxx

            Sxx_out[idx] = lambda_align * Dxx - comm_Omega_Q_xx
            Sxy_out[idx] = lambda_align * Dxy - comm_Omega_Q_xy
        end
    end
end

function calculate_div_active_stress!(div_sx::Vector{Float32}, div_sy::Vector{Float32},
                                        Qxx_in::Vector{Float32}, Qxy_in::Vector{Float32})
    # Calculates divergence of active stress using current activity from observable.
    current_activity = activity_obs[]
    Threads.@threads for j in 2:(NY+1)
        for i in 2:(NX+1)
            idx = IX(i,j)
            dQxx_dx = grad_x_centered(Qxx_in,i,j); dQxx_dy = grad_y_centered(Qxx_in,i,j)
            dQxy_dx = grad_x_centered(Qxy_in,i,j); dQxy_dy = grad_y_centered(Qxy_in,i,j)

            div_sx[idx] = -current_activity * (dQxx_dx + dQxy_dy)
            div_sy[idx] = -current_activity * (dQxy_dx - dQxx_dy)
        end
    end
end

function calculate_div_elastic_stress!(div_out_x::Vector{Float32}, div_out_y::Vector{Float32},
                                         Qxx_vec::Vector{Float32}, Qxy_vec::Vector{Float32},
                                         Hxx_vec::Vector{Float32}, Hxy_vec::Vector{Float32},
                                         lambda_align::Float32)
    # Calculates divergence of elastic stress using current elasticity from observable.
    current_elasticity = elasticity_obs[]
    sigma_el_xx_temp = zeros(Float32, FLUID_SIZE)
    sigma_el_xy_temp = zeros(Float32, FLUID_SIZE)
    sigma_el_yy_temp = zeros(Float32, FLUID_SIZE)

    Threads.@threads for j_idx in 2:(NY+1) # Terms involving H
        for i_idx in 2:(NX+1)
            idx = IX(i_idx,j_idx)
            q_xx = Qxx_vec[idx]; q_xy = Qxy_vec[idx]
            h_xx = Hxx_vec[idx]; h_xy = Hxy_vec[idx]

            comm_QH_xx_val = 0.0f0
            comm_QH_xy_val = 2.0f0 * (q_xx * h_xy - q_xy * h_xx)

            sigma_el_xx_temp[idx] = -lambda_align * comm_QH_xx_val
            sigma_el_xy_temp[idx] = -lambda_align * comm_QH_xy_val
            sigma_el_yy_temp[idx] = lambda_align * comm_QH_xx_val
        end
    end

    Threads.@threads for j_idx in 2:(NY+1) #  Terms involving K_elastic (current_elasticity)
        for i_idx in 2:(NX+1)
            idx = IX(i_idx, j_idx)
            dqxx_dx = grad_x_centered(Qxx_vec, i_idx, j_idx)
            dqxy_dx = grad_x_centered(Qxy_vec, i_idx, j_idx)
            dqxx_dy = grad_y_centered(Qxx_vec, i_idx, j_idx)
            dqxy_dy = grad_y_centered(Qxy_vec, i_idx, j_idx)

            sigma_K_xx_val = -current_elasticity * 2.0f0 * (dqxx_dx^2 + dqxy_dx^2)
            sigma_K_xy_val = -current_elasticity * 2.0f0 * (dqxx_dx * dqxx_dy + dqxy_dx * dqxy_dy)
            sigma_K_yy_val = -current_elasticity * 2.0f0 * (dqxx_dy^2 + dqxy_dy^2)

            sigma_el_xx_temp[idx] += sigma_K_xx_val
            sigma_el_xy_temp[idx] += sigma_K_xy_val
            sigma_el_yy_temp[idx] += sigma_K_yy_val
        end
    end
    set_boundary_scalar!(sigma_el_xx_temp)
    set_boundary_scalar!(sigma_el_xy_temp)
    set_boundary_scalar!(sigma_el_yy_temp)

    Threads.@threads for j_idx in 2:(NY+1) # Divergence of total elastic stress
        for i_idx in 2:(NX+1)
            idx = IX(i_idx, j_idx)
            div_out_x[idx] = grad_x_centered(sigma_el_xx_temp, i_idx, j_idx) + grad_y_centered(sigma_el_xy_temp, i_idx, j_idx)
            div_out_y[idx] = grad_x_centered(sigma_el_xy_temp, i_idx, j_idx) + grad_y_centered(sigma_el_yy_temp, i_idx, j_idx)
        end
    end
end

function _add_jet_force_in_cone!(
    force_vx_array::Vector{Float32},
    paddle_id::Int,
    paddle_base_y_world::Float32,
    force_strength::Float32
)
    paddle_center_y_world = paddle_base_y_world + PADDLE_HEIGHT / 2.0f0
    _, paddle_center_gy_float = world_to_grid(Point2f(0.0f0, paddle_center_y_world))
    paddle_center_gy_idx = clamp(round(Int, paddle_center_gy_float), 2, NY + 1)

    jet_force_x_actual = (paddle_id == 1) ? force_strength : -force_strength

    paddle_height_cells = PADDLE_HEIGHT / FLUID_DY
    max_jet_width_at_tip_cells = max(1, round(Int, paddle_height_cells * JET_MAX_WIDTH_FRAC))

    start_x_world_edge = (paddle_id == 1) ? PADDLE_WIDTH : (COURT_WIDTH - PADDLE_WIDTH)
    start_gxi_jet_float = if paddle_id == 1
        world_to_grid(Point2f(start_x_world_edge + FLUID_DX * 0.5f0, 0.0f0))[1]
    else
        world_to_grid(Point2f(start_x_world_edge - FLUID_DX * 0.5f0, 0.0f0))[1]
    end
    start_gxi_jet_idx = clamp(round(Int, start_gxi_jet_float), 2, NX + 1)

    base_half_width_cells = max(1, round(Int, paddle_height_cells / 6.0f0))
    tip_half_width_cells = max(1, round(Int, max_jet_width_at_tip_cells / 2.0f0))

    for i_dist_cells in 0:(JET_CONE_LENGTH_CELLS - 1)
        current_gxi = if paddle_id == 1
            start_gxi_jet_idx + i_dist_cells
        else
            start_gxi_jet_idx - i_dist_cells
        end

        if !(2 <= current_gxi <= NX + 1) continue end

        width_frac = JET_CONE_LENGTH_CELLS <= 1 ? 1.0f0 : Float32(i_dist_cells) / max(1.0f0, Float32(JET_CONE_LENGTH_CELLS - 1))
        current_jet_half_width_cells = round(Int, base_half_width_cells + (tip_half_width_cells - base_half_width_cells) * width_frac)
        current_jet_half_width_cells = max(1, current_jet_half_width_cells)

        for gy_offset_loop in -(current_jet_half_width_cells-1):(current_jet_half_width_cells-1)
            current_gyi = clamp(paddle_center_gy_idx + gy_offset_loop, 2, NY + 1)
            idx_cone = IX(current_gxi, current_gyi)
            force_vx_array[idx_cone] += jet_force_x_actual
        end
    end
end


function fluid_step!(p1_is_pushing::Bool, p1_y_world::Float32, p2_is_pushing::Bool, p2_y_world::Float32)
    u_old_vx_for_advection = copy(fluid_vx)
    u_old_vy_for_advection = copy(fluid_vy)

    fluid_Qxx0 .= fluid_Qxx; fluid_Qxy0 .= fluid_Qxy
    advect!(0, fluid_Qxx, fluid_Qxx0, u_old_vx_for_advection, u_old_vy_for_advection, DT)
    advect!(0, fluid_Qxy, fluid_Qxy0, u_old_vx_for_advection, u_old_vy_for_advection, DT)

    fluid_color_R0 .= fluid_color_R; fluid_color_G0 .= fluid_color_G; fluid_color_B0 .= fluid_color_B
    advect!(0, fluid_color_R, fluid_color_R0, u_old_vx_for_advection, u_old_vy_for_advection, DT)
    advect!(0, fluid_color_G, fluid_color_G0, u_old_vx_for_advection, u_old_vy_for_advection, DT)
    advect!(0, fluid_color_B, fluid_color_B0, u_old_vx_for_advection, u_old_vy_for_advection, DT)

    calculate_H!(fluid_Hxx, fluid_Hxy, fluid_Qxx, fluid_Qxy, LANDAU_DEGENNES_A, LANDAU_DEGENNES_C)
    calculate_S_flow_term!(Sxx_flow_temp, Sxy_flow_temp, fluid_Qxx, fluid_Qxy,
                           u_old_vx_for_advection, u_old_vy_for_advection, λ)

    Threads.@threads for i in 1:FLUID_SIZE
        fluid_Qxx[i] += DT * (Sxx_flow_temp[i] + γ * fluid_Hxx[i])
        fluid_Qxy[i] += DT * (Sxy_flow_temp[i] + γ * fluid_Hxy[i])
    end
    set_boundary_scalar!(fluid_Qxx); set_boundary_scalar!(fluid_Qxy)

    calculate_H!(fluid_Hxx, fluid_Hxy, fluid_Qxx, fluid_Qxy, LANDAU_DEGENNES_A, LANDAU_DEGENNES_C)

    calculate_div_active_stress!(fluid_div_sigma_active_x, fluid_div_sigma_active_y, fluid_Qxx, fluid_Qxy)
    calculate_div_elastic_stress!(fluid_div_sigma_elastic_x, fluid_div_sigma_elastic_y,
                                  fluid_Qxx, fluid_Qxy, fluid_Hxx, fluid_Hxy, λ)

    Threads.@threads for i in 1:FLUID_SIZE
        active_force_sq_mag = fluid_div_sigma_active_x[i]^2 + fluid_div_sigma_active_y[i]^2
        if active_force_sq_mag > ACTIVITY_GREEN_DIV_STRESS_SQ_THRESHOLD
            fluid_color_G[i] = min(1.0f0, fluid_color_G[i] + ACTIVITY_GREEN_SCALAR_ADD * DT)
        end
    end

    Threads.@threads for i in 1:FLUID_SIZE
        fluid_vx0[i] = fluid_div_sigma_active_x[i] + fluid_div_sigma_elastic_x[i]
        fluid_vy0[i] = fluid_div_sigma_active_y[i] + fluid_div_sigma_elastic_y[i]
    end

    if p1_is_pushing; _add_jet_force_in_cone!(fluid_vx0, 1, p1_y_world, PADDLE_JET_FORCE_STRENGTH); end
    if p2_is_pushing; _add_jet_force_in_cone!(fluid_vx0, 2, p2_y_world, PADDLE_JET_FORCE_STRENGTH); end

    current_viscosity = viscosity_obs[] # Get current value from observable
    general_linear_solve!(1, fluid_vx, fluid_vx0,
                          0.0f0, current_viscosity * INV_DX2, current_viscosity * INV_DY2, # Use 0 for ζ
                          SOLVER_ITER)
    general_linear_solve!(2, fluid_vy, fluid_vy0,
                          0.0f0, current_viscosity * INV_DX2, current_viscosity * INV_DY2, # Use 0 for ζ
                          SOLVER_ITER)

    project!(fluid_vx, fluid_vy, fluid_p, fluid_div)
end


function world_to_grid(pos::Point2f)
    gx = (pos[1] / COURT_WIDTH) * NX + 1.5f0;
    gy = (pos[2] / COURT_HEIGHT) * NY + 1.5f0;
    return gx, gy
end

function get_fluid_velocity_at(pos::Point2f)::Vec2f
    gx, gy = world_to_grid(pos)
    gx = clamp(gx, 1.5f0, NX + 0.5f0)
    gy = clamp(gy, 1.5f0, NY + 0.5f0)

    i0 = floor(Int, gx); i1 = i0 + 1
    j0 = floor(Int, gy); j1 = j0 + 1

    s1 = gx - Float32(i0); s0 = 1.0f0 - s1
    t1 = gy - Float32(j0); t0 = 1.0f0 - t1

    vx = s0*(t0*fluid_vx[IX(i0, j0)] + t1*fluid_vx[IX(i0, j1)]) + s1*(t0*fluid_vx[IX(i1, j0)] + t1*fluid_vx[IX(i1, j1)])
    vy = s0*(t0*fluid_vy[IX(i0, j0)] + t1*fluid_vy[IX(i0, j1)]) + s1*(t0*fluid_vy[IX(i1, j0)] + t1*fluid_vy[IX(i1, j1)])
    return Vec2f(vx, vy) # vel_scale removed
end

function rect_overlaps(r1::Rect2f, r2::Rect2f)
    x_overlap = (r1.origin[1] < r2.origin[1] + r2.widths[1]) && (r1.origin[1] + r1.widths[1] > r2.origin[1])
    y_overlap = (r1.origin[2] < r2.origin[2] + r2.widths[2]) && (r1.origin[2] + r1.widths[2] > r2.origin[2])
    return x_overlap && y_overlap
end

function apply_paddle_jet_color_effect!(
    paddle_id::Int,
    paddle_base_y_world::Float32
)
    paddle_center_y_world = paddle_base_y_world + PADDLE_HEIGHT / 2.0f0
    _, paddle_center_gy_float = world_to_grid(Point2f(0.0f0, paddle_center_y_world))
    paddle_center_gy_idx = clamp(round(Int, paddle_center_gy_float), 2, NY + 1)

    paddle_height_cells = PADDLE_HEIGHT / FLUID_DY
    max_jet_width_at_tip_cells = max(1, round(Int, paddle_height_cells * JET_MAX_WIDTH_FRAC))

    start_x_world_edge = (paddle_id == 1) ? PADDLE_WIDTH : (COURT_WIDTH - PADDLE_WIDTH)

    start_gxi_jet_float = if paddle_id == 1
        world_to_grid(Point2f(start_x_world_edge + FLUID_DX * 0.5f0, 0.0f0))[1]
    else
        world_to_grid(Point2f(start_x_world_edge - FLUID_DX * 0.5f0, 0.0f0))[1]
    end
    start_gxi_jet_idx = clamp(round(Int, start_gxi_jet_float), 2, NX + 1)

    base_half_width_cells = max(1, round(Int, paddle_height_cells / 6.0f0))
    tip_half_width_cells = max(1, round(Int, max_jet_width_at_tip_cells / 2.0f0))

    for i_dist_cells in 0:(JET_CONE_LENGTH_CELLS - 1)
        current_gxi = if paddle_id == 1
            start_gxi_jet_idx + i_dist_cells
        else
            start_gxi_jet_idx - i_dist_cells
        end

        if !(2 <= current_gxi <= NX + 1) continue end

        width_frac = JET_CONE_LENGTH_CELLS <= 1 ? 1.0f0 : Float32(i_dist_cells) / max(1.0f0, Float32(JET_CONE_LENGTH_CELLS - 1))
        current_jet_half_width_cells = round(Int, base_half_width_cells + (tip_half_width_cells - base_half_width_cells) * width_frac)
        current_jet_half_width_cells = max(1, current_jet_half_width_cells)

        for gy_offset_loop in -(current_jet_half_width_cells-1):(current_jet_half_width_cells-1)
            current_gyi = clamp(paddle_center_gy_idx + gy_offset_loop, 2, NY + 1)
            idx_cone = IX(current_gxi, current_gyi)

            if paddle_id == 1
                fluid_color_R[idx_cone] = 1.0f0; fluid_color_G[idx_cone] = 0.0f0; fluid_color_B[idx_cone] = 0.0f0
            else
                fluid_color_R[idx_cone] = 0.0f0; fluid_color_G[idx_cone] = 0.0f0; fluid_color_B[idx_cone] = 1.0f0
            end
            # Q-tensor setting lines were commented out in the provided code.
        end
    end
end


function run()
    slider_panel_width = 300
    fig_width = COURT_WIDTH + slider_panel_width + 20
    fig = Figure(size=(fig_width, COURT_HEIGHT), backgroundcolor=:gainsboro, figure_padding=5)

    game_gl = fig[1, 1] = GridLayout()
    ax = Axis(game_gl[1, 1], aspect=DataAspect(), limits=(0, COURT_WIDTH, 0, COURT_HEIGHT), backgroundcolor=:dimgrey)
    hidedecorations!(ax); hidespines!(ax)

    slider_gl = fig[1, 2] = GridLayout(width=slider_panel_width, tellheight=false)
    Label(slider_gl[1,1], "Fluid Controls", fontsize=20, font=:bold, halign=:center, padding=(0,0,10,0))

    slider_colsizes_setting = (Auto(), Relative(1.0), Auto())
    slider_colgaps_setting = 10

    sg_activity = SliderGrid(slider_gl[2,1], (label = "Activity (α)", range = 0.0f0:0.05f0:1.0f0, format = "{:.2f}", startvalue = activity_obs[]))
    colsize!(sg_activity.layout, 1, slider_colsizes_setting[1]); colsize!(sg_activity.layout, 2, slider_colsizes_setting[2]); colsize!(sg_activity.layout, 3, slider_colsizes_setting[3])
    colgap!(sg_activity.layout, 1, slider_colgaps_setting); colgap!(sg_activity.layout, 2, slider_colgaps_setting)
    on(sg_activity.sliders[1].value) do val; activity_obs[] = val; end

    sg_viscosity = SliderGrid(slider_gl[3,1], (label = "Viscosity (μ)", range = 0.001f0:0.0005f0:0.05f0, format = "{:.4f}", startvalue = viscosity_obs[]))
    colsize!(sg_viscosity.layout, 1, slider_colsizes_setting[1]); colsize!(sg_viscosity.layout, 2, slider_colsizes_setting[2]); colsize!(sg_viscosity.layout, 3, slider_colsizes_setting[3])
    colgap!(sg_viscosity.layout, 1, slider_colgaps_setting); colgap!(sg_viscosity.layout, 2, slider_colgaps_setting)
    on(sg_viscosity.sliders[1].value) do val; viscosity_obs[] = val; end

    sg_elasticity = SliderGrid(slider_gl[4,1], (label = "Elasticity (K)", range = 0.0001f0:0.0005f0:0.05f0, format = "{:.4f}", startvalue = elasticity_obs[]))
    colsize!(sg_elasticity.layout, 1, slider_colsizes_setting[1]); colsize!(sg_elasticity.layout, 2, slider_colsizes_setting[2]); colsize!(sg_elasticity.layout, 3, slider_colsizes_setting[3])
    colgap!(sg_elasticity.layout, 1, slider_colgaps_setting); colgap!(sg_elasticity.layout, 2, slider_colgaps_setting)
    on(sg_elasticity.sliders[1].value) do val; elasticity_obs[] = val; end

    # Initialize fluid arrays
    fill!.((fluid_vx, fluid_vy, fluid_vx0, fluid_vy0, fluid_p, fluid_div,
            Sxx_flow_temp, Sxy_flow_temp, fluid_Qxx0, fluid_Qxy0,
            fluid_Hxx, fluid_Hxy, fluid_div_sigma_active_x, fluid_div_sigma_active_y,
            fluid_div_sigma_elastic_x, fluid_div_sigma_elastic_y,
            fluid_color_R, fluid_color_G, fluid_color_B,
            fluid_color_R0, fluid_color_G0, fluid_color_B0), 0.0f0)

    rand_scale = 0.01f0
    for i in eachindex(fluid_Qxx) # Initial small random Q-tensor values
        fluid_Qxx[i] = (rand(Float32) - 0.5f0) * 2.0f0 * rand_scale
        fluid_Qxy[i] = (rand(Float32) - 0.5f0) * 2.0f0 * rand_scale
    end
    set_boundary_scalar!(fluid_Qxx); set_boundary_scalar!(fluid_Qxy)
    set_boundary_scalar!(fluid_color_R); set_boundary_scalar!(fluid_color_G); set_boundary_scalar!(fluid_color_B)


    arrow_positions = Point2f[];
    fluid_x_range_vis = LinRange(0.0f0 + FLUID_DX/2.0f0, COURT_WIDTH - FLUID_DX/2.0f0, NX);
    fluid_y_range_vis = LinRange(0.0f0 + FLUID_DY/2.0f0, COURT_HEIGHT - FLUID_DY/2.0f0, NY);
    for j in 1:ARROW_SUBSAMPLE:NY, i in 1:ARROW_SUBSAMPLE:NX
        push!(arrow_positions, Point2f(fluid_x_range_vis[i], fluid_y_range_vis[j]));
    end
    arrow_pos_obs[] = arrow_positions
    arrow_dir_obs[] = fill(Vec2f(1.0f0,0.0f0), length(arrow_positions))


    ball_pos = Observable(Point2f(COURT_WIDTH / 2.0f0, COURT_HEIGHT / 2.0f0))
    prev_ball_pos = Observable(Point2f(COURT_WIDTH / 2.0f0, COURT_HEIGHT / 2.0f0))
    ball_vel = Observable(Vec2f(0.0f0, 0.0f0))
    current_ball_speed = Observable(BALL_SPEED_INIT)
    paddle_left_y = Observable(COURT_HEIGHT / 2.0f0 - PADDLE_HEIGHT / 2.0f0)
    paddle_right_y = Observable(COURT_HEIGHT / 2.0f0 - PADDLE_HEIGHT / 2.0f0)
    score_left = Observable(0); score_right = Observable(0); game_active = Observable(true)
    game_message = Observable("Press Serve Keys to Start"); serve_state = Observable(:p1_serve); served_by = Observable(:p1)
    last_update_time = Ref(time()); fluid_time_accumulator = Ref(0.0f0)

    image!(ax, (0.0f0, COURT_WIDTH), (0.0f0, COURT_HEIGHT), fluid_color_image_obs, interpolate=true)
    arrows!(ax, arrow_pos_obs, arrow_dir_obs, arrowsize=Vec2f(ARROW_LENGTH_SCALE*0.25f0, ARROW_LENGTH_SCALE*0.35f0), lengthscale=ARROW_LENGTH_SCALE, arrowcolor=:white, linecolor=:white, linewidth=0.5f0)

    ball_visual_size = BALL_SIZE * 0.9f0
    ball_rect_obs = @lift Rect2f($ball_pos[1]-ball_visual_size/2.0f0, $ball_pos[2]-ball_visual_size/2.0f0, ball_visual_size, ball_visual_size)
    poly!(ax, ball_rect_obs, color=:yellow, strokecolor=:orange, strokewidth=2)
    poly!(ax, @lift(Rect2f(0.0f0, $paddle_left_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color=:lightcyan)
    poly!(ax, @lift(Rect2f(COURT_WIDTH - PADDLE_WIDTH, $paddle_right_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color=:lightcyan)
    score_text_obs = @lift "$($score_left) - $($score_right)"
    text!(ax, score_text_obs, position=Point2f(COURT_WIDTH/2.0f0, COURT_HEIGHT-20.0f0), fontsize=40, color=:white, align=(:center, :top))
    text!(ax, game_message, position=Point2f(COURT_WIDTH/2.0f0, 30.0f0), fontsize=30, color=:yellow, align=(:center, :bottom))

    function reset_ball()
        current_ball_speed[] = BALL_SPEED_INIT; ball_vel[] = Vec2f(0.0f0, 0.0f0)
        local_paddle_y = 0.0f0; ball_x = 0.0f0
        if served_by[] == :p1
            local_paddle_y = paddle_left_y[]; ball_x = PADDLE_WIDTH + BALL_SIZE/2.0f0 + 1.0f0
            serve_state[] = :p1_serve; game_message[] = "P1: Use A/D or Space to Serve/Push/Pull"
        else
            local_paddle_y = paddle_right_y[]; ball_x = COURT_WIDTH - PADDLE_WIDTH - BALL_SIZE/2.0f0 - 1.0f0
            serve_state[] = :p2_serve; game_message[] = "P2: Use ←/→ or Enter to Serve/Push/Pull"
        end
        ball_y = local_paddle_y + PADDLE_HEIGHT/2.0f0
        new_pos = Point2f(ball_x, clamp(ball_y, BALL_SIZE/2.0f0, COURT_HEIGHT-BALL_SIZE/2.0f0))
        ball_pos[] = new_pos; prev_ball_pos[] = new_pos
    end
    reset_ball()

    function update_visualization_data!()
        color_image_data_local = Observable(zeros(RGB{Float32}, NX, NY))[]
        for j_vis in 1:NY, i_vis in 1:NX
            idx_sim_grid = IX(i_vis + 1, j_vis + 1)
            r = clamp(fluid_color_R[idx_sim_grid], 0.0f0, 1.0f0)
            g = clamp(fluid_color_G[idx_sim_grid], 0.0f0, 1.0f0)
            b = clamp(fluid_color_B[idx_sim_grid], 0.0f0, 1.0f0)
            color_image_data_local[i_vis, j_vis] = RGB{Float32}(r, g, b)
        end
        fluid_color_image_obs[] = color_image_data_local

        num_arrows = length(arrow_pos_obs[])
        if num_arrows == 0; arrow_dir_obs[] = Vec2f[]; return; end

        new_arrow_dirs_local = Vector{Vec2f}(undef, num_arrows)
        vis_idx = 1
        epsilon_director = 1e-7 # For director calculation (Float64)

        for j_glob in 1:ARROW_SUBSAMPLE:NY, i_glob in 1:ARROW_SUBSAMPLE:NX
            if vis_idx > num_arrows; break; end
            idx_sim_grid = IX(i_glob + 1, j_glob + 1)
            qxx_val = fluid_Qxx[idx_sim_grid]
            qxy_val = fluid_Qxy[idx_sim_grid]

            angle_director = 0.5f0 * atan(qxy_val, qxx_val)
            director_x = cos(angle_director)
            director_y = sin(angle_director)

            norm_dir = sqrt(director_x^2 + director_y^2)
            if norm_dir > Float32(epsilon_director)
                new_arrow_dirs_local[vis_idx] = Vec2f(director_x/norm_dir, director_y/norm_dir)
            else
                new_arrow_dirs_local[vis_idx] = Vec2f(1.0f0, 0.0f0)
            end
            vis_idx += 1
        end
        if num_arrows > 0 && vis_idx -1 == num_arrows ; arrow_dir_obs[] = new_arrow_dirs_local; end
    end

    on(events(fig).tick) do _
        if !game_active[]; return Consume(false); end
        current_time = time()
        dt_game = Float32(clamp(current_time - last_update_time[], 0.001, 0.05))
        last_update_time[] = current_time

        keys = events(fig).keyboardstate; bp = ball_pos[]; bv = ball_vel[]; cbs_val = current_ball_speed[];
        p_left_y_val = paddle_left_y[]; p_right_y_val = paddle_right_y[]; prev_ball_pos[] = bp

        paddle_delta = PADDLE_SPEED * dt_game
        if Keyboard.w in keys; paddle_left_y[] = min(paddle_left_y[] + paddle_delta, COURT_HEIGHT - PADDLE_HEIGHT); end
        if Keyboard.s in keys; paddle_left_y[] = max(paddle_left_y[] - paddle_delta, 0.0f0); end
        if Keyboard.up in keys; paddle_right_y[] = min(paddle_right_y[] + paddle_delta, COURT_HEIGHT - PADDLE_HEIGHT); end
        if Keyboard.down in keys; paddle_right_y[] = max(paddle_right_y[] - paddle_delta, 0.0f0); end

        direct_pull_force = Vec2f(0.0f0)
        is_p1_pushing_this_frame = (Keyboard.d in keys || (Keyboard.space in keys && served_by[] == :p1)) && serve_state[] == :playing
        is_p2_pushing_this_frame = (Keyboard.left in keys || (Keyboard.enter in keys && served_by[] == :p2)) && serve_state[] == :playing

        if Keyboard.a in keys && serve_state[] == :playing
            paddle_center_y = p_left_y_val + PADDLE_HEIGHT / 2.0f0
            paddle_center_world = Point2f(PADDLE_WIDTH / 2.0f0, paddle_center_y)
            dir_to_paddle = paddle_center_world - bp; dist_sq = sum(dir_to_paddle .^ 2)
            if dist_sq > 1e-4; direct_pull_force += normalize(dir_to_paddle) * PULL_FORCE_STRENGTH; end
        end
        if Keyboard.right in keys && serve_state[] == :playing
            paddle_center_y = p_right_y_val + PADDLE_HEIGHT / 2.0f0
            paddle_center_world = Point2f(COURT_WIDTH - PADDLE_WIDTH / 2.0f0, paddle_center_y)
            dir_to_paddle = paddle_center_world - bp; dist_sq = sum(dir_to_paddle .^ 2)
            if dist_sq > 1e-4; direct_pull_force += normalize(dir_to_paddle) * PULL_FORCE_STRENGTH; end
        end

        fluid_time_accumulator[] += dt_game
        steps_taken = 0; max_steps_per_frame = 5
        while fluid_time_accumulator[] >= DT && steps_taken < max_steps_per_frame
            if is_p1_pushing_this_frame; apply_paddle_jet_color_effect!(1, p_left_y_val); end
            if is_p2_pushing_this_frame; apply_paddle_jet_color_effect!(2, p_right_y_val); end
            fluid_step!(is_p1_pushing_this_frame, p_left_y_val, is_p2_pushing_this_frame, p_right_y_val)
            fluid_time_accumulator[] -= DT
            steps_taken += 1
        end
        if steps_taken > 0; update_visualization_data!(); end

        if serve_state[] == :playing
            total_force_on_ball = Vec2f(0.0f0)
            fluid_vel_at_ball = get_fluid_velocity_at(bp)
            relative_vel = fluid_vel_at_ball - bv
            drag_force = BALL_DRAG_COEFF * relative_vel
            total_force_on_ball += drag_force
            total_force_on_ball += direct_pull_force

            new_bv = bv + total_force_on_ball * dt_game
            new_bp = bp + new_bv * dt_game

            if new_bp[2] < BALL_SIZE/2.0f0 || new_bp[2] > COURT_HEIGHT-BALL_SIZE/2.0f0
                 new_bp = Point2f(new_bp[1], clamp(new_bp[2], BALL_SIZE/2.0f0+0.1f0, COURT_HEIGHT-BALL_SIZE/2.0f0-0.1f0))
                 new_bv = Vec2f(new_bv[1], -new_bv[2])
            end

            paddle_left_rect = Rect2f(0.0f0, p_left_y_val, PADDLE_WIDTH, PADDLE_HEIGHT)
            paddle_right_rect = Rect2f(COURT_WIDTH-PADDLE_WIDTH, p_right_y_val, PADDLE_WIDTH, PADDLE_HEIGHT)
            ball_rect_predict = Rect2f(new_bp[1]-BALL_SIZE/2.0f0, new_bp[2]-BALL_SIZE/2.0f0, BALL_SIZE, BALL_SIZE)

            cbs_val_local = cbs_val
            if new_bv[1]<0.0f0 && rect_overlaps(paddle_left_rect, ball_rect_predict)
                 paddle_center_y_val = p_left_y_val+PADDLE_HEIGHT/2.0f0
                 hit_offset = clamp((new_bp[2]-paddle_center_y_val)/(PADDLE_HEIGHT/2.0f0), -1.0f0, 1.0f0)
                 bounce_angle = hit_offset * (Float32(pi)/3.0f0)
                 cbs_val_local += BALL_SPEED_INCREASE
                 new_bv = Vec2f(cos(bounce_angle), sin(bounce_angle))*cbs_val_local
                 new_bp = Point2f(PADDLE_WIDTH+BALL_SIZE/2.0f0+0.1f0, new_bp[2])
            elseif new_bv[1]>0.0f0 && rect_overlaps(paddle_right_rect, ball_rect_predict)
                 paddle_center_y_val = p_right_y_val+PADDLE_HEIGHT/2.0f0
                 hit_offset = clamp((new_bp[2]-paddle_center_y_val)/(PADDLE_HEIGHT/2.0f0), -1.0f0, 1.0f0)
                 bounce_angle = hit_offset * (Float32(pi)/3.0f0)
                 cbs_val_local += BALL_SPEED_INCREASE
                 world_angle = Float32(pi) - bounce_angle
                 new_bv = Vec2f(cos(world_angle), sin(world_angle))*cbs_val_local
                 new_bp = Point2f(COURT_WIDTH-PADDLE_WIDTH-BALL_SIZE/2.0f0-0.1f0, new_bp[2])
            end
            ball_pos[] = new_bp; ball_vel[] = new_bv; current_ball_speed[] = cbs_val_local

            scored = false; final_pos = ball_pos[]
            if final_pos[1] < BALL_SIZE/2.0f0; score_right[]+=1; served_by[]=:p1; scored=true; end
            if final_pos[1] > COURT_WIDTH-BALL_SIZE/2.0f0; score_left[]+=1; served_by[]=:p2; scored=true; end
            if scored
                 if score_left[] >= SCORE_LIMIT || score_right[] >= SCORE_LIMIT
                     game_active[] = false; winner = score_left[] >= SCORE_LIMIT ? "Left" : "Right"; game_message[] = "$winner Player Wins!\nPress R to Restart"
                 else; reset_ball(); end
            end
        elseif serve_state[] == :p1_serve
            target_y=p_left_y_val+PADDLE_HEIGHT/2.0f0; new_pos=Point2f(bp[1], clamp(target_y, BALL_SIZE/2.0f0, COURT_HEIGHT-BALL_SIZE/2.0f0)); ball_pos[]=new_pos; prev_ball_pos[]=new_pos
        elseif serve_state[] == :p2_serve
            target_y=p_right_y_val+PADDLE_HEIGHT/2.0f0; new_pos=Point2f(bp[1], clamp(target_y, BALL_SIZE/2.0f0, COURT_HEIGHT-BALL_SIZE/2.0f0)); ball_pos[]=new_pos; prev_ball_pos[]=new_pos
        end
        return Consume(false)
    end

    on(events(fig).keyboardbutton) do event
        if event.action != Keyboard.press; return Consume(false); end

        is_serve_key_p1 = event.key == Keyboard.space || event.key == Keyboard.d
        is_serve_key_p2 = event.key == Keyboard.enter || event.key == Keyboard.left
        current_s_state = serve_state[]

        if game_active[] && current_s_state != :playing
            serve_triggered = false; base_angle = 0.0f0
            if current_s_state == :p1_serve && is_serve_key_p1
                serve_triggered=true; base_angle=0.0f0; served_by[]=:p1
            elseif current_s_state == :p2_serve && is_serve_key_p2
                serve_triggered=true; base_angle=Float32(pi); served_by[]=:p2
            end

            if serve_triggered
                 angle_offset = (rand(Float32) * (Float32(pi)/3.0f0)) - (Float32(pi)/6.0f0)
                 angle = base_angle + angle_offset
                 serve_speed = current_ball_speed[]
                 ball_vel[] = Vec2f(cos(angle), sin(angle)) * serve_speed
                 serve_state[] = :playing; game_message[] = ""; return Consume(true)
            end
        end
        if !game_active[] && event.key == Keyboard.r
            score_left[]=0; score_right[]=0; served_by[]=rand([:p1, :p2]); reset_ball(); game_active[]=true
            
            fill!.((fluid_vx, fluid_vy, fluid_vx0, fluid_vy0, fluid_p, fluid_div,
                    Sxx_flow_temp, Sxy_flow_temp, fluid_Qxx0, fluid_Qxy0,
                    fluid_Hxx, fluid_Hxy, fluid_div_sigma_active_x, fluid_div_sigma_active_y,
                    fluid_div_sigma_elastic_x, fluid_div_sigma_elastic_y,
                    fluid_color_R, fluid_color_G, fluid_color_B,
                    fluid_color_R0, fluid_color_G0, fluid_color_B0), 0.0f0)
            rand_scale_init=0.01f0
            for i in eachindex(fluid_Qxx)
                fluid_Qxx[i]=(rand(Float32)-0.5f0)*2.0f0*rand_scale_init
                fluid_Qxy[i]=(rand(Float32)-0.5f0)*2.0f0*rand_scale_init
            end
            set_boundary_scalar!(fluid_Qxx); set_boundary_scalar!(fluid_Qxy)
            set_boundary_scalar!(fluid_color_R); set_boundary_scalar!(fluid_color_G); set_boundary_scalar!(fluid_color_B)
            update_visualization_data!(); return Consume(true)
        end
        return Consume(false)
    end
    update_visualization_data!()
    display(fig); return fig
end

run()