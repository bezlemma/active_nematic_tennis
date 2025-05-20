using GLMakie
using Random
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Colors

# --- Constants for Green Color from Activity ---
const ACTIVITY_GREEN_DIV_STRESS_SQ_THRESHOLD = 0.02f0 # Squared magnitude of div(active_stress) to trigger green (tune this)
const ACTIVITY_GREEN_SCALAR_ADD = 10.02f0             # Amount of green to add if threshold met (tune this)

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
const NX = 160 # Resolution in X
const NY = 120 # Resolution in Y
const DT = 0.001f0 # Time step for Q evolution (may need tuning)
const SOLVER_ITER = 20 # Iterations for linear solver (Helmholtz and Poisson)

# --- Active Nematic Dimensionless Parameters ---
const μ = 0.01f0 #viscosity μ
const ζ = 0.0000001f0 #substrate friction ζ
const λ= 0.7f0 #flow alignment λ
const γ = 1.0f0
const α = 0.6f0 # α (activity strength, >0 for extensile)
const NEMATIC_A_PAPER = 1.0f0
const NEMATIC_B_PAPER = -1.0f0
const NEMATIC_C_PAPER = 1.0f0
const NEMATIC_K_ELASTIC_PAPER = 0.01f0

# --- Paddle Jet Constants ---
const PADDLE_JET_FORCE_STRENGTH = 0.2f0 # NEW: Strength of the body force applied by the paddle jet (tune this!)
const TARGET_JET_S0 = 0.65f0         # Target nematic scalar order parameter (S₀) in the jet
const JET_CONE_LENGTH_CELLS = 10     # How many cells deep the direct paddle influence is
const JET_MAX_WIDTH_FRAC = 1.0f0 / 2.0f0 # Fraction of paddle height for max jet width at cone tip

# --- Ball/Fluid Interaction ---
const BALL_DRAG_COEFF = 0.2f0
const vel_scale = 1.0f0 # Scales fluid velocity for ball interaction

# --- Pull Constants (Push constants are superseded by Jet constants for the push effect) ---
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
global fluid_vx0::Vector{Float32} = zeros(Float32, FLUID_SIZE) # This will store sum of forces (nematic + paddle)
global fluid_vy0::Vector{Float32} = zeros(Float32, FLUID_SIZE) # This will store sum of forces (nematic + paddle)
global fluid_p::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div::Vector{Float32} = zeros(Float32, FLUID_SIZE)

global fluid_Qxx::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_Qxy::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_Qxx0::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_Qxy0::Vector{Float32} = zeros(Float32, FLUID_SIZE)

global fluid_Hxx::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_Hxy::Vector{Float32} = zeros(Float32, FLUID_SIZE)

global fluid_div_sigma_active_x::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_active_y::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_elastic_x::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div_sigma_elastic_y::Vector{Float32} = zeros(Float32, FLUID_SIZE)

# --- Color Tracer Fields ---
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


@inline IX(i, j) = clamp(i, 1, NX + 2) + (clamp(j, 1, NY + 2) - 1) * (NX + 2)

function set_boundary_velocity!(b::Int, x::Vector{Float32})
    N = NX; M = NY
    for i in 1:(N+2)
        x[IX(i, 1)]   = b == 2 ? 0.0f0 : x[IX(i,2)]
        x[IX(i, M+2)] = b == 2 ? 0.0f0 : x[IX(i,M+1)]
    end
    for j in 1:(M+2)
        x[IX(1, j)]   = b == 1 ? 0.0f0 : x[IX(2,j)]
        x[IX(N+2, j)] = b == 1 ? 0.0f0 : x[IX(N+1,j)]
    end
    # Ensure tangential flow for no-slip, or free-slip based on 'b'
     for i in 1:(N+2) # Top/Bottom boundaries
         x[IX(i,1)] = b==1 ? x[IX(i,2)] : -x[IX(i,2)] # If b=1 (vx), tangential means copy. If b=2 (vy), normal is zero, tangential would be -vy[i,2] if it were mirrored.
         x[IX(i,M+2)] = b==1 ? x[IX(i,M+1)] : -x[IX(i,M+1)] # For vy, this should be -vy[i,M+1] to simulate reflection if b=2
     end
     for j in 1:(M+2) # Left/Right boundaries
         x[IX(1,j)] = b==2 ? x[IX(2,j)] : -x[IX(2,j)]   # If b=2 (vy), tangential means copy. If b=1 (vx), normal is zero.
         x[IX(N+2,j)] = b==2 ? x[IX(N+1,j)] : -x[IX(N+1,j)]
     end

    x[IX(1,1)] = 0.5f0*(x[IX(1,2)]+x[IX(2,1)])
    x[IX(1,M+2)] = 0.5f0*(x[IX(1,M+1)]+x[IX(2,M+2)])
    x[IX(N+2,1)] = 0.5f0*(x[IX(N+2,2)]+x[IX(N+1,1)])
    x[IX(N+2,M+2)] = 0.5f0*(x[IX(N+2,M+1)]+x[IX(N+1,M+2)])
end

function set_boundary_scalar!(x::Vector{Float32})
    N = NX; M = NY
    for i in 1:(N+2); x[IX(i, 1)] = x[IX(i, 2)]; x[IX(i, M + 2)] = x[IX(i, M + 1)]; end
    for j in 1:(M+2); x[IX(1, j)] = x[IX(2, j)]; x[IX(N + 2, j)] = x[IX(N + 1, j)]; end
    x[IX(1, 1)] = 0.5f0 * (x[IX(2, 1)] + x[IX(1, 2)]); x[IX(1, M + 2)] = 0.5f0 * (x[IX(2, M + 2)] + x[IX(1, M + 1)])
    x[IX(N + 2, 1)] = 0.5f0 * (x[IX(N + 1, 1)] + x[IX(N + 2, 2)]); x[IX(N + 2, M + 2)] = 0.5f0 * (x[IX(N + 1, M + 2)] + x[IX(N + 2, M + 1)])
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
    for k in 1:iterations
        x_prev_iter = copy(x_out) # Use current x_out as guess for this iteration
        for j in 2:(NY+1), i in 2:(NX+1)
            idx = IX(i,j)
            sum_neighbors_scaled = coeff_lap_term_dx * (x_prev_iter[IX(i-1,j)] + x_prev_iter[IX(i+1,j)]) +
                                 coeff_lap_term_dy * (x_prev_iter[IX(i,j-1)] + x_prev_iter[IX(i,j+1)])

            denominator = coeff_I_term + 2.0f0*coeff_lap_term_dx + 2.0f0*coeff_lap_term_dy
            if abs(denominator) < 1e-9 # Avoid division by zero
                x_out[idx] = 0.0f0
            else
                x_out[idx] = (x0_rhs[idx] + sum_neighbors_scaled) / denominator
            end
        end
        b_type == 1 || b_type == 2 ? set_boundary_velocity!(b_type, x_out) : set_boundary_scalar!(x_out)
    end
end


function advect!(b_type::Int, d::Vector{Float32}, d0::Vector{Float32}, velX::Vector{Float32}, velY::Vector{Float32}, dt::Float32)
    for j in 2:(NY+1), i in 2:(NX+1)
        idx = IX(i, j)
        x_particle = Float32(i) - velX[idx] * dt * INV_DX # Corrected: use cell index for particle origin
        y_particle = Float32(j) - velY[idx] * dt * INV_DY

        x_particle = clamp(x_particle, 1.5f0, NX + 0.5f0)
        y_particle = clamp(y_particle, 1.5f0, NY + 0.5f0)

        i0 = floor(Int, x_particle); i1 = i0 + 1
        j0 = floor(Int, y_particle); j1 = j0 + 1

        s1 = x_particle - i0; s0 = 1.0f0 - s1
        t1 = y_particle - j0; t0 = 1.0f0 - t1

        d[idx] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                 s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)])
    end
    b_type == 1 || b_type == 2 ? set_boundary_velocity!(b_type, d) : set_boundary_scalar!(d)
end

function project!(velX::Vector{Float32}, velY::Vector{Float32}, p::Vector{Float32}, div_field::Vector{Float32})
    for j in 2:(NY+1), i in 2:(NX+1)
        idx = IX(i,j)
        div_field[idx] = (velX[IX(i+1, j)] - velX[IX(i-1, j)]) * 0.5f0 * INV_DX +
                       (velY[IX(i, j+1)] - velY[IX(i, j-1)]) * 0.5f0 * INV_DY
        p[idx] = 0.0f0
    end
    set_boundary_scalar!(div_field); set_boundary_scalar!(p)

    # c_inv_poisson = 1.0f0 / (2.0f0 * (INV_DX2 + INV_DY2)) # Denominator for direct solution, not iterative form
    for k in 1:SOLVER_ITER
        p_old_iter = copy(p)
        for j in 2:(NY+1), i in 2:(NX+1)
            idx = IX(i,j)
            sum_lap_neighbors = (p_old_iter[IX(i+1,j)] + p_old_iter[IX(i-1,j)]) * INV_DX2 +
                                (p_old_iter[IX(i,j+1)] + p_old_iter[IX(i,j-1)]) * INV_DY2
            # Iterative solver for Poisson: p[idx] = (sum_lap_neighbors - div_field[idx]) / (2*(INV_DX2+INV_DY2))
            p[idx] = ((p_old_iter[IX(i+1,j)] + p_old_iter[IX(i-1,j)]) * INV_DX2 +
                      (p_old_iter[IX(i,j+1)] + p_old_iter[IX(i,j-1)]) * INV_DY2 - div_field[idx]) /
                     (2.0f0 * (INV_DX2 + INV_DY2))
        end
        set_boundary_scalar!(p)
    end

    for j in 2:(NY+1), i in 2:(NX+1)
        idx = IX(i,j)
        velX[idx] -= (p[IX(i+1, j)] - p[IX(i-1, j)]) * 0.5f0 * INV_DX
        velY[idx] -= (p[IX(i, j+1)] - p[IX(i, j-1)]) * 0.5f0 * INV_DY
    end
    set_boundary_velocity!(1, velX); set_boundary_velocity!(2, velY)
end

function calculate_H!(Hxx_out::Vector{Float32}, Hxy_out::Vector{Float32},
                      Qxx_in::Vector{Float32}, Qxy_in::Vector{Float32},
                      K_elastic::Float32, A_coeff::Float32, C_coeff::Float32)
    Threads.@threads for j in 2:(NY+1)
        for i in 2:(NX+1)
            idx = IX(i,j)
            qxx_val = Qxx_in[idx]
            qxy_val = Qxy_in[idx]

            lap_qxx = laplacian_scalar(Qxx_in, i, j)
            lap_qxy = laplacian_scalar(Qxy_in, i, j)
            trQ2_val = 2.0f0 * (qxx_val*qxx_val + qxy_val*qxy_val) # Q:Q = Qxx^2 + 2Qxy^2 + Qyy^2 = 2(Qxx^2+Qxy^2) for traceless Qyy=-Qxx

            Hxx_out[idx] = -(A_coeff * qxx_val + 2.0f0 * C_coeff * trQ2_val * qxx_val - K_elastic * lap_qxx)
            Hxy_out[idx] = -(A_coeff * qxy_val + 2.0f0 * C_coeff * trQ2_val * qxy_val - K_elastic * lap_qxy)
        end
    end
    set_boundary_scalar!(Hxx_out); set_boundary_scalar!(Hxy_out)
end

function calculate_S_flow_term!(Sxx_out::Vector{Float32}, Sxy_out::Vector{Float32},
                               Qxx_in::Vector{Float32}, Qxy_in::Vector{Float32},
                               vx::Vector{Float32}, vy::Vector{Float32}, lambda_align::Float32)
    Threads.@threads for j in 2:(NY+1)
        for i in 2:(NX+1)
            idx = IX(i,j)
            qxx = Qxx_in[idx]
            qxy = Qxy_in[idx]

            dvx_dx = grad_x_centered(vx,i,j); dvx_dy = grad_y_centered(vx,i,j)
            dvy_dx = grad_x_centered(vy,i,j); dvy_dy = grad_y_centered(vy,i,j)

            # Strain rate tensor D_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
            Dxx = dvx_dx # Dyy = dvy_dy = -dvx_dx for incompressible
            Dxy = 0.5f0 * (dvx_dy + dvy_dx)
            # Vorticity tensor W_ij = 0.5 * (du_i/dx_j - du_j/dx_i) => W_xy = 0.5 * (dvx_dy - dvy_dx)
            # The code uses omega_xy_tensor = 0.5f0 * (dvy_dx - dvx_dy) which is -W_yx or W_xy but named like general omega.
            # Let's assume omega_xy_tensor is the relevant component for 2D (Ω_xy in Beris-Edwards).
            # Commutator [Ω, Q]_xx = Ω_xz Q_zy - Q_xz Ω_zy for 3D. In 2D, using Ω_xy (rotation in xy plane):
            # Ω = [[0, omega_xy_tensor], [-omega_xy_tensor, 0]]
            # Q = [[qxx, qxy], [qxy, -qxx]]
            # (ΩQ)_xx = Ω_xy * qyx = omega_xy_tensor * qxy
            # (QΩ)_xx = qxx * Ω_xx + qxy * Ω_yx = qxy * (-omega_xy_tensor)
            # [Ω,Q]_xx = (ΩQ)_xx - (QΩ)_xx = omega_xy_tensor * qxy - (-omega_xy_tensor * qxy) = 2 * omega_xy_tensor * qxy
            omega_xy_val = 0.5f0 * (dvy_dx - dvx_dy) # This is W_yx (or equivalent of Ω_z component times Levi-Civita)

            comm_Omega_Q_xx = 2.0f0 * omega_xy_val * qxy
            comm_Omega_Q_xy = -2.0f0 * omega_xy_val * qxx # This corresponds to [Ω,Q]_xy

            # S_ij = ξ D_ij + [Ω,Q]_ij (for general flow-aligning parameter ξ)
            # Your Sxx_out corresponds to (λ Dxx - comm_Omega_Q_xx) if we assume this is the upper-left of S_ij tensor.
            # The original code had lambda_align * Dxx. If lambda_align includes the factor of 2 for S_0, then this is fine.
            # Let's assume lambda_align is ξ.
            Sxx_out[idx] = lambda_align * Dxx - comm_Omega_Q_xx
            Sxy_out[idx] = lambda_align * Dxy - comm_Omega_Q_xy
        end
    end
end

function calculate_div_active_stress!(div_sx::Vector{Float32}, div_sy::Vector{Float32},
                                      Qxx_in::Vector{Float32}, Qxy_in::Vector{Float32},
                                      alpha_activity::Float32)
    Threads.@threads for j in 2:(NY+1)
        for i in 2:(NX+1)
            idx = IX(i,j) # Index for the output arrays div_sx, div_sy
            # Active stress sigma_ij^A = -alpha * Q_ij
            # div(sigma^A)_x = d(sigma_xx^A)/dx + d(sigma_xy^A)/dy = -alpha * (dQxx/dx + dQxy/dy)
            # div(sigma^A)_y = d(sigma_yx^A)/dx + d(sigma_yy^A)/dy = -alpha * (dQxy/dx + dQyy/dy)
            # Since Qyy = -Qxx (traceless), dQyy/dy = -dQxx/dy
            # So, div(sigma^A)_y = -alpha * (dQxy/dx - dQxx/dy)

            dQxx_dx = grad_x_centered(Qxx_in,i,j); dQxx_dy = grad_y_centered(Qxx_in,i,j)
            dQxy_dx = grad_x_centered(Qxy_in,i,j); dQxy_dy = grad_y_centered(Qxy_in,i,j)

            div_sx[idx] = -alpha_activity * (dQxx_dx + dQxy_dy)
            div_sy[idx] = -alpha_activity * (dQxy_dx - dQxx_dy) # dQyy/dy = -dQxx/dy
        end
    end
    # No boundary setting needed here as these are source terms, gradients are on internal points.
end

function calculate_div_elastic_stress!(div_out_x::Vector{Float32}, div_out_y::Vector{Float32},
                                       Qxx_vec::Vector{Float32}, Qxy_vec::Vector{Float32},
                                       Hxx_vec::Vector{Float32}, Hxy_vec::Vector{Float32},
                                       K_elastic::Float32, lambda_align::Float32) # lambda_align is ξ here

    # Temporary arrays for elastic stress components
    # sigma_el_ij = - K (dQ_ik/dx_l)(dQ_jk/dx_l) - lambda [Q,H]_ij + Q_ik H_kj - H_ik Q_kj
    # The last two terms Q_ik H_kj - H_ik Q_kj are part of the generalized stress, often grouped.
    # Let's use the formulation from the paper often cited for these type of sims:
    # sigma_E_ij = -P delta_ij + Q_ik H_kj - H_ik Q_kj - lambda (Q_ik H_kj + H_ik Q_kj)
    # This is complex. The code implements a specific form.
    # The provided code calculates:
    # sigma_el_xx = -lambda_align * comm_QH_xx - 2K ( (dqxx_dx)^2 + (dqxy_dx)^2 )
    # sigma_el_xy = -lambda_align * comm_QH_xy - 2K ( (dqxx_dx*dqxx_dy) + (dqxy_dx*dqxy_dy) )
    # sigma_el_yy =                           - 2K ( (dqxx_dy)^2 + (dqxy_dy)^2 )
    # where comm_QH_xx = 0 and comm_QH_xy = (q_xx*h_xy - q_xy*h_xx) - (h_xx*q_xy - h_xy*q_xx)
    # This comm_QH_xy is actually 2 * (q_xx*h_xy - q_xy*h_xx), so it's 2 * [Q,H]_xy if H is symmetric.

    sigma_el_xx_temp = zeros(Float32, FLUID_SIZE)
    sigma_el_xy_temp = zeros(Float32, FLUID_SIZE)
    sigma_el_yy_temp = zeros(Float32, FLUID_SIZE) # Qyy = -Qxx, Hyy = -Hxx

    # Part 1: Terms involving H (related to Ericksen stress and flow alignment coupling)
    Threads.@threads for j_idx in 2:(NY+1)
        for i_idx in 2:(NX+1)
            idx = IX(i_idx,j_idx)
            q_xx = Qxx_vec[idx]; q_xy = Qxy_vec[idx]
            h_xx = Hxx_vec[idx]; h_xy = Hxy_vec[idx]

            # [Q,H]_xx = Q_xz H_zy - H_xz Q_zy (3D). In 2D:
            # (QH)_xx = qxx hxx + qxy hyx
            # (HQ)_xx = hxx qxx + hxy qyx
            # [Q,H]_xx = (qxx hxx + qxy hxy) - (hxx qxx + hxy qxy) = 0 (if H is symmetric, hyx=hxy)

            # [Q,H]_xy = Q_xz H_zy - H_xz Q_zy (3D). In 2D:
            # (QH)_xy = qxx hxy + qxy hyy = qxx hxy - qxy hxx (since hyy = -hxx)
            # (HQ)_xy = hxx qxy + hxy qyy = hxx qxy - hxy qxx (since qyy = -qxx)
            # [Q,H]_xy = (qxx hxy - qxy hxx) - (hxx qxy - hxy qxx)
            #           = qxx hxy - qxy hxx - hxx qxy + hxy qxx
            #           = 2 * (qxx hxy - qxy hxx)
            
            # The code's comm_QH_xy = (q_xx*h_xy - q_xy*h_xx) - (h_xx*q_xy - h_xy*q_xx) seems to be related but
            # if H is symmetric (h_xy = h_yx), (h_xx*q_xy - h_xy*q_xx) is -(q_xy*h_xx - q_xx*h_xy)
            # so comm_QH_xy = 2 * (q_xx*h_xy - q_xy*h_xx). This is [Q,H]_xy.

            comm_QH_xx_val = 0.0f0 # As derived, assuming H_ij is symmetric like Q_ij
            comm_QH_xy_val = 2.0f0 * (q_xx * h_xy - q_xy * h_xx)


            # Contribution from Q_ik H_kj - H_ik Q_kj (this is [Q,H]_ij itself) if lambda_align is not 1.
            # The paper form often has (Q_ik H_kj - H_ik Q_kj) - lambda_align * (Q_ik H_kj + H_ik Q_kj)
            # The code seems to be using a simplified stress related to flow alignment terms.
            # The term -lambda_align * comm_QH seems related to the distortion part of Ericksen stress.
            # sigma_dist_ij = -lambda * [Q,H]_ij (this appears in some formulations for elastic stress)
            sigma_el_xx_temp[idx] = -lambda_align * comm_QH_xx_val # Will be 0
            sigma_el_xy_temp[idx] = -lambda_align * comm_QH_xy_val
            # sigma_el_yy is not directly set here from this term in the original code.
            # For symmetry, if sigma_el_xx uses [Q,H]_xx, then sigma_el_yy should use [Q,H]_yy.
            # [Q,H]_yy = Q_yx H_xy - H_yx Q_xy = qxy hxy - hxy qxy = 0 for symmetric H, Q.
            # Or if comm_QH_yy is -comm_QH_xx due to tracelessness, it's also 0.
            sigma_el_yy_temp[idx] = lambda_align * comm_QH_xx_val # if sigma_el_yy = -sigma_el_xx from this part
        end
    end

    # Part 2: Terms involving K_elastic (Landau-de Gennes stress / Frank elastic stress part)
    # sigma_K_ij = -K (dQ_kl/dx_i)(dQ_kl/dx_j) (from Leslie-Ericksen, more or less)
    Threads.@threads for j_idx in 2:(NY+1)
        for i_idx in 2:(NX+1)
            idx = IX(i_idx, j_idx)

            dqxx_dx = grad_x_centered(Qxx_vec, i_idx, j_idx)
            dqxy_dx = grad_x_centered(Qxy_vec, i_idx, j_idx)
            # dqyy_dx = -dqxx_dx

            dqxx_dy = grad_y_centered(Qxx_vec, i_idx, j_idx)
            dqxy_dy = grad_y_centered(Qxy_vec, i_idx, j_idx)
            # dqyy_dy = -dqxx_dy

            # Sum over k,l: (dQ_kl/dx_i) * (dQ_kl/dx_j)
            # For sigma_K_xx (i=x, j=x): (dQ_xx/dx)^2 + (dQ_xy/dx)^2 + (dQ_yx/dx)^2 + (dQ_yy/dx)^2
            # = (dQ_xx/dx)^2 + (dQ_xy/dx)^2 + (dQ_xy/dx)^2 + (-dQ_xx/dx)^2
            # = 2 * ( (dQ_xx/dx)^2 + (dQ_xy/dx)^2 )
            sigma_K_xx_val = -K_elastic * 2.0f0 * (dqxx_dx^2 + dqxy_dx^2)

            # For sigma_K_xy (i=x, j=y): (dQ_xx/dx)(dQ_xx/dy) + (dQ_xy/dx)(dQ_xy/dy) + (dQ_yx/dx)(dQ_yx/dy) + (dQ_yy/dx)(dQ_yy/dy)
            # = (dQ_xx/dx)(dQ_xx/dy) + (dQ_xy/dx)(dQ_xy/dy) + (dQ_xy/dx)(dQ_xy/dy) + (-dQ_xx/dx)(-dQ_xx/dy)
            # = 2 * ( (dQ_xx/dx)(dQ_xx/dy) + (dQ_xy/dx)(dQ_xy/dy) )
            sigma_K_xy_val = -K_elastic * 2.0f0 * (dqxx_dx * dqxx_dy + dqxy_dx * dqxy_dy)

            # For sigma_K_yy (i=y, j=y): (dQ_xx/dy)^2 + (dQ_xy/dy)^2 + (dQ_yx/dy)^2 + (dQ_yy/dy)^2
            # = 2 * ( (dQ_xx/dy)^2 + (dQ_xy/dy)^2 )
            sigma_K_yy_val = -K_elastic * 2.0f0 * (dqxx_dy^2 + dqxy_dy^2)

            sigma_el_xx_temp[idx] += sigma_K_xx_val
            sigma_el_xy_temp[idx] += sigma_K_xy_val
            sigma_el_yy_temp[idx] += sigma_K_yy_val
        end
    end
    set_boundary_scalar!(sigma_el_xx_temp)
    set_boundary_scalar!(sigma_el_xy_temp)
    set_boundary_scalar!(sigma_el_yy_temp)

    # Part 3: Divergence of the total elastic stress
    Threads.@threads for j_idx in 2:(NY+1)
        for i_idx in 2:(NX+1)
            idx = IX(i_idx, j_idx)
            div_out_x[idx] = grad_x_centered(sigma_el_xx_temp, i_idx, j_idx) + grad_y_centered(sigma_el_xy_temp, i_idx, j_idx)
            div_out_y[idx] = grad_x_centered(sigma_el_xy_temp, i_idx, j_idx) + grad_y_centered(sigma_el_yy_temp, i_idx, j_idx)
        end
    end
    # No boundary setting for div_out_x, div_out_y as they are source terms.
end


# --- NEW HELPER FUNCTION FOR ADDING PADDLE JET BODY FORCE ---
function _add_jet_force_in_cone!(
    force_vx_array::Vector{Float32}, # This is fluid_vx0
    # force_vy_array::Vector{Float32}, # If y-force needed in future
    paddle_id::Int,
    paddle_base_y_world::Float32,
    force_strength::Float32
)
    paddle_center_y_world = paddle_base_y_world + PADDLE_HEIGHT / 2.0f0
    _, paddle_center_gy_float = world_to_grid(Point2f(0f0, paddle_center_y_world))
    paddle_center_gy_idx = clamp(round(Int, paddle_center_gy_float), 2, NY + 1)

    jet_force_x_actual = (paddle_id == 1) ? force_strength : -force_strength

    paddle_height_cells = PADDLE_HEIGHT / FLUID_DY
    max_jet_width_at_tip_cells = max(1, round(Int, paddle_height_cells * JET_MAX_WIDTH_FRAC))
    
    start_x_world_edge = (paddle_id == 1) ? PADDLE_WIDTH : (COURT_WIDTH - PADDLE_WIDTH)
    start_gxi_jet_float = if paddle_id == 1
        world_to_grid(Point2f(start_x_world_edge + FLUID_DX * 0.5f0, 0f0))[1]
    else
        world_to_grid(Point2f(start_x_world_edge - FLUID_DX * 0.5f0, 0f0))[1]
    end
    start_gxi_jet_idx = clamp(round(Int, start_gxi_jet_float), 2, NX + 1)

    base_half_width_cells = max(1, round(Int, (paddle_height_cells / 3.0) / 2.0)) 
    tip_half_width_cells = max(1, round(Int, max_jet_width_at_tip_cells / 2.0))   

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
            # If y-component of force is needed:
            # force_vy_array[idx_cone] += ... 
        end
    end
end


function fluid_step!(p1_is_pushing::Bool, p1_y_world::Float32, p2_is_pushing::Bool, p2_y_world::Float32)
    dt_q_evolution = DT

    u_old_vx_for_advection = copy(fluid_vx)
    u_old_vy_for_advection = copy(fluid_vy)

    fluid_Qxx0 .= fluid_Qxx
    fluid_Qxy0 .= fluid_Qxy
    advect!(0, fluid_Qxx, fluid_Qxx0, u_old_vx_for_advection, u_old_vy_for_advection, dt_q_evolution)
    advect!(0, fluid_Qxy, fluid_Qxy0, u_old_vx_for_advection, u_old_vy_for_advection, dt_q_evolution)

    fluid_color_R0 .= fluid_color_R
    fluid_color_G0 .= fluid_color_G
    fluid_color_B0 .= fluid_color_B
    advect!(0, fluid_color_R, fluid_color_R0, u_old_vx_for_advection, u_old_vy_for_advection, dt_q_evolution)
    advect!(0, fluid_color_G, fluid_color_G0, u_old_vx_for_advection, u_old_vy_for_advection, dt_q_evolution)
    advect!(0, fluid_color_B, fluid_color_B0, u_old_vx_for_advection, u_old_vy_for_advection, dt_q_evolution)

    calculate_H!(fluid_Hxx, fluid_Hxy, fluid_Qxx, fluid_Qxy,
                 NEMATIC_K_ELASTIC_PAPER, NEMATIC_A_PAPER, NEMATIC_C_PAPER)

    Sxx_flow_temp = zeros(Float32, FLUID_SIZE)
    Sxy_flow_temp = zeros(Float32, FLUID_SIZE)
    calculate_S_flow_term!(Sxx_flow_temp, Sxy_flow_temp, fluid_Qxx, fluid_Qxy,
                           u_old_vx_for_advection, u_old_vy_for_advection, λ)

    Threads.@threads for i in 1:FLUID_SIZE
        fluid_Qxx[i] += dt_q_evolution * (Sxx_flow_temp[i] + γ * fluid_Hxx[i])
        fluid_Qxy[i] += dt_q_evolution * (Sxy_flow_temp[i] + γ * fluid_Hxy[i])
    end
    set_boundary_scalar!(fluid_Qxx)
    set_boundary_scalar!(fluid_Qxy)

    calculate_H!(fluid_Hxx, fluid_Hxy, fluid_Qxx, fluid_Qxy,
                  NEMATIC_K_ELASTIC_PAPER, NEMATIC_A_PAPER, NEMATIC_C_PAPER)

    calculate_div_active_stress!(fluid_div_sigma_active_x, fluid_div_sigma_active_y,
                                 fluid_Qxx, fluid_Qxy, α)
    calculate_div_elastic_stress!(fluid_div_sigma_elastic_x, fluid_div_sigma_elastic_y,
                                  fluid_Qxx, fluid_Qxy, fluid_Hxx, fluid_Hxy,
                                  NEMATIC_K_ELASTIC_PAPER, λ)

    # --- Add Green Color Based on Nematic Activity ---
    Threads.@threads for i in 1:FLUID_SIZE
        active_force_sq_mag = fluid_div_sigma_active_x[i]^2 + fluid_div_sigma_active_y[i]^2
        if active_force_sq_mag > ACTIVITY_GREEN_DIV_STRESS_SQ_THRESHOLD
            fluid_color_G[i] = min(1.0f0, fluid_color_G[i] + ACTIVITY_GREEN_SCALAR_ADD)
        end
    end
    # Note: Boundaries for fluid_color_G were handled by the advect step.
    # This source term might lead to green appearing near boundaries if activity is high there.
    # The visualization will clamp it.

    Threads.@threads for i in 1:FLUID_SIZE
        fluid_vx0[i] = fluid_div_sigma_active_x[i] + fluid_div_sigma_elastic_x[i]
        fluid_vy0[i] = fluid_div_sigma_active_y[i] + fluid_div_sigma_elastic_y[i]
    end

    if p1_is_pushing
        _add_jet_force_in_cone!(fluid_vx0, 1, p1_y_world, PADDLE_JET_FORCE_STRENGTH)
    end
    if p2_is_pushing
        _add_jet_force_in_cone!(fluid_vx0, 2, p2_y_world, PADDLE_JET_FORCE_STRENGTH)
    end

    general_linear_solve!(1, fluid_vx, fluid_vx0,
                          ζ, μ * INV_DX2, μ * INV_DY2,
                          SOLVER_ITER)
    general_linear_solve!(2, fluid_vy, fluid_vy0,
                          ζ, μ * INV_DX2, μ * INV_DY2,
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

    s1 = gx - i0; s0 = 1.0f0 - s1
    t1 = gy - j0; t0 = 1.0f0 - t1

    vx = s0*(t0*fluid_vx[IX(i0, j0)] + t1*fluid_vx[IX(i0, j1)]) + s1*(t0*fluid_vx[IX(i1, j0)] + t1*fluid_vx[IX(i1, j1)])
    vy = s0*(t0*fluid_vy[IX(i0, j0)] + t1*fluid_vy[IX(i0, j1)]) + s1*(t0*fluid_vy[IX(i1, j0)] + t1*fluid_vy[IX(i1, j1)])
    return Vec2f(vx, vy) * vel_scale
end

function rect_overlaps(r1::Rect2f, r2::Rect2f)
    x_overlap = (r1.origin[1] < r2.origin[1] + r2.widths[1]) && (r1.origin[1] + r1.widths[1] > r2.origin[1])
    y_overlap = (r1.origin[2] < r2.origin[2] + r2.widths[2]) && (r1.origin[2] + r1.widths[2] > r2.origin[2])
    return x_overlap && y_overlap
end

# --- Paddle Jet Helper Function (Now for Q and Color only) ---
function apply_paddle_jet_effect!( # Renamed in thought process, but keeping user's name for now
    paddle_id::Int, 
    paddle_base_y_world::Float32 
)
    paddle_center_y_world = paddle_base_y_world + PADDLE_HEIGHT / 2.0f0
    _, paddle_center_gy_float = world_to_grid(Point2f(0f0, paddle_center_y_world)) 
    paddle_center_gy_idx = clamp(round(Int, paddle_center_gy_float), 2, NY + 1)

    # jet_speed_x_actual = (paddle_id == 1) ? PADDLE_JET_SPEED : -PADDLE_JET_SPEED # REMOVED
    director_angle = (paddle_id == 1) ? 0.0f0 : Float32(pi)

    paddle_height_cells = PADDLE_HEIGHT / FLUID_DY
    max_jet_width_at_tip_cells = max(1, round(Int, paddle_height_cells * JET_MAX_WIDTH_FRAC))

    start_x_world_edge = (paddle_id == 1) ? PADDLE_WIDTH : (COURT_WIDTH - PADDLE_WIDTH)
    
    start_gxi_jet_float = if paddle_id == 1
        world_to_grid(Point2f(start_x_world_edge + FLUID_DX * 0.5f0, 0f0))[1]
    else
        world_to_grid(Point2f(start_x_world_edge - FLUID_DX * 0.5f0, 0f0))[1]
    end
    start_gxi_jet_idx = clamp(round(Int, start_gxi_jet_float), 2, NX + 1)

    base_half_width_cells = max(1, round(Int, (paddle_height_cells / 3.0) / 2.0)) 
    tip_half_width_cells = max(1, round(Int, max_jet_width_at_tip_cells / 2.0))   

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

        for gy_offset_loop in -(current_jet_half_width_cells -1) : (current_jet_half_width_cells-1) 
            current_gyi = clamp(paddle_center_gy_idx + gy_offset_loop, 2, NY + 1)
            idx_cone = IX(current_gxi, current_gyi)

            # 1. SET Fluid Velocity -- REMOVED
            # fluid_vx[idx_cone] = jet_speed_x_actual # REMOVED
            # fluid_vy[idx_cone] = 0.0f0             # REMOVED

            # 2. SET Fluid Color
            if paddle_id == 1 
                fluid_color_R[idx_cone] = 1.0f0
                fluid_color_G[idx_cone] = 0.0f0
                fluid_color_B[idx_cone] = 0.0f0
            else 
                fluid_color_R[idx_cone] = 0.0f0
                fluid_color_G[idx_cone] = 0.0f0
                fluid_color_B[idx_cone] = 1.0f0
            end

            # 3. SET Nematic Alignment (Q tensor)
            #fluid_Qxx[idx_cone] = TARGET_JET_S0 * cos(2.0f0 * director_angle)
            #fluid_Qxy[idx_cone] = TARGET_JET_S0 * sin(2.0f0 * director_angle)
        end
    end
end


function run()
    fig = Figure(size=(COURT_WIDTH, COURT_HEIGHT), backgroundcolor=:black, figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect(), limits=(0, COURT_WIDTH, 0, COURT_HEIGHT), backgroundcolor=:dimgrey)
    hidedecorations!(ax); hidespines!(ax)

    # Initialize fluid arrays
    fill!(fluid_vx, 0.0f0); fill!(fluid_vy, 0.0f0); fill!(fluid_vx0, 0.0f0); fill!(fluid_vy0, 0.0f0)
    fill!(fluid_p, 0.0f0); fill!(fluid_div, 0.0f0)

    rand_scale = 0.01f0
    for i in eachindex(fluid_Qxx)
        fluid_Qxx[i] = (rand(Float32) - 0.5f0) * 2.0f0 * rand_scale
        fluid_Qxy[i] = (rand(Float32) - 0.5f0) * 2.0f0 * rand_scale
    end
    set_boundary_scalar!(fluid_Qxx); set_boundary_scalar!(fluid_Qxy)
    fill!(fluid_Qxx0, 0.0f0); fill!(fluid_Qxy0, 0.0f0) # These are used as temp storage in advection
    fill!(fluid_Hxx, 0.0f0); fill!(fluid_Hxy, 0.0f0)
    fill!(fluid_div_sigma_active_x, 0.0f0); fill!(fluid_div_sigma_active_y, 0.0f0)
    fill!(fluid_div_sigma_elastic_x, 0.0f0); fill!(fluid_div_sigma_elastic_y, 0.0f0)

    fill!(fluid_color_R, 0.0f0); fill!(fluid_color_G, 0.0f0); fill!(fluid_color_B, 0.0f0)
    fill!(fluid_color_R0, 0.0f0); fill!(fluid_color_G0, 0.0f0); fill!(fluid_color_B0, 0.0f0) # Temp for advection
    set_boundary_scalar!(fluid_color_R); set_boundary_scalar!(fluid_color_G); set_boundary_scalar!(fluid_color_B)


    arrow_positions = Point2f[];
    fluid_x_range_vis = LinRange(0+FLUID_DX/2, COURT_WIDTH-FLUID_DX/2, NX);
    fluid_y_range_vis = LinRange(0+FLUID_DY/2, COURT_HEIGHT-FLUID_DY/2, NY);
    for j in 1:ARROW_SUBSAMPLE:NY, i in 1:ARROW_SUBSAMPLE:NX
        push!(arrow_positions, Point2f(fluid_x_range_vis[i], fluid_y_range_vis[j]));
    end
    arrow_pos_obs[] = arrow_positions
    arrow_dir_obs[] = fill(Vec2f(1,0), length(arrow_positions)) # Initial direction


    ball_pos = Observable(Point2f(COURT_WIDTH / 2f0, COURT_HEIGHT / 2f0)); prev_ball_pos = Observable(Point2f(COURT_WIDTH / 2f0, COURT_HEIGHT / 2f0))
    ball_vel = Observable(Vec2f(0f0, 0f0)); current_ball_speed = Observable(BALL_SPEED_INIT)
    paddle_left_y = Observable(COURT_HEIGHT / 2f0 - PADDLE_HEIGHT / 2f0); paddle_right_y = Observable(COURT_HEIGHT / 2f0 - PADDLE_HEIGHT / 2f0)
    score_left = Observable(0); score_right = Observable(0); game_active = Observable(true)
    game_message = Observable("Press Serve Keys to Start"); serve_state = Observable(:p1_serve); served_by = Observable(:p1)
    last_update_time = Ref(time()); fluid_time_accumulator = Ref(0.0f0)

    image!(ax, (0.0f0, COURT_WIDTH), (0.0f0, COURT_HEIGHT), fluid_color_image_obs, interpolate=true)
    arrows!(ax, arrow_pos_obs, arrow_dir_obs, arrowsize=Vec2f(ARROW_LENGTH_SCALE*0.25, ARROW_LENGTH_SCALE*0.35), lengthscale=ARROW_LENGTH_SCALE, arrowcolor=:white, linecolor=:white, linewidth=0.5)

    ball_visual_size = BALL_SIZE * 0.9f0
    ball_rect_obs = @lift Rect2f($ball_pos[1]-ball_visual_size/2f0, $ball_pos[2]-ball_visual_size/2f0, ball_visual_size, ball_visual_size)
    poly!(ax, ball_rect_obs, color=:yellow, strokecolor=:orange, strokewidth=2)
    poly!(ax, @lift(Rect2f(0, $paddle_left_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color=:lightcyan)
    poly!(ax, @lift(Rect2f(COURT_WIDTH - PADDLE_WIDTH, $paddle_right_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color=:lightcyan)
    score_text_obs = @lift "$($score_left) - $($score_right)"
    text!(ax, score_text_obs, position=Point2f(COURT_WIDTH/2f0, COURT_HEIGHT-20f0), fontsize=40, color=:white, align=(:center, :top))
    text!(ax, game_message, position=Point2f(COURT_WIDTH/2f0, 30f0), fontsize=30, color=:yellow, align=(:center, :bottom))

    function reset_ball()
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

    function update_visualization_data!()
        color_image_data_local = Observable(zeros(RGB{Float32}, NX, NY))[] # Get the actual array
        for j_vis in 1:NY
            for i_vis in 1:NX
                idx_sim_grid = IX(i_vis + 1, j_vis + 1) # Map vis grid (1-NX) to sim grid (2 to NX+1)
                r = clamp(fluid_color_R[idx_sim_grid], 0.0f0, 1.0f0)
                g = clamp(fluid_color_G[idx_sim_grid], 0.0f0, 1.0f0)
                b = clamp(fluid_color_B[idx_sim_grid], 0.0f0, 1.0f0)
                color_image_data_local[i_vis, j_vis] = RGB{Float32}(r, g, b)
            end
        end
        fluid_color_image_obs[] = color_image_data_local # Trigger update

        num_arrows = length(arrow_pos_obs[])
        new_arrow_dirs_local = Vector{Vec2f}(undef, num_arrows)
        vis_idx = 1
        for j_glob in 1:ARROW_SUBSAMPLE:NY # Loop over visualization grid points
            for i_glob in 1:ARROW_SUBSAMPLE:NX
                if vis_idx > num_arrows; break; end
                idx_sim_grid = IX(i_glob + 1, j_glob + 1) # Map to simulation grid index
                qxx_val = fluid_Qxx[idx_sim_grid]
                qxy_val = fluid_Qxy[idx_sim_grid]
                director_x = 0.0f0; director_y = 0.0f0; epsilon = 1e-7
                
                # Director from Q tensor (standard way for visualization)
                # n_x = cos(phi), n_y = sin(phi)
                # Q_xx = S * (cos^2(phi) - 1/2)  => S * ( (cos(2phi)+1)/2 - 1/2 ) = S/2 * cos(2phi) -- if S is S0 order param
                # Q_xy = S * cos(phi)sin(phi)    => S/2 * sin(2phi)
                # So, if Q_xx = A, Q_xy = B, then 2A/S = cos(2phi), 2B/S = sin(2phi)
                # We can use (Q_xx, Q_xy) as a vector proportional to (cos(2phi), sin(2phi))
                # Or, eigenvector corresponding to the largest eigenvalue of Q.
                # Largest eigenvalue is S/2 (using S0 as eigenvalue related to order parameter)
                # Eigenvector (nx, ny) satisfies:
                # Qxx*nx + Qxy*ny = (S/2)*nx
                # Qxy*nx - Qxx*ny = (S/2)*ny  (since Qyy = -Qxx)
                # One common director choice: (Qxy, S/2 - Qxx) or (S/2 + Qxx, Qxy)
                # Or, if Qxx = S/2 cos(2θ), Qxy = S/2 sin(2θ), then θ is angle of director.
                # We can also just plot (Q_xx, Q_xy) and normalize for direction of n d n (dyadic product tensor).
                # Or more simply, if Qxx = S (cos^2θ - sin^2θ)/2 and Qxy = S sinθcosθ.
                # Let's stick to a common visualization: principal eigenvector direction.
                # For Q = [[qxx, qxy], [qxy, -qxx]], eigenvalues are +/- sqrt(qxx^2 + qxy^2)
                lambda_eigen = sqrt(qxx_val^2 + qxy_val^2)
                if lambda_eigen < epsilon # Isotropic or very small order
                    director_x = 1.0f0; director_y = 0.0f0; # Default
                else
                    # Eigenvector for positive eigenvalue lambda_eigen:
                    # (qxx - lambda_eigen)vx + qxy*vy = 0
                    # qxy*vx + (-qxx - lambda_eigen)vy = 0
                    # A non-trivial solution for (A,B) in (qxx-lambda)A + qxyB = 0 is A=qxy, B=-(qxx-lambda)
                    director_x = qxy_val 
                    director_y = lambda_eigen - qxx_val 
                    # Alternative: director_x = lambda_eigen + qxx_val, director_y = qxy_val
                    # Both should give orthogonal eigenvectors. We need the one for positive lambda.
                    # Let's test: if qxx > 0, qxy = 0, lambda = qxx. director_x=0, director_y=0 (bad choice here)
                    # if qxx > 0, qxy = 0, lambda = qxx. Then Qxx = S/2. director = (1,0) or (0,1) based on definition.
                    # For Qxx = S/2, Qxy = 0, dir is (1,0). (cos(2a)=1, sin(2a)=0 => 2a=0 => a=0). (nx=1,ny=0)
                    # My eigenvector (0, S/2 - S/2) = (0,0) is not good.
                    # Let's use the standard: if Q = S/2 {{cos(2f), sin(2f)},{sin(2f), -cos(2f)}}, director is (cos f, sin f)
                    # So, effectively angle is 0.5*atan(Qxy,Qxx)
                    angle_double = atan(qxy_val, qxx_val) # Gives angle of (Qxx, Qxy) vector, which is 2*phi
                    angle_director = 0.5f0 * angle_double
                    director_x = cos(angle_director)
                    director_y = sin(angle_director)
                end
                
                norm_dir = sqrt(director_x^2 + director_y^2)
                if norm_dir > epsilon
                    new_arrow_dirs_local[vis_idx] = Vec2f(director_x/norm_dir, director_y/norm_dir)
                else
                    new_arrow_dirs_local[vis_idx] = Vec2f(1.0f0, 0.0f0) # Default for isotropic
                end
                vis_idx += 1
            end
            if vis_idx > num_arrows break end
        end
        # Assign only if all arrows were processed, or handle partial update carefully.
        if num_arrows > 0 && vis_idx -1 == num_arrows 
             arrow_dir_obs[] = new_arrow_dirs_local
        elseif num_arrows == 0
             arrow_dir_obs[] = Vec2f[] 
        end
    end


    on(events(fig).tick) do _
        if !game_active[]; return Consume(false); end
        current_time = time(); dt_game = Float32(clamp(current_time - last_update_time[], 0.001, 0.05)); last_update_time[] = current_time
        keys = events(fig).keyboardstate; bp = ball_pos[]; bv = ball_vel[]; cbs = current_ball_speed[]; 
        p_left_y_val = paddle_left_y[]; p_right_y_val = paddle_right_y[]; prev_ball_pos[] = bp 

        paddle_delta = PADDLE_SPEED * dt_game
        if Keyboard.w in keys; paddle_left_y[] += paddle_delta; end; if Keyboard.s in keys; paddle_left_y[] -= paddle_delta; end
        paddle_left_y[] = clamp(paddle_left_y[], 0.0f0, COURT_HEIGHT - PADDLE_HEIGHT)
        if Keyboard.up in keys; paddle_right_y[] += paddle_delta; end; if Keyboard.down in keys; paddle_right_y[] -= paddle_delta; end
        paddle_right_y[] = clamp(paddle_right_y[], 0.0f0, COURT_HEIGHT - PADDLE_HEIGHT)

        direct_pull_force = Vec2f(0.0f0)

        is_p1_pushing_this_frame = (Keyboard.d in keys || (Keyboard.space in keys && served_by[] == :p1)) && serve_state[] == :playing
        is_p2_pushing_this_frame = (Keyboard.left in keys || (Keyboard.enter in keys && served_by[] == :p2)) && serve_state[] == :playing

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
        steps_taken = 0; max_steps_per_frame = 5 # Max fluid steps per game frame
        while fluid_time_accumulator[] >= DT && steps_taken < max_steps_per_frame
            # Apply Q and Color modifications from paddle (if keys are held)
            if is_p1_pushing_this_frame
                apply_paddle_jet_effect!(1, p_left_y_val); 
            end
            if is_p2_pushing_this_frame
                apply_paddle_jet_effect!(2, p_right_y_val); 
            end

            # Pass paddle state to fluid_step for body force application
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

            new_bv = bv + total_force_on_ball * dt_game # Assume ball mass = 1 for F=ma -> a=F
            new_bp = bp + new_bv * dt_game

            if new_bp[2] < BALL_SIZE/2f0 || new_bp[2] > COURT_HEIGHT-BALL_SIZE/2f0
                 new_bp = Point2f(new_bp[1], clamp(new_bp[2], BALL_SIZE/2f0+0.1f0, COURT_HEIGHT-BALL_SIZE/2f0-0.1f0))
                 new_bv = Vec2f(new_bv[1], -new_bv[2]) # Reflect y-velocity
            end

            paddle_left_rect=Rect2f(0f0, p_left_y_val, PADDLE_WIDTH, PADDLE_HEIGHT) # Use current paddle_left_y.val
            paddle_right_rect=Rect2f(COURT_WIDTH-PADDLE_WIDTH, p_right_y_val, PADDLE_WIDTH, PADDLE_HEIGHT)
            ball_rect_predict = Rect2f(new_bp[1]-BALL_SIZE/2f0, new_bp[2]-BALL_SIZE/2f0, BALL_SIZE, BALL_SIZE)

            if new_bv[1]<0f0 && rect_overlaps(paddle_left_rect, ball_rect_predict)
                 paddle_center_y_val=p_left_y_val+PADDLE_HEIGHT/2f0
                 hit_offset=clamp((new_bp[2]-paddle_center_y_val)/(PADDLE_HEIGHT/2f0), -1.0f0, 1.0f0)
                 bounce_angle=hit_offset*(pi/3f0) # Max bounce angle pi/3
                 cbs += BALL_SPEED_INCREASE
                 new_bv=Vec2f(cos(bounce_angle), sin(bounce_angle))*cbs
                 new_bp=Point2f(PADDLE_WIDTH+BALL_SIZE/2f0+0.1f0, new_bp[2]) # Move ball slightly out of paddle
            elseif new_bv[1]>0f0 && rect_overlaps(paddle_right_rect, ball_rect_predict)
                 paddle_center_y_val=p_right_y_val+PADDLE_HEIGHT/2f0
                 hit_offset=clamp((new_bp[2]-paddle_center_y_val)/(PADDLE_HEIGHT/2f0), -1.0f0, 1.0f0)
                 bounce_angle=hit_offset*(pi/3f0)
                 cbs += BALL_SPEED_INCREASE
                 world_angle=pi-bounce_angle # Angle from positive x-axis for right paddle
                 new_bv=Vec2f(cos(world_angle), sin(world_angle))*cbs
                 new_bp=Point2f(COURT_WIDTH-PADDLE_WIDTH-BALL_SIZE/2f0-0.1f0, new_bp[2])
             end
            ball_pos[] = new_bp; ball_vel[] = new_bv; current_ball_speed[] = norm(new_bv) # Update speed based on new velocity

            scored = false; final_pos = ball_pos[] # Use updated ball_pos
            if final_pos[1] < BALL_SIZE/2f0; score_right[]+=1; served_by[]=:p1; scored=true; end
            if final_pos[1] > COURT_WIDTH-BALL_SIZE/2f0; score_left[]+=1; served_by[]=:p2; scored=true; end
            if scored
                 if score_left[] >= SCORE_LIMIT || score_right[] >= SCORE_LIMIT
                     game_active[] = false; winner = score_left[] >= SCORE_LIMIT ? "Left" : "Right"; game_message[] = "$winner Player Wins!\nPress R to Restart"
                 else; reset_ball(); end
            end
        elseif serve_state[] == :p1_serve
            target_y=p_left_y_val+PADDLE_HEIGHT/2f0; new_pos=Point2f(bp[1], clamp(target_y, BALL_SIZE/2f0, COURT_HEIGHT-BALL_SIZE/2f0)); ball_pos[]=new_pos; prev_ball_pos[]=new_pos
        elseif serve_state[] == :p2_serve
            target_y=p_right_y_val+PADDLE_HEIGHT/2f0; new_pos=Point2f(bp[1], clamp(target_y, BALL_SIZE/2f0, COURT_HEIGHT-BALL_SIZE/2f0)); ball_pos[]=new_pos; prev_ball_pos[]=new_pos
        end
        return Consume(false)
    end

    on(events(fig).keyboardbutton) do event
        is_serve_key_p1 = event.key == Keyboard.space || event.key == Keyboard.d
        is_serve_key_p2 = event.key == Keyboard.enter || event.key == Keyboard.left
        current_state = serve_state[]

        if game_active[] && event.action == Keyboard.press && current_state != :playing
            serve_triggered = false; base_angle = 0f0
            if current_state == :p1_serve && is_serve_key_p1; serve_triggered=true; base_angle=0f0; served_by[]=:p1; end
            if current_state == :p2_serve && is_serve_key_p2; serve_triggered=true; base_angle=pi; served_by[]=:p2; end

            if serve_triggered
                 angle_offset = (rand(Float32) * (pi/3f0)) - (pi/6f0) # +/- 30 degrees
                 angle=base_angle + angle_offset
                 serve_speed=current_ball_speed[] # Use current_ball_speed observable's value
                 ball_vel[]=Vec2f(cos(angle), sin(angle))*serve_speed
                 serve_state[] = :playing; game_message[] = ""; return Consume(true)
            end
        end
        if !game_active[] && event.key == Keyboard.r && event.action == Keyboard.press # Restart game
            score_left[]=0; score_right[]=0; served_by[]=rand([:p1, :p2]); reset_ball(); game_active[]=true
            # Reset fluid state completely
            fill!(fluid_vx, 0.0f0); fill!(fluid_vy, 0.0f0); fill!(fluid_vx0, 0.0f0); fill!(fluid_vy0, 0.0f0)
            fill!(fluid_p, 0.0f0); fill!(fluid_div, 0.0f0)
            rand_scale_init=0.01f0
            for i in eachindex(fluid_Qxx)
                fluid_Qxx[i]=(rand(Float32)-0.5f0)*2.0f0*rand_scale_init
                fluid_Qxy[i]=(rand(Float32)-0.5f0)*2.0f0*rand_scale_init
            end
            set_boundary_scalar!(fluid_Qxx); set_boundary_scalar!(fluid_Qxy)
            fill!(fluid_Hxx,0.0f0); fill!(fluid_Hxy,0.0f0)
            fill!(fluid_div_sigma_active_x,0.0f0); fill!(fluid_div_sigma_active_y,0.0f0)
            fill!(fluid_div_sigma_elastic_x,0.0f0); fill!(fluid_div_sigma_elastic_y,0.0f0)
            fill!(fluid_color_R, 0.0f0); fill!(fluid_color_G, 0.0f0); fill!(fluid_color_B, 0.0f0)
            fill!(fluid_color_R0, 0.0f0); fill!(fluid_color_G0, 0.0f0); fill!(fluid_color_B0, 0.0f0)
            set_boundary_scalar!(fluid_color_R); set_boundary_scalar!(fluid_color_G); set_boundary_scalar!(fluid_color_B)
            update_visualization_data!(); return Consume(true)
        end
        return Consume(false)
    end
    update_visualization_data!() # Initial draw
    display(fig); return fig
end

run()