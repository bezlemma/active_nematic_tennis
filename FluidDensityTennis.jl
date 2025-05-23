using GLMakie
using Random
using GeometryBasics: Rect2f, Point2f, Vec2f
using LinearAlgebra # For normalize, norm

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

# --- Fluid Simulation Setup ---
const FLUID_ENABLED = true
const FLUID_NX = 128
const FLUID_NY = 96
const FLUID_DT = 0.01f0
const FLUID_VISC = 0.00005f0
const FLUID_DIFF = 0.00005f0
const FLUID_SOLVER_ITER = 12
const FLUID_FORCE_SCALE = 400.0 # Passive fluid push from paddles (unused now?)
const BALL_DRAG_COEFF = 0.05f0 # How much ball is affected by general fluid motion
const vel_scale = 10000.0f0 # Scales interpolated fluid velocity for drag effect

# --- Push/Pull Constants ---
const PUSH_VELOCITY = 450.0f0   # Strength of the outward "push" fluid effect
const PUSH_DENSITY = 400.0f0   # Density added by pushing 
const PUSH_PLUME_WIDTH_PX = 8.0f0 #
const PULL_FORCE_STRENGTH = 2000.0f0 # Direct force applied to the ball when pulling
const PULL_EFFECT_RADIUS_SQ = (PADDLE_HEIGHT * 1.2f0)^2 # Squared radius for pull visualization
const PULL_DENSITY_ADD = 1.0f0 # Density added near paddle center during pull 

# --- Fluid Constants ---
const FLUID_DENS_MAX = 2000.0f0
const DENSITY_DECAY_RATE = 0.1f0
const FLUID_DX = COURT_WIDTH / FLUID_NX
const FLUID_DY = COURT_HEIGHT / FLUID_NY
const FLUID_SIZE = (FLUID_NX + 2) * (FLUID_NY + 2)

# --- Global Fluid Arrays ---
global fluid_vx::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_vy::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_vx0::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_vy0::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_dens::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_dens0::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_p::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_div::Vector{Float32} = zeros(Float32, FLUID_SIZE)
global fluid_dens_obs = Observable(zeros(Float32, FLUID_NX, FLUID_NY))

@inline IX(i, j) = clamp(i, 1, FLUID_NX + 2) + (clamp(j, 1, FLUID_NY + 2) - 1) * (FLUID_NX + 2)

# --- Fluid Simulation 
function set_boundary!(b::Int, x::Vector{Float32})
    N = FLUID_NX; M = FLUID_NY
    for i in 1:(N + 2); x[IX(i, 1)] = b == 2 ? -x[IX(i, 2)] : x[IX(i, 2)]; x[IX(i, M+2)] = b == 2 ? -x[IX(i, M+1)] : x[IX(i, M+1)]; end
    for j in 1:(M + 2); x[IX(1, j)] = b == 1 ? -x[IX(2, j)] : x[IX(2, j)]; x[IX(N+2, j)] = b == 1 ? -x[IX(N+1, j)] : x[IX(N+1, j)]; end
    x[IX(1, 1)] = 0.5f0 * (x[IX(2, 1)] + x[IX(1, 2)]); x[IX(1, M+2)] = 0.5f0 * (x[IX(2, M+2)] + x[IX(1, M+1)])
    x[IX(N+2, 1)] = 0.5f0 * (x[IX(N+1, 1)] + x[IX(N+2, 2)]); x[IX(N+2, M+2)] = 0.5f0 * (x[IX(N+1, M+2)] + x[IX(N+2, M+1)])
end

function linear_solve!(b::Int, x::Vector{Float32}, x0::Vector{Float32}, a::Float32, c::Float32)
    c_inv = 1.0f0 / c
    for k in 1:FLUID_SOLVER_ITER
        for j in 2:(FLUID_NY + 1); for i in 2:(FLUID_NX + 1)
            idx = IX(i, j)
            neighbors_sum = x[IX(i-1, j)] + x[IX(i+1, j)] + x[IX(i, j-1)] + x[IX(i, j+1)]
            x[idx] = (x0[idx] + a * neighbors_sum) * c_inv
        end; end
        set_boundary!(b, x)
    end
end

function diffuse!(b::Int, x::Vector{Float32}, x0::Vector{Float32}, diff_rate::Float32, dt::Float32)
    if diff_rate == 0.0f0 return end
    a = dt * diff_rate * FLUID_NX * FLUID_NY
    linear_solve!(b, x, x0, a, 1.0f0 + 4.0f0 * a)
end

function advect!(b::Int, d::Vector{Float32}, d0::Vector{Float32}, velX::Vector{Float32}, velY::Vector{Float32}, dt::Float32)
    dtx = dt * FLUID_NX; dty = dt * FLUID_NY
    for j in 2:(FLUID_NY + 1); for i in 2:(FLUID_NX + 1)
        idx = IX(i, j)
        x = Float32(i) - dtx * velX[idx]; y = Float32(j) - dty * velY[idx]
        x = clamp(x, 1.5f0, FLUID_NX + 0.5f0); y = clamp(y, 1.5f0, FLUID_NY + 0.5f0)
        i0 = floor(Int, x); i1 = i0 + 1; j0 = floor(Int, y); j1 = j0 + 1
        s1 = x - i0; s0 = 1.0f0 - s1; t1 = y - j0; t0 = 1.0f0 - t1
        d[idx] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                 s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)])
    end; end
    set_boundary!(b, d)
end

function project!(velX::Vector{Float32}, velY::Vector{Float32}, p::Vector{Float32}, div::Vector{Float32})
    h_x = 1.0f0; h_y = 1.0f0
    for j in 2:(FLUID_NY + 1); for i in 2:(FLUID_NX + 1)
        div[IX(i, j)] = -0.5f0 * ((velX[IX(i+1, j)] - velX[IX(i-1, j)]) * h_x +
                                  (velY[IX(i, j+1)] - velY[IX(i, j-1)]) * h_y)
        p[IX(i, j)] = 0.0f0
    end; end
    set_boundary!(0, div); set_boundary!(0, p)
    linear_solve!(0, p, div, 1.0f0, 4.0f0)
    for j in 2:(FLUID_NY + 1); for i in 2:(FLUID_NX + 1)
        velX[IX(i, j)] -= 0.5f0 * (p[IX(i+1, j)] - p[IX(i-1, j)]) / h_x
        velY[IX(i, j)] -= 0.5f0 * (p[IX(i, j+1)] - p[IX(i, j-1)]) / h_y
    end; end
    set_boundary!(1, velX); set_boundary!(2, velY)
end

# --- Main Fluid Simulation Step
function fluid_step!()
    if !FLUID_ENABLED return end
    dt = FLUID_DT; visc = FLUID_VISC; diff = FLUID_DIFF

    vx0_temp = copy(fluid_vx); vy0_temp = copy(fluid_vy); dens0_temp = copy(fluid_dens)
    # --- Velocity Step ---
    diffuse!(1, fluid_vx, vx0_temp, visc, dt)
    diffuse!(2, fluid_vy, vy0_temp, visc, dt)
    project!(fluid_vx, fluid_vy, fluid_p, fluid_div) 
    # Advect Velocity
    vx_pre_advect = copy(fluid_vx); vy_pre_advect = copy(fluid_vy)
    advect!(1, fluid_vx, vx_pre_advect, vx_pre_advect, vy_pre_advect, dt)
    advect!(2, fluid_vy, vy_pre_advect, vx_pre_advect, vy_pre_advect, dt)
    project!(fluid_vx, fluid_vy, fluid_p, fluid_div) 

    # --- Density Step ---
    diffuse!(0, fluid_dens, dens0_temp, diff, dt) 
    dens_pre_advect = copy(fluid_dens)
    advect!(0, fluid_dens, dens_pre_advect, fluid_vx, fluid_vy, dt)
end

# --- Interaction Functions ---

function world_to_grid(pos::Point2f)
    gx = (pos[1] / COURT_WIDTH) * FLUID_NX + 1.5f0 
    gy = (pos[2] / COURT_HEIGHT) * FLUID_NY + 1.5f0 
    return gx, gy
end

# Interpolate fluid velocity
function get_fluid_velocity_at(pos::Point2f)::Vec2f
    if !FLUID_ENABLED return Vec2f(0.0f0) end
    gx, gy = world_to_grid(pos)
    # Clamp to the valid *sampling* range (1.5 to N+0.5 for correct interpolation)
    gx = clamp(gx, 1.5f0, FLUID_NX + 0.5f0); gy = clamp(gy, 1.5f0, FLUID_NY + 0.5f0)
    i0 = floor(Int, gx); i1 = i0 + 1; j0 = floor(Int, gy); j1 = j0 + 1

    s1 = gx - i0; s0 = 1.0f0 - s1; t1 = gy - j0; t0 = 1.0f0 - t1
    vx = s0 * (t0 * fluid_vx[IX(i0, j0)] + t1 * fluid_vx[IX(i0, j1)]) +
         s1 * (t0 * fluid_vx[IX(i1, j0)] + t1 * fluid_vx[IX(i1, j1)])
    vy = s0 * (t0 * fluid_vy[IX(i0, j0)] + t1 * fluid_vy[IX(i0, j1)]) +
         s1 * (t0 * fluid_vy[IX(i1, j0)] + t1 * fluid_vy[IX(i1, j1)])

    return Vec2f(vx, vy) * vel_scale
end


# --- Helper Function for Rectangle Overlap
function rect_overlaps(r1::Rect2f, r2::Rect2f)
    x_overlap = (r1.origin[1] < r2.origin[1] + r2.widths[1]) && (r1.origin[1] + r1.widths[1] > r2.origin[1])
    y_overlap = (r1.origin[2] < r2.origin[2] + r2.widths[2]) && (r1.origin[2] + r1.widths[2] > r2.origin[2])
    return x_overlap && y_overlap
end






function run()
    fig = Figure(size = (COURT_WIDTH, COURT_HEIGHT), backgroundcolor = :black, figure_padding = 0)
    fig.scene.backgroundcolor[] = RGBf(0.0, 0.0, 0.0)
    ax = Axis(fig[1, 1], aspect = DataAspect(), limits = (0, COURT_WIDTH, 0, COURT_HEIGHT), backgroundcolor = :black)
    hidedecorations!(ax); hidespines!(ax)

    if FLUID_ENABLED
        global fluid_vx = zeros(Float32, FLUID_SIZE); global fluid_vy = zeros(Float32, FLUID_SIZE)
        global fluid_vx0 = zeros(Float32, FLUID_SIZE); global fluid_vy0 = zeros(Float32, FLUID_SIZE)
        global fluid_dens = zeros(Float32, FLUID_SIZE); global fluid_dens0 = zeros(Float32, FLUID_SIZE)
        global fluid_p = zeros(Float32, FLUID_SIZE); global fluid_div = zeros(Float32, FLUID_SIZE)
        global fluid_dens_obs = Observable(zeros(Float32, FLUID_NX, FLUID_NY))
    end

    ball_pos = Observable(Point2f(COURT_WIDTH / 2f0, COURT_HEIGHT / 2f0))
    prev_ball_pos = Observable(Point2f(COURT_WIDTH / 2f0, COURT_HEIGHT / 2f0))
    ball_vel = Observable(Vec2f(0f0, 0f0))
    current_ball_speed = Observable(BALL_SPEED_INIT)
    paddle_left_y = Observable(COURT_HEIGHT / 2f0 - PADDLE_HEIGHT / 2f0)
    paddle_right_y = Observable(COURT_HEIGHT / 2f0 - PADDLE_HEIGHT / 2f0)
    score_left = Observable(0); score_right = Observable(0)
    game_active = Observable(true)
    game_message = Observable("Press Serve Keys to Start")
    serve_state = Observable(:p1_serve)
    served_by = Observable(:p1)
    last_update_time = Ref(time())
    fluid_time_accumulator = Ref(0.0f0)

    if FLUID_ENABLED
        fluid_x_range = LinRange(0 + FLUID_DX/2, COURT_WIDTH - FLUID_DX/2, FLUID_NX)
        fluid_y_range = LinRange(0 + FLUID_DY/2, COURT_HEIGHT - FLUID_DY/2, FLUID_NY)
        heatmap!(ax, fluid_x_range, fluid_y_range, fluid_dens_obs,
                 colormap = :turbo, interpolate=true, colorrange = (0.0f0, 10.0f0),)
    end
    ball_visual_size = BALL_SIZE * 0.9f0
    ball_rect_obs = @lift Rect2f($ball_pos[1] - ball_visual_size/2f0, $ball_pos[2] - ball_visual_size/2f0, ball_visual_size, ball_visual_size)
    poly!(ax, ball_rect_obs, color = :yellow, strokecolor=:orange, strokewidth=2)
    poly!(ax, @lift(Rect2f(0, $paddle_left_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color = :white)
    poly!(ax, @lift(Rect2f(COURT_WIDTH - PADDLE_WIDTH, $paddle_right_y, PADDLE_WIDTH, PADDLE_HEIGHT)), color = :white)
    lines!(ax, [COURT_WIDTH/2f0, COURT_WIDTH/2f0], [0, COURT_HEIGHT], color=:white, linestyle=:dash, linewidth=2)
    score_text_obs = @lift "$($score_left) - $($score_right)"
    text!(ax, score_text_obs, position = Point2f(COURT_WIDTH/2f0, COURT_HEIGHT - 20f0), fontsize = 40, color = :white, align = (:center, :top), space = :pixel)
    text!(ax, game_message, position = Point2f(COURT_WIDTH/2f0, 30f0), fontsize = 30, color = :yellow, align = (:center, :bottom), space = :pixel)

    function reset_ball()
        current_ball_speed[] = BALL_SPEED_INIT
        ball_vel[] = Vec2f(0f0, 0f0)
        local_paddle_y::Float32 = 0.0f0
        ball_x::Float32 = 0.0f0
        if served_by[] == :p1
            local_paddle_y = paddle_left_y[]
            ball_x = PADDLE_WIDTH + BALL_SIZE / 2f0
            serve_state[] = :p1_serve
            game_message[] = "P1: Use A/D or Space to Serve/Push/Pull"
        else
            local_paddle_y = paddle_right_y[]
            ball_x = COURT_WIDTH - PADDLE_WIDTH - BALL_SIZE / 2f0
            serve_state[] = :p2_serve
            game_message[] = "P2: Use ←/→ or Enter to Serve/Push/Pull"
        end
        ball_y = local_paddle_y + PADDLE_HEIGHT / 2f0
        new_pos = Point2f(ball_x, clamp(ball_y, BALL_SIZE / 2f0, COURT_HEIGHT - BALL_SIZE / 2f0))
        ball_pos[] = new_pos
        prev_ball_pos[] = new_pos
    end
    reset_ball()

    # --- Game Loop ---
    on(events(fig).tick) do _
        if !game_active[] return Consume(false) end

        current_time = time()
        dt = Float32(clamp(current_time - last_update_time[], 0.001, 0.05))
        last_update_time[] = current_time

        keys = events(fig).keyboardstate
        bp = ball_pos[]
        bv = ball_vel[]
        cbs = current_ball_speed[]
        p_left_y = paddle_left_y[]
        p_right_y = paddle_right_y[]
        prev_ball_pos[] = bp

        # --- Fluid Simulation Update ---
        if FLUID_ENABLED
            fluid_time_accumulator[] += dt
            direct_pull_force = Vec2f(0.0f0) # Force applied directly to the ball

            push_vel_dt = PUSH_VELOCITY * dt
            push_dens_dt = PUSH_DENSITY * dt
            pull_dens_dt = PULL_DENSITY_ADD * dt

            # --- Left Paddle Actions ---
            if Keyboard.d in keys # PUSH
                paddle_center_y = p_left_y + PADDLE_HEIGHT / 2f0
                target_x_world = PADDLE_WIDTH + FLUID_DX * 0.5f0
                pc_gx, pc_gy = world_to_grid(Point2f(target_x_world, paddle_center_y))
                center_gxi = clamp(floor(Int, pc_gx), 2, FLUID_NX + 1) # Get grid X index
                center_gyi = clamp(floor(Int, pc_gy), 2, FLUID_NY + 1) # Get grid Y index of center

                plume_half_width_grid = ceil(Int, (PUSH_PLUME_WIDTH_PX / FLUID_DY) / 2.0f0)

                gy_start = clamp(center_gyi - plume_half_width_grid, 2, FLUID_NY + 1)
                gy_end = clamp(center_gyi + plume_half_width_grid, 2, FLUID_NY + 1)
                for gyi in gy_start:gy_end
                    idx = IX(center_gxi, gyi)
                    fluid_vx[idx] += push_vel_dt   # Add velocity outwards
                    fluid_dens[idx] += push_dens_dt # Add density
                end

            elseif Keyboard.a in keys && serve_state[] == :playing # PULL
                paddle_center_y = p_left_y + PADDLE_HEIGHT / 2f0
                paddle_center_world = Point2f(PADDLE_WIDTH / 2f0, paddle_center_y) # Target center
                dir_to_paddle = paddle_center_world - bp
                dist_sq = sum(dir_to_paddle.^2)
                if dist_sq > 1e-4
                    force_dir = normalize(dir_to_paddle)
                    strength_factor = 1.0f0 # Constant strength for now
                    direct_pull_force += force_dir * PULL_FORCE_STRENGTH * strength_factor
                end

                pc_gx, pc_gy = world_to_grid(paddle_center_world)
                gxi_c = clamp(floor(Int, pc_gx), 2, FLUID_NX + 1)
                gyi_c = clamp(floor(Int, pc_gy), 2, FLUID_NY + 1)
                idx_c = IX(gxi_c, gyi_c)
                fluid_dens[idx_c] += pull_dens_dt * 5

                radius_grid_sq_x = (PULL_EFFECT_RADIUS_SQ / FLUID_DX^2)
                radius_grid_sq_y = (PULL_EFFECT_RADIUS_SQ / FLUID_DY^2)
                min_ix = clamp(floor(Int, pc_gx - sqrt(radius_grid_sq_x)), 2, FLUID_NX + 1)
                max_ix = clamp(ceil(Int, pc_gx + sqrt(radius_grid_sq_x)), 2, FLUID_NX + 1)
                min_iy = clamp(floor(Int, pc_gy - sqrt(radius_grid_sq_y)), 2, FLUID_NY + 1)
                max_iy = clamp(ceil(Int, pc_gy + sqrt(radius_grid_sq_y)), 2, FLUID_NY + 1)
                 for ix_r in min_ix:max_ix
                     for iy_r in min_iy:max_iy
                         if (ix_r - pc_gx)^2 + (iy_r - pc_gy)^2 * (FLUID_DX/FLUID_DY)^2 < radius_grid_sq_x
                             fluid_dens[IX(ix_r, iy_r)] += pull_dens_dt
                         end
                     end
                 end
            end

            if Keyboard.left in keys # PUSH
                paddle_center_y = p_right_y + PADDLE_HEIGHT / 2f0
              
                target_x_world = COURT_WIDTH - PADDLE_WIDTH - FLUID_DX * 0.5f0
                pc_gx, pc_gy = world_to_grid(Point2f(target_x_world, paddle_center_y))
                center_gxi = clamp(floor(Int, pc_gx), 2, FLUID_NX + 1) # Get grid X index
                center_gyi = clamp(floor(Int, pc_gy), 2, FLUID_NY + 1) # Get grid Y index of center

                plume_half_width_grid = ceil(Int, (PUSH_PLUME_WIDTH_PX / FLUID_DY) / 2.0f0)

                gy_start = clamp(center_gyi - plume_half_width_grid, 2, FLUID_NY + 1)
                gy_end = clamp(center_gyi + plume_half_width_grid, 2, FLUID_NY + 1)

                for gyi in gy_start:gy_end
                    idx = IX(center_gxi, gyi)
                    fluid_vx[idx] -= push_vel_dt  # Add velocity outwards (negative x)
                    fluid_dens[idx] += push_dens_dt
                end

            elseif Keyboard.right in keys && serve_state[] == :playing # PULL 
                paddle_center_y = p_right_y + PADDLE_HEIGHT / 2f0
                paddle_center_world = Point2f(COURT_WIDTH - PADDLE_WIDTH / 2f0, paddle_center_y)
                dir_to_paddle = paddle_center_world - bp
                dist_sq = sum(dir_to_paddle.^2)
                 if dist_sq > 1e-4
                    force_dir = normalize(dir_to_paddle)
                    strength_factor = 1.0f0
                    direct_pull_force += force_dir * PULL_FORCE_STRENGTH * strength_factor
                 end

                 pc_gx, pc_gy = world_to_grid(paddle_center_world)
                 gxi_c = clamp(floor(Int, pc_gx), 2, FLUID_NX + 1)
                 gyi_c = clamp(floor(Int, pc_gy), 2, FLUID_NY + 1)
                 idx_c = IX(gxi_c, gyi_c)
                 fluid_dens[idx_c] += pull_dens_dt * 5

                 radius_grid_sq_x = (PULL_EFFECT_RADIUS_SQ / FLUID_DX^2)
                 radius_grid_sq_y = (PULL_EFFECT_RADIUS_SQ / FLUID_DY^2)
                 min_ix = clamp(floor(Int, pc_gx - sqrt(radius_grid_sq_x)), 2, FLUID_NX + 1)
                 max_ix = clamp(ceil(Int, pc_gx + sqrt(radius_grid_sq_x)), 2, FLUID_NX + 1)
                 min_iy = clamp(floor(Int, pc_gy - sqrt(radius_grid_sq_y)), 2, FLUID_NY + 1)
                 max_iy = clamp(ceil(Int, pc_gy + sqrt(radius_grid_sq_y)), 2, FLUID_NY + 1)
                 for ix_r in min_ix:max_ix
                     for iy_r in min_iy:max_iy
                         if (ix_r - pc_gx)^2 + (iy_r - pc_gy)^2 * (FLUID_DX/FLUID_DY)^2 < radius_grid_sq_x
                             fluid_dens[IX(ix_r, iy_r)] += pull_dens_dt
                         end
                     end
                 end
            end
            # --- Fluid Simulation Step ---
            while fluid_time_accumulator[] >= FLUID_DT
                fluid_step!() # Applies diffusion, advection, projection
                fluid_time_accumulator[] -= FLUID_DT
            end

            # --- Density Decay & Observable Update 
            decay_factor = 1.0f0 - DENSITY_DECAY_RATE * dt
            if decay_factor < 0.999f0 # Avoid unnecessary multiplication if decay is tiny
                 @. fluid_dens = fluid_dens * decay_factor
                 # Optional: Clear very low densities to prevent creep
                 fluid_dens[fluid_dens .< 0.01f0] .= 0.0f0
            end
            clamp!(fluid_dens, 0.0f0, FLUID_DENS_MAX) # Clamp density

            # Update observable ONCE per frame *after* all simulation steps for the frame
            density_view = reshape(view(fluid_dens, [IX(i,j) for i=2:FLUID_NX+1, j=2:FLUID_NY+1]), (FLUID_NX, FLUID_NY))
            fluid_dens_obs[] = density_view
        end 

        paddle_delta = PADDLE_SPEED * dt
        if Keyboard.w in keys; paddle_left_y[] += paddle_delta; end
        if Keyboard.s in keys; paddle_left_y[] -= paddle_delta; end
        paddle_left_y[] = clamp(paddle_left_y[], 0.0f0, COURT_HEIGHT - PADDLE_HEIGHT)
        if Keyboard.up in keys; paddle_right_y[] += paddle_delta; end
        if Keyboard.down in keys; paddle_right_y[] -= paddle_delta; end
        paddle_right_y[] = clamp(paddle_right_y[], 0.0f0, COURT_HEIGHT - PADDLE_HEIGHT)

        # --- Ball Logic ---
        if serve_state[] == :playing
            # Calculate total force on the ball
            total_force = Vec2f(0.0f0)

            if FLUID_ENABLED
                fluid_vel_at_ball = get_fluid_velocity_at(bp) # Interpolated fluid velocity
                relative_vel = fluid_vel_at_ball - bv # Fluid vel relative to ball
                drag_force = BALL_DRAG_COEFF * relative_vel
                total_force += drag_force
            end

            new_bv = bv + total_force * dt # Update velocity first
            new_bp = bp + new_bv * dt     # Update position using *new* velocity
            collided_y = false
            if new_bp[2] - BALL_SIZE/2f0 <= 0f0
                new_bp = Point2f(new_bp[1], BALL_SIZE/2f0 + 0.1f0)
                new_bv = Vec2f(new_bv[1], -new_bv[2]) # Reflect Y velocity
                collided_y = true
            elseif new_bp[2] + BALL_SIZE/2f0 >= COURT_HEIGHT
                new_bp = Point2f(new_bp[1], COURT_HEIGHT - BALL_SIZE/2f0 - 0.1f0)
                new_bv = Vec2f(new_bv[1], -new_bv[2]) # Reflect Y velocity
                collided_y = true
            end

            paddle_left_rect = Rect2f(0f0, p_left_y, PADDLE_WIDTH, PADDLE_HEIGHT)
            paddle_right_rect = Rect2f(COURT_WIDTH - PADDLE_WIDTH, p_right_y, PADDLE_WIDTH, PADDLE_HEIGHT)
            ball_rect_predict = Rect2f(new_bp[1]-BALL_SIZE/2f0, new_bp[2]-BALL_SIZE/2f0, BALL_SIZE, BALL_SIZE)

            if new_bv[1] < 0f0 && rect_overlaps(paddle_left_rect, ball_rect_predict) # Moving left, hit left paddle
                new_bp = Point2f(PADDLE_WIDTH + BALL_SIZE/2f0 + 0.1f0, new_bp[2])

                paddle_center = p_left_y + PADDLE_HEIGHT / 2f0
                hit_offset = clamp((new_bp[2] - paddle_center) / (PADDLE_HEIGHT * 0.55f0), -1.0f0, 1.0f0) # Slightly wider range
                bounce_angle = hit_offset * (pi / 3f0) # Max angle ~60 degrees
                cbs += BALL_SPEED_INCREASE
                new_bv = Vec2f(cos(bounce_angle), sin(bounce_angle)) * cbs # New velocity vector

            elseif new_bv[1] > 0f0 && rect_overlaps(paddle_right_rect, ball_rect_predict) # Moving right, hit right paddle
                new_bp = Point2f(COURT_WIDTH - PADDLE_WIDTH - BALL_SIZE/2f0 - 0.1f0, new_bp[2])

                paddle_center = p_right_y + PADDLE_HEIGHT / 2f0
                hit_offset = clamp((new_bp[2] - paddle_center) / (PADDLE_HEIGHT * 0.55f0), -1.0f0, 1.0f0) # Slightly wider range
                bounce_angle = hit_offset * (pi / 3f0) # Max angle ~60 degrees
                cbs += BALL_SPEED_INCREASE
                world_angle = pi - bounce_angle # Reflect angle for right paddle
                new_bv = Vec2f(cos(world_angle), sin(world_angle)) * cbs # New velocity vector
            end

            ball_pos[] = new_bp
            ball_vel[] = new_bv
            current_ball_speed[] = cbs

            scored = false
            final_pos = ball_pos[]
            if final_pos[1] - BALL_SIZE/2f0 <= 0f0 # Left boundary miss
                score_right[] += 1; served_by[] = :p1; scored = true
            elseif final_pos[1] + BALL_SIZE/2f0 >= COURT_WIDTH # Right boundary miss
                score_left[] += 1; served_by[] = :p2; scored = true
            end

            # Handle scoring outcome
            if scored
                if score_left[] >= SCORE_LIMIT || score_right[] >= SCORE_LIMIT
                    game_active[] = false
                    winner = score_left[] >= SCORE_LIMIT ? "Left Player" : "Right Player"
                    game_message[] = "$winner Wins!\nPress R to Restart"
                else
                    reset_ball() # Reset for next point 
                end
            end

        # Ball follow paddle before serve
        elseif serve_state[] == :p1_serve
            target_y = p_left_y + PADDLE_HEIGHT / 2f0
            new_pos = Point2f(bp[1], clamp(target_y, BALL_SIZE/2f0, COURT_HEIGHT - BALL_SIZE/2f0))
            ball_pos[] = new_pos; prev_ball_pos[] = new_pos
        elseif serve_state[] == :p2_serve
            target_y = p_right_y + PADDLE_HEIGHT / 2f0
            new_pos = Point2f(bp[1], clamp(target_y, BALL_SIZE/2f0, COURT_HEIGHT - BALL_SIZE/2f0))
            ball_pos[] = new_pos; prev_ball_pos[] = new_pos
        end

        return Consume(false)
    end # end on tick

    on(events(fig).keyboardbutton) do event
        is_serve_key_p1 = event.key == Keyboard.space || event.key == Keyboard.a || event.key == Keyboard.d
        is_serve_key_p2 = event.key == Keyboard.enter || event.key == Keyboard.left || event.key == Keyboard.right
        current_state = serve_state[] # Cache current state

        if game_active[] && event.action == Keyboard.press
            serve_triggered = false
            base_angle = 0f0

            if current_state == :p1_serve && is_serve_key_p1
                serve_triggered = true
                base_angle = 0f0 # Serve right
                served_by[] = :p1 # Ensure correct server is tracked
            elseif current_state == :p2_serve && is_serve_key_p2
                serve_triggered = true
                base_angle = pi # Serve left
                served_by[] = :p2 # Ensure correct server is tracked
            end

            if serve_triggered
                angle_spread = pi/3f0
                angle = base_angle + (rand(Float32) * angle_spread - angle_spread/2f0)
                serve_speed = current_ball_speed[]
                ball_vel[] = Vec2f(cos(angle), sin(angle)) * serve_speed
                serve_state[] = :playing
                game_message[] = "" # Clear serve message
                return Consume(true) # Consume the serve event
            end
        end

        # --- Restart Logic ---
        if !game_active[] && event.key == Keyboard.r && event.action == Keyboard.press
            score_left[] = 0; score_right[] = 0
            served_by[] = rand([:p1, :p2]) # Randomly assign first server on restart
            reset_ball() # Resets position, speed, state, and message
            game_active[] = true
            if FLUID_ENABLED # Reset fluid state on restart
                 fill!(fluid_dens, 0.0f0)
                 fill!(fluid_vx, 0.0f0)
                 fill!(fluid_vy, 0.0f0)
                 # Clear observable as well
                 density_view = reshape(view(fluid_dens, [IX(i,j) for i=2:FLUID_NX+1, j=2:FLUID_NY+1]), (FLUID_NX, FLUID_NY))
                 fluid_dens_obs[] = density_view
            end
            return Consume(true)
        end

        return Consume(false)
    end # end on keyboard

    display(fig)
    return fig
end

run()