import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc, PathPatch

# Boid kwargs
W_S = 0.05 # separation constant
W_A = 0.2 # alignment constant
W_C = 0.2 # cohesion constant
W_D = 2.0 # path deviation constant
path_tangent_nudge = 0.1 # amount along the tangent that the boid is nudged (m)

W_F = 0.1 # force multiplier
N_R = 0.5 # neighborhood radius
N_FOV = np.pi / 2

# Path kwargs
path_radius = 0.8 # radius of traced path (m)
inner_box_x_rad = 0.42195 # half width of the inner rectangle
inner_box_y_rad = 0.365 # half height of the inner rectangle

# Instantiate Robotarium object
iters = 10000
N_boids = 16
initial_conditions = np.array([
    [-0.60, path_radius, 0],
    [-0.60, path_radius-0.40, 0],
    [-0.20, path_radius, 0],
    [-0.20, path_radius-0.40, 0],
    [ 0.20, path_radius, 0],
    [ 0.20, path_radius-0.40, 0],
    [ 0.60, path_radius, 0],
    [ 0.60, path_radius-0.40, 0],

    [-0.60, -path_radius+0.40, np.pi],
    [-0.60, -path_radius, np.pi],
    [-0.20, -path_radius+0.40, np.pi],
    [-0.20, -path_radius, np.pi],
    [ 0.20, -path_radius+0.40, np.pi],
    [ 0.20, -path_radius, np.pi],
    [ 0.60, -path_radius+0.40, np.pi],
    [ 0.60, -path_radius, np.pi],
]).T
initial_velocities = np.zeros(shape=(2,N_boids))
initial_velocities[0,:8] = 0.12
initial_velocities[0,8:] = -0.12
rng = np.random.default_rng(56)
max_velocities = rng.uniform(low=0.05, high=0.13, size=(N_boids,))

r = robotarium.Robotarium(number_of_robots=N_boids, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Draw middle boundary
r.axes.add_patch(Rectangle(xy=(-inner_box_x_rad, -inner_box_y_rad), width=inner_box_x_rad*2, height=inner_box_y_rad*2, color="#074614", lw=0, zorder=0))
r.axes.add_patch(Circle(xy=(inner_box_x_rad, 0.0), radius=inner_box_y_rad, color="#074614", lw=0, zorder=0))
r.axes.add_patch(Circle(xy=(-inner_box_x_rad, 0.0), radius=inner_box_y_rad, color="#074614", lw=0, zorder=0))
r.axes.add_patch(Arc(xy=(inner_box_x_rad, 0.0), width=path_radius*2, height=path_radius*2, theta1=-90, theta2=90, color="#000000", lw=2, zorder=0))
r.axes.add_patch(Arc(xy=(-inner_box_x_rad, 0.0), width=path_radius*2, height=path_radius*2, theta1=90, theta2=270, color="#000000", lw=2, zorder=0))
r.axes.plot([-inner_box_x_rad, inner_box_x_rad], [-path_radius, -path_radius], color="#000000", lw=2, zorder=0)
r.axes.plot([-inner_box_x_rad, inner_box_x_rad], [path_radius, path_radius], color="#000000", lw=2, zorder=0)

# Create controllers
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
# _, uni_to_si_states = create_si_to_uni_mapping() # whart does this used for idkkk
si_to_uni_dyn = create_si_to_uni_dynamics()
def threshold_wheel_speeds(dxu, wheel_radius, base_length, max_wheel_velocity):
    # Compute wheel speeds
    dxdd = np.vstack((
        (2*dxu[0,:] - base_length*dxu[1,:]) / (2*wheel_radius),
        (2*dxu[0,:] + base_length*dxu[1,:]) / (2*wheel_radius)
    ))
    
    # How much does each robot exceed the limit?
    max_per_robot = np.max(np.abs(dxdd), axis=0)
    scale = np.minimum(1.0, max_wheel_velocity / (max_per_robot + 1e-8))
    
    return dxu * scale[None, :]

def compute_racing_force(positions:np.ndarray) -> np.ndarray:
    """
    We want a force that nudges boids to race.

    1. Add a force that pushes boids to the racing line.
    I'll point that force to the closest track point.
    There are 4 segments, 2 lines 2 semicircles, can be clever based on x position.

    2. Add a force along the clockwise direction, some short amount along the tangent.
    """
    x = positions[0, :] # (N,)
    y = positions[1, :] # (N,)

    racing_force = np.zeros_like(positions)

    # Masks for each region
    left  = x < -inner_box_x_rad
    right = x >  inner_box_x_rad
    mid   = ~left & ~right

    # --- Left semicircle ---
    if left.any():
        cx, cy = -inner_box_x_rad, 0.0
        disp = positions[:, left] - np.array([[cx], [cy]])
        unit = disp / (np.linalg.norm(disp, axis=0, keepdims=True) + 1e-8)
        candidate = np.array([[cx], [cy]]) + unit * path_radius
        centering = candidate - positions[:, left]
        tangent = np.stack([unit[1], -unit[0]]) * path_tangent_nudge
        racing_force[:, left] = centering + tangent

    # --- Right semicircle ---
    if right.any():
        cx, cy = inner_box_x_rad, 0.0
        disp = positions[:, right] - np.array([[cx], [cy]])
        unit = disp / (np.linalg.norm(disp, axis=0, keepdims=True) + 1e-8)
        candidate = np.array([[cx], [cy]]) + unit * path_radius
        centering = candidate - positions[:, right]
        tangent = np.stack([unit[1], -unit[0]]) * path_tangent_nudge
        racing_force[:, right] = centering + tangent

    # --- Middle: top and bottom ---
    top = mid & (y >= 0)
    bot = mid & (y <  0)

    if top.any():
        centering = np.array([[0.0], [path_radius]]) - positions[:, top]
        centering[0, :] = 0  # x component already zero by construction
        racing_force[:, top] = centering + np.array([[path_tangent_nudge], [0]])

    if bot.any():
        centering = np.array([[0.0], [-path_radius]]) - positions[:, bot]
        centering[0, :] = 0
        racing_force[:, bot] = centering + np.array([[-path_tangent_nudge], [0]])

    return racing_force
def boids_velocities(x:np.ndarray, past_velos:np.ndarray) -> np.ndarray:
    """
    Takes in get_poses and previous velocities, returns new boid velocities.

    In addition to normal boid behavior, this function also has a trajectory force,
    which nudges the robots to be line followers.
    """
    # x is (3,N) shaped, (x,y,theta)
    # 1. Compute the neighborhood of each point.
    positions = x[:2, :] # (2,N)
    pairwise_displacement = positions[:, None, :] - positions[:, :, None] # (2,N,N)
    pairwise_displacement[:, np.arange(N_boids), np.arange(N_boids)] = 1e10
    pairwise_sq_distance = np.sum(pairwise_displacement**2, axis=0, keepdims=True) # (1,N,N)
    neighborhood = (pairwise_sq_distance < N_R**2).astype(float) # (1,N,N), neighborhood[i,j] = True if j neighbors i
    neighborhood_count = neighborhood.sum(axis=1) # (1,N), number of neighbors per bot
    # 2. Compute separation vector, s
    IDW_displacement = pairwise_displacement / (pairwise_sq_distance + 1e-8) # (2,N,N)
    IDW_displacement_masked = IDW_displacement * neighborhood # only sum over neighbors
    IDW_average_displacement = np.sum(IDW_displacement_masked, axis=1) # (2,N)
    s = IDW_average_displacement
    # 3. Compute alignment vector, a
    average_velocities = (past_velos @ neighborhood[0]) / (neighborhood_count + 1e-8) # (2,N)
    a = average_velocities
    # 4. Compute vector to centroid, c
    average_position = (positions @ neighborhood[0]) / (neighborhood_count + 1e-8) # (2,N)
    c = average_position - positions # from pos to neighborhood centroid
    # 5. Compute deviation from proscribed path
    d = compute_racing_force(positions)

    # 6. f = W_S * s + W_A * a + W_C * c + [path deviation force?]
    boid_forces = W_F * (W_S * s + W_A * a + W_C * c + W_D * d) # (2,N)
    proposed_velocities = past_velos + boid_forces
    # 7. Clamp each velocity to max_velocities
    proposed_velocity_magnitudes = np.linalg.norm(proposed_velocities, axis=0) # (N,)
    scalars = np.minimum(1.0, max_velocities / (proposed_velocity_magnitudes + 1e-8)) # (N,), if velos are too high, the ratio will bring it back to clamp
    clamped_velocities = proposed_velocities * scalars[None, :] # (2,N)

    # Return velocities, (2,N) shaped
    return clamped_velocities

x = r.get_poses()
r.step() # do a blank step (idk the examples have it)
past_velocities = initial_velocities
for _ in range(iters):
    x = r.get_poses()

    # Calculate next velos
    dxi = boids_velocities(x, past_velocities) # theoretical boid behavior that I haven't made yet
    dxi_safe = si_barrier_cert(dxi, x[:2, :]) # safety layer
    dxu = si_to_uni_dyn(dxi_safe, x) # convert to unicycle [v; ω]

    # Can't trust the unicycle dynamics to not overrun the actuators
    dxu_safe = threshold_wheel_speeds(dxu, r.wheel_radius, r.base_length, r.max_wheel_velocity)

    # Set and iterate
    past_velocities = dxi_safe
    r.set_velocities(np.arange(N_boids), dxu_safe)
    r.step()


r.call_at_scripts_end()
