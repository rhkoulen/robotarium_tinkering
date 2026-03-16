import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc, PathPatch

# Boid kwargs
W_S = 0.1 # separation constant
W_A = 2.0 # alignment constant
W_C = 1.0 # cohension constant
W_F = 0.1 # force multiplier
N_R = 1.0 # neighborhood radius
path_radius = 0.8 # radius of traced path

# Instantiate Robotarium object
iters = 5000
N_boids = 8
initial_conditions = np.array([
    [-0.3, path_radius+0.1, 0],
    [-0.3, path_radius-0.1, 0],
    [-0.1, path_radius+0.1, 0],
    [-0.1, path_radius-0.1, 0],
    [ 0.1, path_radius+0.1, 0],
    [ 0.1, path_radius-0.1, 0],
    [ 0.3, path_radius+0.1, 0],
    [ 0.3, path_radius-0.1, 0],
]).T
initial_velocities = np.zeros(shape=(2,N_boids))
initial_velocities[0,:] = 0.12
rng = np.random.default_rng(56)
max_velocities = rng.permutation(np.array([0.7, 0.10, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13]))

r = robotarium.Robotarium(number_of_robots=N_boids, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Draw middle boundary (100:1 scale)
r.axes.add_patch(Rectangle(xy=(-0.42195, -0.365), width=0.8439, height=0.73, color="#074614", lw=0, zorder=0))
r.axes.add_patch(Circle(xy=(0.42195, 0.0), radius=0.365, color="#074614", lw=0, zorder=0))
r.axes.add_patch(Circle(xy=(-0.42195, 0.0), radius=0.365, color="#074614", lw=0, zorder=0))
r.axes.add_patch(Arc(xy=(0.42195, 0.0), width=path_radius*2, height=path_radius*2, theta1=-90, theta2=90, color="#000000", lw=2))
r.axes.add_patch(Arc(xy=(-0.42195, 0.0), width=path_radius*2, height=path_radius*2, theta1=90, theta2=270, color="#000000", lw=2))
r.axes.plot([-0.42195, 0.42195], [-path_radius, -path_radius], color="#000000", lw=2)
r.axes.plot([-0.42195, 0.42195], [path_radius, path_radius], color="#000000", lw=2)

# Create controllers
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
# _, uni_to_si_states = create_si_to_uni_mapping() # whart does this used for idkkk
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
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
    # 2. Compute alignment vector, a
    average_velocities = (past_velos @ neighborhood[0]) / (neighborhood_count + 1e-8) # (2,N)
    a = average_velocities
    # 3. Compute vector to centroid, c
    average_position = (positions @ neighborhood[0]) / (neighborhood_count + 1e-8) # (2,N)
    c = average_position - positions # from pos to neighborhood centroid
    # 4. Compute deviation from proscribed path
    # TODO: how to get distance from the "track", which is two line segments and two semicircles
    # min distance from each component
    # I can do distance from a line segment, but not distance from a semicircle

    # 5. f = W_S * s + W_A * a + W_C * c + [path deviation force?]
    boid_forces = W_F * (W_S * s + W_A * a + W_C * c) # (2,N)
    proposed_velocities = past_velos + boid_forces
    # 6. Clamp each velocity to max_velocities
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
    dxi = si_barrier_cert(dxi, x[:2, :]) # safety layer
    dxu = si_to_uni_dyn(dxi, x) # convert to unicycle [v; ω]

    # Set and iterate
    past_velocities = dxi
    r.set_velocities(np.arange(N_boids), dxu)
    r.step()


r.call_at_scripts_end()
