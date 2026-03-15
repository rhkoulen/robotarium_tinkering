import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc, PathPatch

# Boid kwargs
W_S = 1.0
W_A = 1.0
W_D = 1.0
path_radius = 0.7

# Instantiate Robotarium object
iters = 500
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
rng = np.random.default_rng(56)
max_velocities = rng.shuffle(np.array([0.7, 0.10, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13]))

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
def boids_velocities(x:np.ndarray) -> np.ndarray:
    """
    Takes in get_poses, and returns boid velocities.

    In addition to normal boid behavior, this function also has a trajectory force,
    which nudges the robots to be line followers.

    """
    # For each bot (x is (3,8) shaped)
    # 1. Compute separation vector, s
    # 2. Compute alignment vector, a
    # 3. Compute cohesion centroid, c (probably with an ignorance radius, or IPW)
    # 4. Compute deviation from proscribed path

    # 5. f = W_S * s + W_A * a + W_C * c + [path deviation force?]
    # 6. Clamp each velocity to max_velocities
    raise NotImplementedError()

x = r.get_poses()
r.step()
for _ in range(iters):
    x = r.get_poses()

    # Calculate next velos
    dxi = boids_velocities(x) # theoretical boid behavior that I haven't made yet
    dxi = si_barrier_cert(dxi, x) # safety layer
    dxu = si_to_uni_dyn(dxi, x) # convert to unicycle [v; ω]

    # Set and iterate
    r.set_velocities(np.arange(N_boids), dxu)
    r.step()


r.call_at_scripts_end()
