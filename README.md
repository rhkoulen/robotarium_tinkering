Introduction
============

The Robotarium is a project at Georgia Institute of Technology allowing public, remote access to a state-of-the-art multi-robot testbed. This is forked off an open-source, re-implementation of the [MATLAB simulator](https://github.com/robotarium/robotarium-matlab-simulator) in Python. This is not official code, you should instead look at [their repo](https://github.com/robotarium/robotarium_python_simulator).

I'm mostly using this as a testbed for some stuff I want to try in multi-agent planning.

## Installation
The installation instructions on the source repo are outdated (as of 2 March 2026). Ignore their advice and just do a modern setup. I recommend conda, and running the environment.yml that I made. It might have some backward compatibility issues if you use features that aren't on the Robotarium runner, I'm no expert to be sure, but their installation instructions just don't compile.

1. Have some installation of conda. I use miniforge3, stuck on v25.11.0 since the upgrade command doesn't work. I suspect any recent version of anaconda, miniconda, or miniforge would work just fine.
2. Open your conda enabled shell, clone this repo, and cd in.
3. `conda env create -f environment.yml`
4. `conda activate robotarium`

You win! If you don't want my build and want your own, you need matplotlib, cvxopt, scipy>=0.18, and numpy<2. Then `pip install -e .` to get the rps module built.

## My Stuff
You can find the stuff that I made in `.\experiments\`







## Official Instructions
One week after I made this repo to fix stuff, they pushed a fix! Here's the official install instructions:

The following dependencies are required for utilization of the simulator:
- [NumPy] (http://www.numpy.org)
- [matplotlib] (http://matplotlib.org/index.html)
- [CVXOPT] (http://cvxopt.org/index.html)

NOTE: The SciPy stack and matplotlib can be difficult to install on Windows. However, [this] (http://www.lfd.uci.edu/~gohlke/pythonlibs/) link provides .whl files necessary for installation. Make sure to install all the dependencies for each version part of the SciPy and matplotlib stack!

## Dependency Installation

The guide below will show you how to install the necessary dependencies. The simulator has been thoroughly tested on Python 3.10.x+ versions.


### Pip

Pip is the standard dependency manager for python.  To install the simulator, use
```
# Install Dependencies
pip install numpy==2.2.6 matplotlib==3.10.8 cvxopt==1.3.2

# Installing the Robotarium Simulator
# Navigate to the cloned simulator directory containing the setup.py script. Then run:
pip install .
**Note the dot after install**
```

### Submission Dependencies

The current list of libraries supported by the robotarium is contained in [libraries.txt](./libraries.txt).  If you cannot find a library that is required for your submission, please submit a pull request adding your package to the libraries.txt file so the team can evaluate its addition into the robotarium.


## Usage
To run one of the examples:

 ```
 python "path_to_simulator"/rps/examples/plotting/barrier_certificates_with_plotting.py
 ```

## Issues
Please enter a ticket in the [issue tracker](https://github.com/robotarium/robotarium_python_simulator/issues).
