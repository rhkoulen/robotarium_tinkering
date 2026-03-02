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
You can find the stuff that I made in ``
