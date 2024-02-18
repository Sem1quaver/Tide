# Import necessary libraries
import numpy as np  # For numerical computations
from scipy.special import lpmv  # Import Legendre functions
import dedalus.public as d3  # Dedalus is a library for solving partial differential equations
import logging  # For logging
logger = logging.getLogger(__name__)  # Create a logger

# Simulation units
meter = 1 / 6.37122e6  # Meter, in units of Earth's radius
hour = 1  # Hour
second = hour / 3600  # Second

# Set parameters
Ri = 6.37122e6 * meter  # Inner radius of the shell
Ro = 6.38122e6 * meter  # Outer radius of the shell
Omega = 7.292e-5 / second  # Earth's rotation angular speed
g = 9.80616 * meter / second**2  # Gravitational acceleration
nu = 1e5 * meter**2 / second / 32**2  # Viscosity coefficient, for describing the viscosity of the fluid
Nphi, Ntheta, Nr = 192, 96, 6  # Number of discrete points in angles and radius in spherical coordinates
Rayleigh = 3500  # Rayleigh number, describes the intensity of thermal convection in the fluid
Prandtl = 1  # Prandtl number, describes the ratio of momentum diffusion to heat diffusion
dealias = 3/2  # Anti-aliasing factor, used to handle nonlinear terms in spectral methods
timestepper = d3.SBDF2
max_timestep = 0.05
stop_sim_time = 360 * hour  # Simulation stop time
max_timestep = 1  # Maximum timestep
dtype = np.float64  # Data type
mesh = None  # Mesh, None means using the default global mesh

# Establish basis
coords = d3.SphericalCoordinates('phi', 'theta', 'r')  # Spherical coordinates
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)  # Distributor, used to distribute data among multiple processes
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)  # Shell basis
sphere = shell.outer_surface  # Outer surface of the shell

# Define fields
p = dist.Field(name='p', bases=shell)  # Pressure field
b = dist.Field(name='b', bases=shell)  # Buoyancy field
h = dist.Field(name='h', bases=shell)  # Depth field
u = dist.VectorField(coords, name='u', bases=shell)  # Velocity field
V = dist.VectorField(coords, name='V', bases=shell)  # Tidal potential vector field
tau_p = dist.Field(name='tau_p')  # Tau term of pressure, used to handle boundary conditions
tau_b1 = dist.Field(name='tau_b1', bases=sphere)  # The first tau term of buoyancy
tau_b2 = dist.Field(name='tau_b2', bases=sphere)  # The second tau term of buoyancy
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)  # The first tau term of velocity

# Define substitution terms
kappa = (Rayleigh * Prandtl)**(-1/2)  # Thermal diffusion coefficient
nu = (Rayleigh / Prandtl)**(-1/2)  # Momentum diffusion coefficient
phi, theta, r = dist.local_grids(shell)  # Get local grids
er = dist.VectorField(coords, bases=shell.radial_basis)  # Unit radial vector
er['g'][2] = 1
rvec = dist.VectorField(coords, bases=shell.radial_basis)  # Radius vector
rvec['g'][2] = r
Omega_m = dist.VectorField(coords, name='Omega_m', bases=shell)  # Rotation vector
Omega_m['g'] = np.ones_like(V['g']) * Omega
lift_basis = shell.derivative_basis(1)  # Lift basis
lift = lambda A: d3.Lift(A, lift_basis, -1)  # Lift operation
grad_u = d3.grad(u) + rvec*lift(tau_u1)  # Gradient of velocity, including the first order tau term
grad_b = d3.grad(b) + rvec*lift(tau_b1)  # Gradient of buoyancy, including the first order tau term
cross_V = d3.cross(V, Omega_m)  # Cross product of tidal potential
cross_u = d3.cross(u, Omega_m)  # Cross product of velocity

# Define the problem
problem = d3.IVP([p, b, h, u, V, tau_p, tau_b1, tau_b2, tau_u1], namespace=locals())  # Initial value problem
problem.add_equation("trace(grad_u) + tau_p = 0")  # Continuity equation
problem.add_equation("dt(h) + nu * lap(lap(h)) + (Ri - Ro) * trace(grad_u) = - div(h * u)")  # Continuity equation
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")  # Evolution equation for buoyancy
problem.add_equation("dt(u) - nu*div(grad_u) + g * grad(h) + g * grad(V@er) + 2 * cross_u + grad(p) - b*er  = - u@grad(u)")  # Momentum equation
problem.add_equation("b(r=Ri) = 1")  # Buoyancy condition at the inner boundary
problem.add_equation("u(r=Ri) = 0")  # Velocity condition at the inner boundary
problem.add_equation("b(r=Ro) = 0")  # Buoyancy condition at the outer boundary
problem.add_equation("integ(p) = 0")  # The integral of pressure is zero, used to set the pressure gauge
problem.add_equation("dt(V) - cross_V = 0")  # Evolution equation for tidal potential

# Build the solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial condition - buoyancy field
#b.fill_random('g', seed=42, distribution='normal', scale=1e-3)  # Fill the buoyancy field with random noise
#b['g'] *= (r - Ri) * (Ro - r)  # Suppress noise at the walls
#b['g'] += (Ri - Ri*Ro/r) / (Ri - Ro)  # Add linear background

# Initial condition - potential field
V['g'][2] = 0  # Initialize the field value to zero
theta_m = np.pi / 2  # Set the value of theta_m
phi_m = 0  # Set the value of phi_m
for n in range(2, 40):
    for m in range(0, n+1):
        A = 1 if m == 0 else 2  # Choose the value of A based on the value of m
        N_nm = ((-1)**m * ((2*n+1)/(4*np.pi) * np.math.factorial(n-m) / np.math.factorial(n+m))**0.5)  # Calculate the value of N_n^m
        P_nm = lpmv(m, n, np.cos(theta))  # Calculate P_n^m using theta
        P_nm_m = lpmv(m, n, np.cos(theta_m))  # Calculate P_n^m using theta_m
        K_n = Ri * 1/81 * (6.37122e6/384403000)**(n+1)  # Calculate the value of K_n
        a_nm = (-1)**m * A * 4*np.pi/(2*n+1) * K_n * N_nm * P_nm_m * np.sin(m*phi_m)  # Calculate b_nm using phi_m
        b_nm = (-1)**m * A * 4*np.pi/(2*n+1) * K_n * N_nm * P_nm_m * np.sin(m*phi_m)  # Calculate b_nm using phi_m
        V['g'] += N_nm * P_nm * (a_nm * np.cos(m*phi) + b_nm * np.sin(m*phi))  # Update the field value

# Analysis
flux = er @ (-kappa*d3.grad(b) + u*b)  # Calculate heat flux
snapshots = solver.evaluator.add_file_handler('snapshots_shell', sim_dt=1 * hour, max_writes=10)  # Add file handler for saving data snapshots
snapshots.add_task(b(r=(Ri+Ro)/2), scales=dealias, name='bmid')  # Calculate and save buoyancy at the middle position
snapshots.add_task(flux(r=Ro), scales=dealias, name='flux_r_outer')  # Calculate and save heat flux at the outer boundary
snapshots.add_task(flux(r=Ri), scales=dealias, name='flux_r_inner')  # Calculate and save heat flux at the inner boundary
snapshots.add_task(flux(phi=0), scales=dealias, name='flux_phi_start')  # Calculate and save heat flux at the starting angle
snapshots.add_task(flux(phi=3*np.pi/2), scales=dealias, name='flux_phi_end')  # Calculate and save heat flux at the ending angle
snapshots.add_task(h, name='height')  # Add height data

# CFL condition
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)  # CFL condition for dynamically adjusting the timestep
CFL.add_velocity(u)  # Add velocity field

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)  # Global flow properties
flow.add_property(np.sqrt(u@u)/nu, name='Re')  # Add Reynolds number

# Main loop
try:
    logger.info('Starting main loop')  # Start main loop
    while solver.proceed:  # While the simulation is not over
        timestep = CFL.compute_timestep()
        solver.step(timestep)  # Perform one step of the simulation
        if (solver.iteration-1) % 10 == 0:  # Log every 10 steps
            max_Re = flow.max('Re')  # Calculate maximum Reynolds number
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))  # Log
except:
    logger.error('Exception raised, triggering end of main loop.')  # If an exception occurs, log the error and end the main loop
    raise
finally:
    solver.log_stats()  # Log the statistics of the simulation
