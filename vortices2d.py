#####################################
# 2D GPE model of vortices in a BEC #
#####################################

# .. Layout Setup .............................................................

import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 0.5
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"]
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

myfontsize = 15

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
   'size':myfontsize})
rc('text', usetex=True)
rc('legend', fontsize=myfontsize)

# .. Model Parameters .........................................................

NX  = 64        # number of grid points in x-direction
NY  = 64        # number of grid points in y-direction

G   = 1e-2      # non-linear GPE coupling / interaction strength
XI  = 4.        # healing length, resolved with 4 grid points, if you use a
                # small grid you can go down to 2 grid points

DT = 0.002      # time step size
DX = 1. / XI    # grid cell size

ITERATION_STEPS     = 500                           # Number of iteration steps
STEPS_PER_ITERATION = 10                            # Number of time steps of
                                                    # size DT per iteration
# NOTE The total iteration time is then given by
#      ITERATION_STEPS*STEPS_PER_ITERATION*DT.

# Initialization mode selection: The configurations of the different modes are
# defined in the `main` function.

TEST_VORTEXPAIR_GRID = False        # If True, initial grid will be vortices on
                                    # fixed positions for testing.
RANDOM_VORTEX_PAIR_GRID = False     # If True, initial grid will be vortex grid
                                    # with an equal number of vortices and
                                    # antivortices placed at random positions.
RANDOM_VORTEX_GRID = False          # If True, initial grid will be vortex grid
                                    # composed of vortices with equal
                                    # quantization placed at random positions.
REGULAR_VORTEX_GRID = False         # If True, initial grid will be regular
                                    # vortex grid, i.e. equal distances between
                                    # vortices and antivortices.
REGULAR_VORTEX_GRID_OFFSET = True   # If True, initial grid will be regular
                                    # vortex grid with gaussian distributed
                                    # offsets.

k = 2*np.pi*np.fft.fftfreq(NX, d=DX)
k = np.meshgrid(k, k)
H_KIN = (k[0]*k[0] + k[1]*k[1]) / 2.    # Kinetic part of the Hamiltonian

FIGURE_PATH = "plots/vortices2d"

# .. Model utility functions ..................................................

### create a grid with homogeneous condensate background density
def create_condensed_grid(nx_grid, ny_grid, N):
    # Initialize grid in Fourier space
    grid = np.zeros((nx_grid,ny_grid), dtype='complex')

    # Fill condensate mode k=0 with all particles
    grid[0,0] = np.sqrt(N/2.) + 1j*np.sqrt(N/2.)

    # Fourier transfrom to obtain Real space grid for further manipulations
    grid = np.fft.fft2(grid) / np.sqrt(nx_grid*ny_grid)

    return grid


### calculate the total number of particles on the grid
def calculate_particle_number(grid):
    return np.sum(np.sum(np.abs(grid)**2, axis=0), axis=0)


### Vortex with winding number n = +- 1
def add_2d_vortex_simple(grid, x_pos, y_pos, n, N, g):
    grid_h = grid
    
    f0 = np.sqrt(N / (grid.shape[0]*grid.shape[1]))
    xi = 1. / (2.*f0*np.sqrt(g))
    
    for x2 in np.arange(0, grid.shape[0]):
        for y2 in np.arange(0, grid.shape[1]):
            # shape of the vortex, rho = 0 at the core,
            # i.e. if x_pos = x2 and y_pos = y2
            rho = x_pos - x2 + 1j*n*(y_pos-y2)
            # multiply homogeneous background field with shape of vortex    
            grid_h[x2, y2] *= (1.0/xi/np.sqrt(2. + np.abs(rho)**2/xi/xi)) * rho

    return grid_h

### Vortex with winding number n  
def add_2d_vortex_simple_n(grid, x_pos, y_pos, n, N, g):
    grid_h = grid
    
    f0 = np.sqrt(N / (grid.shape[0]*grid.shape[1]))
    xi = 1. / (2*f0*np.sqrt(g))

    sn = int(n/np.sqrt(n*n))
    wn = int(np.sqrt(n*n))
    
    for x2 in np.arange(0, grid.shape[0]):
        for y2 in np.arange(0, grid.shape[1]):
            rho = x_pos - x2 + 1j*sn*(y_pos-y2)
            rho *= (1. / xi / np.sqrt(2. + np.abs(rho)**2/xi/xi))
            rho = rho**wn
            grid_h[x2,y2] *= rho
        
    return grid_h

#######################################
## add a single vortex to the grid 
## @params:
## grid : numerical real space grid
## x_pos : x-position of vortex
## y_pos : y-position of vortex
## n : quantization of vortex
## N : total particle number
## g : non-linear coupling
########################################
def add_2d_vortex(grid, x_pos, y_pos, n, N, g):
    grid_h = grid
    if n*n == 1:
        grid_h = add_2d_vortex_simple(grid_h, x_pos, y_pos, n, N, g)
        ## Adding "mirror vortices" on all 8 adjacent cells to respect periodic boundaries:
        grid_h = add_2d_vortex_simple(grid_h, x_pos+grid.shape[0], y_pos, n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos+grid.shape[0], y_pos-grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos, y_pos-grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos-grid.shape[0], y_pos-grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos-grid.shape[0], y_pos, n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos-grid.shape[0], y_pos+grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos, y_pos+grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple(grid_h, x_pos+grid.shape[0], y_pos+grid.shape[1], n, N, g)

    else:
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos, y_pos, n, N,g)
        ## Adding "mirror vortices" on all 8 adjacent cells to respect periodic boundaries:
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos+grid.shape[0], y_pos, n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos+grid.shape[0], y_pos-grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos, y_pos-grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos-grid.shape[0], y_pos-grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos-grid.shape[0], y_pos, n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos-grid.shape[0], y_pos+grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos, y_pos+grid.shape[1], n, N, g)
        grid_h = add_2d_vortex_simple_n(grid_h, x_pos+grid.shape[0], y_pos+grid.shape[1], n, N, g)
    return grid_h

#######################################################################################################################################
## Function creates a grid with randomly placed singly quantized vortices, equal number of vortices with n=1 and antivortices with n=-1
## @params:
## nx_grid : number of grid points in x-direction
## ny_grid : number of grid points in y-direction
## num_pairs : number of vortex pairs with quantization n= +-1 on the grid
## N : total particle number
## g : non-linear coupling
########################################################################################################################################
def create_2d_random_vortexpair_grid(nx_grid, ny_grid, num_pairs, N, g):
    grid = create_condensed_grid(nx_grid, ny_grid, N)

    for i in np.arange(0, num_pairs):
        grid = add_2d_vortex(grid, nx_grid*np.random.random(), ny_grid*np.random.random(), 1, N, g)
        grid = add_2d_vortex(grid, nx_grid*np.random.random(), ny_grid*np.random.random(), -1, N, g)

    return grid

### function creates a grid with randomly placed n-quantized vortices,
### equal quantization of all vortices 
def create_2d_random_vortex_grid(nx_grid, ny_grid, num_vortices, n, N, g):
    grid = create_condensed_grid(nx_grid, ny_grid,N)

    for i in np.arange(0, num_vortices):
        grid = add_2d_vortex(grid, nx_grid*np.random.random(), ny_grid*np.random.random(), n, N, g)

    return grid

##########################################################################################################################################################################################
## Function creates a grid with a regular vortex configuration, vortices can also be highly quantized (n>1 or n<-1), arrangement in a checkerboard manner, i.e. vortices would correspond
## for example to black fields and antivortices to white fields of a checkerboard
## @params:
## nx_grid : number of grid points in x-direction
## ny_grid : number of grid points in y-direction
## lx : number of defects in x-direction of checkerboard
## ly : number of defects in y-direction of checkerboard
## n : quantization of the vortices
## N : total particle number
## g : non-linear coupling
#######################################################################################################################################################################################
def create_2d_regular_vortex_grid(nx_grid, ny_grid, lx, ly, n, N, g):
    grid = create_condensed_grid(nx_grid, ny_grid, N)

    dist_x = nx_grid / lx
    dist_y = ny_grid / ly
    
    sy = 1
    for x in np.arange(1, lx+1):
        sx = 1
        for y in np.arange(1, ly+1):
            grid = add_2d_vortex(grid, dist_x*x, dist_y*y, n=sy*sx*n, N=N, g=g)
            sx *= -1
        sy*= -1

    return grid

### function that creates regular vortex grid with gaussian distributed offsets
def create_2d_regular_vortex_grid_offset(nx_grid, ny_grid, lx, ly, n, N, g):
    grid = create_condensed_grid(nx_grid, ny_grid, N)

    dist_x = nx_grid / lx
    dist_y = ny_grid / ly

    offsetx  = np.random.normal(0., 0.1, (lx, ly))
    offsety  = np.random.normal(0., 0.1, (lx, ly))
    
    sy = 1
    for x in np.arange(1, lx+1):
        sx = 1
        for y in np.arange(1, ly+1):
            grid = add_2d_vortex(grid,
                                 dist_x*(x+offsetx[x-1,y-1]),
                                 dist_y*(y+offsety[x-1,y-1]),
                                 n=sy*sx*n, N=N, g=g)
            sx *= -1
        sy*= -1

    return grid

### function that creates vortices perdefined positions for testing
def create_2d_test_vortexpair_grid(nx_grid, ny_grid, num_pairs, N, g):
    grid = create_condensed_grid(nx_grid, ny_grid,N)

    for i in np.arange(0, num_pairs):
        grid = add_2d_vortex(grid, 32., 0., 1, N=N, g=g)
        grid = add_2d_vortex(grid, 32., 32., -1, N=N, g=g)

    return grid

def h_pot(psi):
    """Returns the potential part of the Hamiltonian.
    
    Args:
        psi (2d array): State
    
    Returns:
        2d array: Potential part of the Hamiltonian
    """
    return np.conjugate(psi)*psi

# .. Core functions and plotting ..............................................

def propagate_state(psi0, *, num_steps, dt=DT, g=1.):
    """Propagates a given state in time by applying the split-step fourier
    method to the GPE model.
    
    Args:
        psi0 (2d array): Initial state
        num_steps (int): Number of time steps to do
        dt (float, optional): Time step size
        g (float, optional): Interaction strength
    
    Returns:
        2d array: Propagated state
    """
    # Propagate half a time step using the potential part of the Hamiltonian
    psi = np.exp(-1j*(dt/2)*g*h_pot(psi0))*psi0

    for i in range(num_steps):
        # Propagate full time step with each part of the Hamiltonian. Base
        # transformation is done via fast fourier transform.
        psi = np.fft.ifft2(np.exp(-1j*dt*H_KIN)*np.fft.fft2(psi))
        psi = np.exp(-1j*dt*g*h_pot(psi))*psi

    # Propagate half a time step using the kinetic part of the Hamiltonian
    psi = np.fft.ifft2(np.exp(-1j*dt*H_KIN)*np.fft.fft2(psi))
    psi = np.exp(-1j*(dt/2)*g*h_pot(psi))*psi

    return psi

def plot(grid, *, nx_grid, ny_grid, xi, file_name):
    # Density plot
    plt.imshow(
        np.abs(grid)**2,
        interpolation='nearest',
        origin='lower left',
        label = r'',
        vmin=0,
        vmax=10,
        extent=[0, nx_grid/xi, 0, ny_grid/xi]
    )
    cbar = plt.colorbar()
    cbar.set_label(r'Density', labelpad=5, fontsize=myfontsize)
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('$y$ $[\\xi]$')
    plt.savefig('{}/density/{}.png'.format(FIGURE_PATH, file_name), dpi=300,
                bbox_inches='tight')
    plt.close()

    # Phase plot
    plt.imshow(
        np.angle(grid),
        interpolation='nearest',
        origin='lower left',
        label = r'',
        vmin=-np.pi,
        vmax=np.pi,
        cmap ='hsv',
        extent=[0, nx_grid/xi, 0, ny_grid/xi]
    )
    cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    cbar.set_label(r'Phase angle', labelpad=5, fontsize=myfontsize)
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('$y$ $[\\xi]$')
    plt.savefig('{}/phase/{}.png'.format(FIGURE_PATH, file_name), dpi=300,
                bbox_inches='tight')
    plt.close()

def main():
    # Create directories to store figures in
    os.system('mkdir -p {}/phase'.format(FIGURE_PATH))
    os.system('mkdir -p {}/density'.format(FIGURE_PATH))

    # Calculate homogeneous background density
    rho = 1. / (2*G*XI**2)
    print("Homogeneous background density:" , rho)
    
    # Calculate total particle number of simulation
    N = rho * NX * NY
    print("Particle number: " , N)

    # -- Initialization ---------------------------------------

    if TEST_VORTEXPAIR_GRID:
        num_vortex_pairs = 1 # number of vortex pairs with quantization n = +-1 
        grid = create_2d_test_vortexpair_grid(NX, NY, num_vortex_pairs, N, G)

    elif RANDOM_VORTEX_PAIR_GRID:
        num_vortex_pairs = 5 # number of vortex pairs with quantization n = +-1
        grid = create_2d_random_vortexpair_grid(NX, NY, num_vortex_pairs, N, G)

    elif RANDOM_VORTEX_GRID:
        num_vortices = 10 # total number of vortices
        n = 2             # quantization of vortices
        grid = create_2d_random_vortex_grid(NX, NY, num_vortices, n, N, G)

    elif REGULAR_VORTEX_GRID:
        lx = 4 # number of vortices in x-direction on checkerboard
        ly = 4 # number of vortices in y-direction on checkerboard
        n = 4  # quantization of the vortices 
        grid = create_2d_regular_vortex_grid(NX, NY, lx, ly, n, N, G)
    
    elif REGULAR_VORTEX_GRID_OFFSET:
        lx = 4 # number of vortices in x-direction on checkerboard
        ly = 4 # number of vortices in y-direction on checkerboard
        n = 4  # quantization of the vortices 
        grid = create_2d_regular_vortex_grid_offset(NX, NY, lx, ly, n, N, G)

    else:
        # One of the initialization modes must be chosen. Don't continue.
        print("Error, please choose one of the available initialization modes!")
        return

    # Normalize to initially set particle number
    grid *= np.sqrt(N / calculate_particle_number(grid))
    print("Particle number on created vortex grid: ",
          calculate_particle_number(grid))

    # ---------------------------------------------------------

    # Plot initial state
    plot(grid, nx_grid=NX, ny_grid=NY, xi=XI, file_name=0)

    # Propagate in time and plot results
    for i in tqdm(range(ITERATION_STEPS)):
        grid = propagate_state(grid, num_steps=STEPS_PER_ITERATION, dt=DT, g=G)
        plot(grid, nx_grid=NX, ny_grid=NY, xi=XI, file_name=i+1)

if __name__=='__main__':
    main()
