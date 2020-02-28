#####################################
# 2D GPE model of solitons in a BEC #
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

L           = 64                                    # Domain size
N           = 256                                   # Number of grid points
DX          = L / N                                 # Cell size
GRID_AXIS   = np.arange(-L/2, L/2, DX, dtype=complex)
GRID        = np.meshgrid(GRID_AXIS, GRID_AXIS)     # The grid

DT          = 0.01                                  # Time step size

ITERATION_STEPS     = 500                           # Number of iteration steps
STEPS_PER_ITERATION = 10                            # Number of time steps of
                                                    # size DT per iteration
# NOTE The total iteration time is then given by
#      ITERATION_STEPS*STEPS_PER_ITERATION*DT.

k = 2*np.pi*np.fft.fftfreq(N, d=DX)
k = np.meshgrid(k,k)
H_KIN = (k[0]*k[0] + k[1]*k[1]) / 2                 # Kinetic part of the
                                                    # Hamiltonian

FIGURE_PATH = "plots/soliton2d"                     # Where to store the figures
DATA_PATH   = "data/soliton2d"                      # Where to store the data

# .. Model utility functions ..................................................

def h_pot(psi):
    """Returns the potential part of the Hamiltonian.
    
    Args:
        psi (2d array): State
    
    Returns:
        2d array: Potential part of the Hamiltonian
    """
    return np.conjugate(psi)*psi-1

def noise(sigma):
    """Returns a NxN grid containing gaussian distributed values with mean of 1
    and the given std.
    
    Args:
        sigma (float): Standard deviation of the gaussian distribution
    
    Returns:
        2d array: Grid containing random values around 1
    """
    return np.random.normal(1, sigma, (N,N)).astype(complex)

def dark_soliton(z0, nu=0.):
    """Returns a dark soliton in two dimensions, i.e., a (shifted) vertical
    line.
    
    Args:
        z0 (TYPE): Horizontal shift
        nu (float, optional): Greyness
    
    Returns:
        2d array: Dark soliton
    """
    gamma = 1. / np.sqrt(1-nu**2)
    return (1j*nu + np.tanh((GRID[0]-z0)/gamma)/gamma) * np.ones((N,N))

def ring(R, nu=0.):
    """Returns a centered ring-like dark soliton.
    
    Args:
        R (float): Radius
        nu (float, optional): Greyness
    
    Returns:
        2d array: Ring-like dark soliton
    """
    gamma = 1. / np.sqrt(1-nu**2)
    r = np.sqrt(GRID[0]*GRID[0] + GRID[1]*GRID[1])
    return (1j*nu + np.tanh((r-R)/gamma)/gamma) * np.ones((N,N))

# .. Core functions and plotting ..............................................

def plot(psi, file_name):
    """Performs a density and phase plot of the given state.
    
    Args:
        psi (2d array): State
        file_name: File name
    """
    # Density plot
    plt.imshow(
        np.abs(psi)**2,
        interpolation='nearest',
        origin='lower left',
        label = r'',
        vmin=0,
        vmax=1.1,
        extent=[-L/2, L/2-DX, -L/2, L/2-DX]
    )
    cbar =  plt.colorbar()
    cbar.set_label(r'Density', labelpad=5, fontsize=myfontsize)
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('$y$ $[\\xi]$')
    plt.savefig('{}/density/{}.png'.format(FIGURE_PATH, file_name), dpi=300,
                bbox_inches='tight')
    plt.close()

    # Phase plot
    plt.imshow(
        np.angle(psi),
        interpolation='nearest',
        origin='lower left',
        label = r'',
        vmin=-np.pi,
        vmax=np.pi,
        cmap ='hsv',
        extent=[-L/2, L/2-DX, -L/2, L/2-DX]
    )
    cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    cbar.set_label(r'Phase angle', labelpad=5, fontsize=myfontsize)
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('$y$ $[\\xi]$')
    plt.savefig('{}/phase/{}.png'.format(FIGURE_PATH, file_name), dpi=300,
                bbox_inches='tight')
    plt.close()

def propagate_state(psi0, *, num_steps, dt=DT):
    """Propagates a given state in time by applying the split-step fourier
    method to the GPE model.
    
    Args:
        psi0 (2d array): Initial state
        num_steps (int): Number of time steps to do
        dt (float, optional): Time step size
    
    Returns:
        2d array: Propagated state
    """
    # Propagate half a time step using the potential part of the Hamiltonian
    psi = np.exp(-1j*(dt/2)*h_pot(psi0))*psi0

    for i in range(num_steps):
        # Propagate full time step with each part of the Hamiltonian. Base
        # transformation is done via fast fourier transform.
        psi = np.fft.ifft2(np.exp(-1j*dt*H_KIN)*np.fft.fft2(psi))
        psi = np.exp(-1j*dt*h_pot(psi))*psi

    # Propagate half a time step using the kinetic part of the Hamiltonian
    psi = np.fft.ifft2(np.exp(-1j*dt*H_KIN)*np.fft.fft2(psi))
    psi = np.exp(-1j*(dt/2)*h_pot(psi))*psi

    return psi

def main():
    # Create directories to store data and plots
    os.system('mkdir -p {}'.format(DATA_PATH))
    os.system('mkdir -p {}/phase'.format(FIGURE_PATH))
    os.system('mkdir -p {}/density'.format(FIGURE_PATH))

    # -- Initialization ---------------------------------------

    # Initialize two parallel black solitons
    psi = dark_soliton(z0=-10., nu=0.) * dark_soliton(z0=10., nu=-0.)
    
    # Initialize a ring-like grey soliton
    # psi = ring(R=5., nu=-0.5)

    # Add noise
    psi *= noise(sigma=0.02)

    # Load data locally
    # psi = np.load('data/soliton2d/500.npy')

    # ---------------------------------------------------------

    # Plot and save initial state
    plot(psi, 0)
    np.save('{}/{}'.format(DATA_PATH, 0), psi)

    # Propagate in time, plot and save results
    for i in tqdm(range(ITERATION_STEPS)):
        psi = propagate_state(psi, num_steps=STEPS_PER_ITERATION, dt=DT)
        plot(psi, i+1)
        np.save('{}/{}'.format(DATA_PATH, i+1), psi)

if __name__ == "__main__":
    main()
