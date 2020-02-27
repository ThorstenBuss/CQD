#####################################
# 1D GPE model of solitons in a BEC #
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

myfontsize = 15

from matplotlib import rc
rc('font', **{'family':'serif','serif':['Computer Modern Roman'],
   'size':myfontsize})
rc('text', usetex=True)
rc('legend', fontsize=myfontsize)

# .. Model Parameters .........................................................

L       = 40                                        # Domain size
N       = 600                                       # Number of grid points
DX      = L / N                                     # Cell size
GRID    = np.arange(-L/2, L/2, DX, dtype=complex)   # The grid

DT      = 0.01                                      # Time step size

H_KIN   = 0.5*(2*np.pi*np.fft.fftfreq(N, d=DX))**2  # Kinetic part of the
                                                    # Hamiltonian

FIGURE_PATH = "plots/solitons1d"                    # Where to store the figures

# .. Model utility functions ..................................................

def abs_square(arr):
    """Returns the absolute square of a complex array.
    
    Args:
        arr (np.array): Input array
    
    Returns:
        np.array: Absolute square of input
    """
    return np.conjugate(arr)*arr

def h_pot(psi):
    """Returns the potential part of the Hamiltonian (in diagonal form) which
    is given by the absolute square of the wavefunction.
    
    Args:
        psi (np.array): State
    
    Returns:
        np.array: Potential part of the Hamiltonian
    """
    return abs_square(psi)

def dark_soliton(z0, nu=0.5):
    """Returns a dark (grey) soliton.
    
    Args:
        z0 (float): Initial position
        nu (float, optional): Greyness
    
    Returns:
        Dark soliton
    """
    gamma = 1./np.sqrt(1.-nu**2)
    return 1j*nu + np.tanh((GRID-z0)/gamma)/gamma

def black_soliton(z0):
    """Returns a black soliton, i.e., a stationary dark soliton.
    
    Args:
        z0 (float): Initial position
    
    Returns:
        Black soliton
    """
    return dark_soliton(z0, nu=0.)

# .. Core functions and plotting ..............................................

def time_evolution(psi0, num_steps, dt=DT):
    """Calculates the GPE time evolution of the probability density given an
    initial state using the split-step fourier method.
    
    Args:
        psi0 (np.array): Initial state
        num_steps (int): Number of iteration steps
        dt (float, optional): Time step size
    
    Returns:
        Probability density for all iteration times; states are sorted in
        columns in temporal order.
    """
    prob_densities = np.zeros((num_steps+1, N))
    psi = psi0
    prob_densities[0] = np.real(abs_square(psi))

    for i in range(num_steps):
        psi = np.fft.fft(psi)
        psi = np.exp(-1j*dt*H_KIN)*psi
        psi = np.fft.ifft(psi)
        psi = np.exp(-1j*dt*h_pot(psi))*psi
        prob_densities[i+1] = np.real(abs_square(psi))

    return prob_densities

def run_and_plot(psi0, *, num_steps, dt=DT, file_name):
    """Runs the model given an initial state and plots the temporal
    development of the probability density.
    
    Args:
        psi0 (np.array): Initial state
        num_steps (int): Number of iteration steps
        dt (float, optional): Time step size
        file_name (str): Figure title
    """
    prob_densities = time_evolution(psi0, num_steps=num_steps, dt=dt)

    plt.imshow(prob_densities, cmap=plt.get_cmap("BuPu"), origin='lower', 
               extent=[-L/2, L/2-DX, dt/2, (num_steps+1/2)*dt], aspect='auto')
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('t')
    cbar = plt.colorbar()
    cbar.set_label(r'Density', labelpad=5, fontsize=myfontsize)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

def run_and_plot_sym(nu, dt=DT):
    """Creates a symmetric soliton configuration as initial state and runs the
    model. Two solitons are placed at z0=+-10 with opposite velocities.
    
    Args:
        nu (float): Greyness. One soliton gets nu, the other gets -nu
        dt (float, optional): Time step size
    """
    psi0  = dark_soliton(-10., nu=nu) * dark_soliton(10., nu=-nu)
    run_and_plot(psi0, num_steps=int(20/(nu*dt)),
                 file_name=FIGURE_PATH+'/nu{}.png'.format(nu))

def main():
    # Store the created plots here
    os.system('mkdir -p {}'.format(FIGURE_PATH))

    psi0  = black_soliton(0)
    psi1  = dark_soliton(0,0.5)

    plt.plot(np.real(GRID), np.real(np.angle(psi0)),label='$\\nu=0$')
    plt.plot(np.real(GRID), np.real(np.angle(psi1)),label='$\\nu=0.5$')
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('Phase')
    plt.legend()
    plt.savefig(FIGURE_PATH+'/phase.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.plot(np.real(GRID), np.real(abs_square(psi0)),label='$\\nu=0$')
    plt.plot(np.real(GRID), np.real(abs_square(psi1)),label='$\\nu=0.5$')
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(FIGURE_PATH+'/density.png', dpi=300, bbox_inches='tight')
    plt.close()

    psi0  = black_soliton(-10.) * black_soliton(10.)
    run_and_plot(psi0, num_steps=100, file_name=FIGURE_PATH+'/nu0.0.png')

    run_and_plot_sym(0.3)
    run_and_plot_sym(0.5)
    run_and_plot_sym(0.8)
    run_and_plot_sym(0.95)

    psi0  = (dark_soliton(-10., nu=0.3)
             * dark_soliton(-2., nu=-0.05)
             * dark_soliton(6., nu=0.967746031217134))

    run_and_plot(psi0, num_steps=10000, dt=DT/2,
                 file_name=FIGURE_PATH+'/nu0.3nu0.05.png')
    # NOTE Changing the `dt` argument also changes the time scale.

if __name__ == "__main__":
    main()
