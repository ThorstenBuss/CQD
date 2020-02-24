import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 0.5
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"]
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import scipy.linalg as LA
import numpy as np
import os

myfontsize = 9

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size':myfontsize})
rc('text', usetex=True)
rc('legend', fontsize=myfontsize)

L       =  40
npoints = 600
dt      =   0.01

dx      = L/npoints
grid  = np.arange(-L/2,L/2,dx,complex)

def H1diag(t):
    k = 2*np.pi*np.fft.fftfreq(npoints,d=dx)
    Es = k*k/2
    return Es

def H2diag(t,psi):
    return np.conjugate(psi)*psi

def TimeStap(psi,t):
    psi  = np.exp(-1j*(dt/2)*H2diag(t,psi))*psi
    t   += dt/2
    psi_ = np.fft.fft(psi)
    psi_ = np.exp(-1j*dt*H1diag(t))*psi_
    t   += dt/2
    psi  = np.fft.ifft(psi_)
    psi  = np.exp(-1j*(dt/2)*H2diag(t,psi))*psi
    return psi

def grey_soliton(nu=0.5,z0=0):
    gamma=1/np.sqrt(1-nu**2)
    return 1j*nu+1/gamma*np.tanh((grid-z0)/(gamma))

def dark_soliton(z0=0):
    return grey_soliton(nu=0,z0=z0)

def TimeEvolution(psi0, tsteps_, dt_=dt):
    all_psis = np.zeros((tsteps_+1,npoints))
    t = 0
    psi = np.exp(-1j*(dt_/2)*H2diag(t,psi0))*psi0
    t += dt_/2
    all_psis[0] = np.real(np.conjugate(psi)*psi)
    for i in range(tsteps_):
        psi_ = np.fft.fft(psi)
        psi_ = np.exp(-1j*dt_*H1diag(t))*psi_
        t += dt_/2
        psi  = np.fft.ifft(psi_)
        psi  = np.exp(-1j*dt_*H2diag(t,psi))*psi
        t += dt_/2
        all_psis[i+1] = np.real(np.conjugate(psi)*psi)
    return all_psis

def plot(psi0, name, tsteps_, dt_=dt):
    all_psis = TimeEvolution(psi0, tsteps_, dt_)
    plt.imshow(all_psis, cmap=plt.get_cmap("BuPu"), origin='lower', 
           extent=[-L/2,L/2-dx, dt_/2, (tsteps_+1/2)*dt_], aspect='auto')
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('t')
    cbar = plt.colorbar()
    cbar.set_label(r'Density',labelpad=5,fontsize=20)
    plt.savefig(name,dpi=300)
    plt.close()

def plot_sym(nu):
    psi0  = grey_soliton(nu,-10)
    psi0 *= grey_soliton(-nu, 10)
    plot(psi0,'plots/nu{}.png'.format(nu),tsteps_=int(20/(nu*dt)))

def main():
    os.system('mkdir -p plots/{}')
    psi0  = dark_soliton(-10)
    psi0 *= dark_soliton( 10)
    plot(psi0,'plots/nu0.0.png',10000//2)

    plot_sym(0.3)
    plot_sym(0.5)
    plot_sym(0.8)
    plot_sym(0.95)

    psi0  = grey_soliton( 0.3,  -10)
    psi0 *= grey_soliton(-0.05,   -2)
    psi0 *= grey_soliton( 0.967746031217134,  6)

    plot(psi0,'plots/nu0.3nu0.05.png',10000,dt/2)

if __name__ == "__main__":
    main()
