import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 0.5
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import scipy.linalg as LA
import numpy as np

L       =  40
npoints = 500
dx      = L/npoints
dt      =   0.025
tsteps  = 3000

grid  = np.arange(-L/2,L/2,dx,complex)

def H1diag(t):
    k = 2*np.pi*np.fft.fftfreq(npoints,d=dx)
    Es = k*k/2
    return Es

def H2diag(t,psi):
    return 1/2*np.conjugate(psi)*psi

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
    return 1j*nu+1/gamma*np.tanh((grid-z0)/(np.sqrt(2)*gamma))

def dark_soliton(z0=0):
    return grey_soliton(nu=0,z0=z0)

def TimeEvolution(psi0):
    all_psis = np.zeros((tsteps+1,npoints))
    t = 0
    psi = np.exp(-1j*(dt/2)*H2diag(t,psi0))*psi0
    t += dt/2
    all_psis[0] = np.real(np.conjugate(psi)*psi)
    for i in range(tsteps):
        psi_ = np.fft.fft(psi)
        psi_ = np.exp(-1j*dt*H1diag(t))*psi_
        t += dt/2
        psi  = np.fft.ifft(psi_)
        psi  = np.exp(-1j*dt*H2diag(t,psi))*psi
        t += dt/2
        all_psis[i+1] = np.real(np.conjugate(psi)*psi)
    return all_psis

def plot(psi0,name):
    all_psis = TimeEvolution(psi0)
    plt.imshow(all_psis, vmin = 0., vmax = 1.1, cmap=plt.get_cmap("BuPu"), origin='lower', 
           extent=[-L/2,L/2-dx, dt/2, (tsteps+1/2)*dt], aspect='auto')
    plt.xlabel('$x$ $[\epsilon]$')
    plt.ylabel('t')
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def plot_sym(nu):
    psi0  = grey_soliton(nu,-10)
    psi0 *= grey_soliton(-nu, 10)
    plot(psi0,'nu{}.svg'.format(nu))

def main():
    plot_sym(0.3)
    plot_sym(0.5)
    plot_sym(0.8)
    plot_sym(0.95)

    psi0  = grey_soliton( 0.3,  -10)
    psi0 *= grey_soliton(-0.05,   -2)
    psi0 *= grey_soliton( 0.967746031217134,  6)

    plot(psi0,'nu0.3nu0.05.svg')

if __name__ == "__main__":
    main()
