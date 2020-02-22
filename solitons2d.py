import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 0.5
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import numpy as np

dase = 'rand1/'
L       =  64
npoints = 256
dt      =   0.005
dx      = L/npoints

grid = np.arange(-L/2,L/2,dx,complex)
grid = np.meshgrid(grid,grid)

def rand1d(x):
    ret = 0
    s = np.random.uniform(0,2*np.pi,L//4)
    for i in range(1,L//4):
        ret += np.sin(i*x*(2*np.pi)/L+s[i-1])
    return 0.01*ret


def rand2d():
    k  = 2*np.pi*np.fft.fftfreq(npoints,d=dx)
    k  = np.meshgrid(k,k)
    e  = -(k[0]*k[0]+k[1]*k[1])/(2*20)
    r_ = np.exp(e).astype(complex)
    s  = np.random.uniform(0,1,(npoints,npoints)).astype(complex)
    r_*= np.exp(1j*2*np.pi*s)
    r  = np.fft.ifft2(r_)
    return 50*r+1

def grey_soliton_rand_pos(nu=0.,z0=0):
    gamma=1/np.sqrt(1-nu**2)
    ret = 1j*nu+1/gamma*np.tanh((grid[0]-z0-rand1d(grid[1]))/gamma)
    return ret*np.ones((npoints,npoints))

def grey_soliton(nu=0.,z0=0):
    gamma=1/np.sqrt(1-nu**2)
    ret = 1j*nu+1/gamma*np.tanh((grid[0]-z0)/gamma)
    return ret*np.ones((npoints,npoints))

def ring(R=5,nu=-0.5):
    gamma=1/np.sqrt(1-nu**2)
    r = np.sqrt(grid[0]*grid[0]+grid[1]*grid[1])
    ret = 1j*nu+1/gamma*np.tanh((r-R)/gamma)
    return ret*np.ones((npoints,npoints))

def ro(psi):
    return np.conjugate(psi)*psi

def TimeEvolution(psi0, tsteps_, dt_=dt):
    k = 2*np.pi*np.fft.fftfreq(npoints,d=dx)
    k = np.meshgrid(k,k)
    H1 = (k[0]*k[0]+k[1]*k[1])/2
    t = 0
    psi = np.exp(-1j*(dt_/2)*ro(psi0))*psi0
    t += dt_/2
    for i in range(tsteps_):
        psi_ = np.fft.fft2(psi)
        psi_ = np.exp(-1j*dt_*H1)*psi_
        t += dt_/2
        psi  = np.fft.ifft2(psi_)
        psi  = np.exp(-1j*dt_*ro(psi))*psi
        t += dt_/2
        if i%10==0:
            np.save('data/{}{}'.format(dase,i//10), psi)
            plt.imshow(np.real(ro(psi)), cmap=plt.get_cmap("BuPu"), origin='lower', 
                extent=[-L/2, L/2-dx, -L/2, L/2-dx],vmin=0, vmax=1.1)
            plt.colorbar()
            plt.savefig('plots/{}{}.png'.format(dase,i//10))
            plt.close()
    psi_ = np.fft.fft2(psi)
    psi_ = np.exp(-1j*dt*H1)*psi_
    t += dt/2
    psi  = np.fft.ifft2(psi_)
    psi  = np.exp(-1j*(dt/2)*ro(psi))*psi
    return psi

def main():
    #psi = grey_soliton(0.,-10)*grey_soliton(-0.,10)
    #psi = ring()
    psi = grey_soliton_rand_pos(0.,-10)*grey_soliton_rand_pos(-0.,10)
    #psi = rand2d()*grey_soliton(0.,-10)*grey_soliton(-0.,10)
    psi = TimeEvolution(psi, 10001, dt_=dt)

if __name__ == "__main__":
    main()
