import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 0.5
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import numpy as np
import os

dase = 'hallo'
L       =  64
npoints = 256
dx      = L/npoints

grid = np.arange(-L/2,L/2,dx,complex)
grid = np.meshgrid(grid,grid)

def rand1d(x):
    ret = 0
    s = np.random.uniform(0,2*np.pi,L//4)
    for i in range(1,L//4):
        ret += np.sin(i*x*(2*np.pi)/L+s[i-1])
    return 0.02*ret


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

def plot(grid,nx_grid,ny_grid,i):
    fulllengthy = ny_grid
    factory = 8.
    stepsy = (int)(fulllengthy/factory)
    maxLengthy = (int)(ny_grid*(stepsy*factory)/fulllengthy)

    xtick_locs = np.r_[0:maxLengthy:1j*(stepsy+1)]
    xtick_lbls = np.r_[0:stepsy*factory:1j*(stepsy+1)]

    fulllengthx = nx_grid
    factorx = 8.
    stepsx = (int)(fulllengthx/factorx)
    maxLengthx = (int)(nx_grid*(stepsx*factorx)/fulllengthx)

    ytick_locs = np.r_[0:maxLengthx:1j*(stepsx+1)]
    ytick_lbls = np.r_[0:stepsx*factorx:1j*(stepsx+1)]     


    ##### Density #######################################
    plt.imshow(np.abs(grid)**2,interpolation='nearest', origin='lower left', label = r'', vmin=0, vmax=1.1)
    cbar =  plt.colorbar()
    cbar.set_label(r'Density',labelpad=5,fontsize=20)
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('$y$ $[\\xi]$')
    plt.savefig('plots/{}/density/{}.png'.format(dase,i),dpi=300)
    plt.close()


    #### Phase ########################################
    plt.imshow(np.angle(grid),interpolation='nearest', origin='lower left', label = r'', vmin=-np.pi, vmax=np.pi, cmap ='hsv')
    cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    cbar.set_label(r'Phase angle',labelpad=5,fontsize=20)
    plt.xlabel('$x$ $[\\xi]$')
    plt.ylabel('$y$ $[\\xi]$')
    plt.savefig('plots/{}/phase/{}.png'.format(dase,i),dpi=300)
    plt.close()

def TimeEvolution(psi0, tsteps_, dt):
    k = 2*np.pi*np.fft.fftfreq(npoints,d=dx)
    k = np.meshgrid(k,k)
    H1 = (k[0]*k[0]+k[1]*k[1])/2
    t = 0
    psi = np.exp(-1j*(dt/2)*(np.conjugate(psi0)*psi0-1))*psi0
    t += dt/2
    for i in range(tsteps_):
        psi_ = np.fft.fft2(psi)
        psi_ = np.exp(-1j*dt*H1)*psi_
        t += dt/2
        psi  = np.fft.ifft2(psi_)
        psi  = np.exp(-1j*dt*(np.conjugate(psi)*psi-1))*psi
        t += dt/2
    psi_ = np.fft.fft2(psi)
    psi_ = np.exp(-1j*dt*H1)*psi_
    t += dt/2
    psi  = np.fft.ifft2(psi_)
    psi  = np.exp(-1j*(dt/2)*(np.conjugate(psi)*psi-1))*psi
    return psi

def main():
    os.system('mkdir -p data/{}'.format(dase))
    os.system('mkdir -p plots/{}/phase'.format(dase))
    os.system('mkdir -p plots/{}/density'.format(dase))

    #psi = grey_soliton(0.,-10)*grey_soliton(-0.,10)
    #psi = ring()
    psi = grey_soliton_rand_pos(0.,-10)*grey_soliton_rand_pos(-0.,10)
    #psi = rand2d()*grey_soliton(0.,-10)*grey_soliton(-0.,10)

    plot(psi,npoints,npoints,0)
    np.save('data/{}/{}'.format(dase,0), psi)
    for i in range(1000):
        psi = TimeEvolution(psi, 5, 0.02)
        plot(psi,npoints,npoints,i+1)
        np.save('data/{}/{}'.format(dase,i+1), psi)

if __name__ == "__main__":
    main()