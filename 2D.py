import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 0.5
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
import numpy as np

dase = 'rand'
L       =  40
npoints = 600
dt      =   0.005
dx      = L/npoints

grid = np.arange(-L/2,L/2,dx,complex)
grid = np.meshgrid(grid,grid)

def rand(x):
  ret = 0
  s = np.random.uniform(0,2*np.pi,40)
  for i in range(1,21):
    a = 0.1*np.exp(-i*i/50)
    ret += a*np.sin(i*x*(2*np.pi)/L+s[i-1])
  return ret

def grey_soliton_rand(nu=0.5,z0=0):
    gamma=1/np.sqrt(1-nu**2)
    ret = 1j*nu+1/gamma*np.tanh((grid[0]-z0-rand(grid[1]))/gamma)
    return ret*np.ones((npoints,npoints))

def grey_soliton(nu=0.5,z0=0):
    gamma=1/np.sqrt(1-nu**2)
    ret = 1j*nu+1/gamma*np.tanh((grid[0]-z0-0.1*np.sin(grid[1]*(2*np.pi)/L*5))/gamma)
    return ret*np.ones((npoints,npoints))

def dark_soliton(z0=0):
    return grey_soliton(nu=0,z0=z0)

def ring(R=5,nu=-0.5):
    gamma=1/np.sqrt(1-nu**2)
    r = np.sqrt(grid[0]*grid[0]+grid[1]*grid[1])
    ret = 1j*nu+1/gamma*np.tanh((r-R)/(gamma))
    return ret*np.ones((npoints,npoints))

def ro(psi):
    return np.conjugate(psi)*psi

def H2(psi):
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
        if i%50==0:
            np.save('data/{}{}'.format(dase,i//50), np.real(ro(psi)))
            plt.imshow(np.real(ro(psi)), cmap=plt.get_cmap("BuPu"), origin='lower', 
                extent=[-L/2, L/2-dx, -L/2, L/2-dx])
            plt.savefig('plots/{}{}.svg'.format(dase,i//50))
            plt.close()
    psi_ = np.fft.fft2(psi)
    psi_ = np.exp(-1j*dt*H1)*psi_
    t += dt/2
    psi  = np.fft.ifft2(psi_)
    psi  = np.exp(-1j*(dt/2)*ro(psi))*psi
    return psi

def main():
    #psi = grey_soliton(0.,-5)*grey_soliton(-0.,5)
    #psi = ring()
    psi = grey_soliton_rand(0.,-5)*grey_soliton_rand(-0.,5)
    psi = TimeEvolution(psi, 5001, dt_=dt)

if __name__ == "__main__":
    main()
