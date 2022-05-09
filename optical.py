import numpy as np
from numpy import sqrt,pi,nan,inf,sign,abs,exp,log,sin,cos
import scipy, scipy.optimize, functools
from sellmeiers import index

def phaseshift(λ,L,sell='air',temp=20): # λ in nm, L in mm # return total phase in radians
    return 2*pi*index(λ,sell,temp=temp)*1e6*L/λ
def coherencelength(Δλ,λ): # in mm (Δλ,λ in nm)
    assert Δλ<λ, 'arguments in wrong order'
    return 1e-6*λ**2/Δλ
def frequencybandwidth(Δλ,λ): # returns df in GHz for dlambda,lambda in nm, or df in Hz for dlambda,lambda in m
    assert Δλ<λ, 'arguments in wrong order'
    return Δλ*299792458/λ**2 # in GHz
def gaussianbeam(λ,ω0,r,z): # all units the same
    # returns complex amplitude of field at the particular location
    # https://en.wikipedia.org/wiki/Gaussian_beam
    zr, k = pi*ω0**2/λ, 2*pi/λ # print('ω0',ω0,'zr',zr)
    def Rinv(z):
        return z/(z**2 + zr**2)
    def ω(z):
        return ω0*sqrt(1+(z/zr)**2)
    return (ω0/ω(z)) * np.exp(-(r/ω(z))**2) * np.exp(-1j*( k*z + 0.5*k*r**2*Rinv(z) + np.arctan(z/zr) ))
def pulsewidth(λ,Δλ,transformlimited=True): # returns pulse width Δt in ns for λ,dλ in nm
    assert Δλ<λ, 'arguments in wrong order'
    if transformlimited:
        return 2*log(2)/pi/frequencybandwidth(Δλ,λ) # frequencybandwidth in GHz
    return λ**2/299792458/Δλ
def pulsebandwidth(Δt,λ): # returns dλ in nm for pulse width Δt in ns and λ in nm
    # Δλ/λ = Δf/f = Δf*λ/c = (1/Δt)*λ/c
    return λ**2/299792458/Δt
def transformlimitedbandwidth(dt,λ): # dt=tFWHM in ns, returns Δλ in nm
    # tFWHM * fFWHM = 2 log(2) / pi = 0.441271
    df = 2*log(2)/pi/dt # in GHz
    return df*λ**2/299792458
def transformlimitedpulse(f,f0,dt): # dt = tFWHM
    return exp(-pi**2 * (f-f0)**2 * dt**2 / log(2))
    # tFWHM * fFWHM = 2 log(2) / pi = 0.441271
    # transformlimitedpulse(1,0,0.441271/2) = 0.5
def pulsesum(x,t,f0,reptime,dt,sell='air'): # dt = tFWHM
    c = 299.792458 # in mm/ns
    num,frr = 10*int(reptime/dt),reptime
    fs = f0 + frr*np.linspace(-num,num,2*num+1)
    ns = (lambda x:1)(fs)
    return transformlimitedpulse(fs,f0,dt) * exp(1j*2*pi*fs*(x*ns/c-t))
def fwhmgaussian(x,x0,fwhm):
    return exp(-4 * log(2) * (x-x0)**2 / fwhm**2)
def loadfile(file,dtype=None,delimiter='\t',skip=0): # print(data.dtype.names) # print(data['Amplitude'])
    return np.array(np.genfromtxt(file, dtype=dtype, delimiter=delimiter, names=True, skip_header=skip).tolist())
def besselj(n,β): # sum (k=-inf to inf) of besselj(k,β)² = 1
    return scipy.special.jv(n,β)
def sidebandpower(β): # β = π*Vin/Vpi = modulation index (https://en.wikipedia.org/wiki/Frequency_modulation#Bessel_functions)
    return besselj(1,β)**2 / besselj(0,β)**2
def sidebandphase(powerratio):
    wx = np.linspace(0,scipy.special.jn_zeros(0,1)[0],10000) # np.linspace(0,2.4048,24049)
    from waves import Wave
    return Wave(sidebandpower(wx),wx).xaty(powerratio)
def dbm(volts,Z=50):
    return 10*np.log10(volts**2/Z/0.001)
def vrms(powerindbm,Z=50):
    mW = 10**(powerindbm/10)
    return np.sqrt(Z*0.001*mW)
def vpeak(powerindbm,Z=50):
    mW = 10**(powerindbm/10)
    return np.sqrt(2*Z*0.001*mW)
def vpp(powerindbm,Z=50):
    return 2*vpeak(powerindbm,Z)
def resonantfrequency(L,n=1,units=True): # L in mm, returns f in GHz
    c = 299792458
    Δf = 299792458/(2*n*L*0.001)*1e-9
    return (Δf,'GHz') if units else Δf
def freespectralrange(λ,L,sell): # λ in nm, L in mm
    Δλ = λ**2/(1e6*L*groupindex(λ,sell=sell))
    return Δλ
def fabryperotTvsλ(λ0=1064,d=1,n=1.5,F=40,fsrs=20,plot=False): # λ0 in nm, d in mm
    d = round(d*1e6*n/λ0)*λ0/n/1e6
    Δλ = λ0**2/(2*d*1e6*n)
    wx = np.linspace(λ0-fsrs*Δλ,λ0+fsrs*Δλ,1001)
    from waves import Wave
    w = Wave( 1/(1+F*sin(2*pi*d*1e6*n/wx)**2), wx )
    if plot: w.plot()
    return w
def fabryperotlossreflectionfactor(A): # A = min/max from Fabry-Perot scan, returns GR
    # I(λ) = c/[(1-GR)² + 4GRsin²(2πλ/ΔλFSR)] where GR = lossreflectionfactor
    return (1-sqrt(A))/(1+sqrt(A))
def fabryperotloss(A,nn,verbose=False,dB=True):
    gr = fabryperotlossreflectionfactor(A)
    r = (nn-1)**2 / (nn+1)**2
    g = gr/r
    if verbose: print(f"A:{A:.3f}, GR:{gr:.4f}, R:{r:.4f}, G:{g:.3f}({10*np.log10(g):.2f}dB)")
    return 10*np.log10(g) if dB else g
def NA(λ,ω): # λ in nm, ω in µm
    return λ/1000/(pi*ω)
def anglecouplingloss(NA,θ): # see OFR fiber formula or marcuse1977 - Loss Analysis of Single-Mode Fiber Splices
    return exp(-(θ/NA)**2)
def gapcouplingloss(λ,ω,gap): # λ in nm, ω,gap in µm # marcuse1977 - Loss Analysis of Single-Mode Fiber Splices
    b = 2*pi*ω**2/(λ/1000) # b = confocal parameter = 2 × Rayleigh range = kω²
    return 1/(1+(gap/b)**2)
def EOindex(λ,sell='ktpz',temp=20,E=1,low=True): # index change corresponding to E in kV/mm
    # KTP from BierleinVanherzeele89
    # LN from www.redoptronics.com/LiNbO3-crystal-electro-optical.html:
    #   r33 = 32 pm/V, r13 = r23 = 10 pm/V, r22 = -r11 = 6.8 pm/V at low frequency and r33 = 31 pm/V, r31(typo) = 8.6 pm/V, r22 = 3.4 pm/V at high electric frequency.
    if low:
        r = {'ktpz':36.3,'ktpy':15.7,'ktpx':9.5,'lnz':32,'lny':10,'lnx':10}[sell.replace('wg','')]
    else:
        r = {'ktpz':35,'ktpy':13.8,'ktpx':8.8,'lnz':32,'lny':8.6,'lnx':8.6}[sell.replace('wg','')]
    return 0.5 * index(λ,temp=temp,sell=sell)**3 * 1e-12*r * 1e6*E
def alpha2dB(α): # α in 1/mm, where loss = exp(-αL)
    return 100*α/log(10) # loss in dB/cm
def dB2alpha(dBpercm): # loss in dB/cm
    return dBpercm*log(10)/100 # α in 1/mm
def photonpower(λ): # λ in nm, returns power in nW per GHz rate
    h,c = 6.62607015e-34,299792458
    return h*c/(λ*1e-9)*1e18
def photonrate(λ): # λ in nm, returns rate in GHz per nW power
    return 1/photonpower(λ)
def braggwavelength(Λ,neff,m): # Λ in µm, m = bragg order
    return 2000*neff*Λ/m # bragg wavelength in nm
def extend(A,n=1):
    (a,b),(y,z) = A[:2],A[-2:]
    AA = [2*a-b,*A,2*z-y]
    return AA if n<=1 else extend(AA,n-1)
def sinc(x):
    return np.sinc(x/pi)
def gauss(x,sigma=1,fwhm=None):
    if fwhm is not None: return np.exp( -4*log(2)*x**2/fwhm**2 )
    return np.exp( -x**2 / sigma**2 / 2 )

if __name__ == '__main__':
    print(phaseshift(1064,1.064))
