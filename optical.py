import numpy as np
from numpy import sqrt,pi,nan,inf,sign,abs,exp,log,sin,cos
import scipy, scipy.optimize, functools
import sellmeier

def phaseshift(λ,L,sell='air',temp=20): # λ in nm, L in mm # return total phase in radians
    return 2*pi*sellmeier.index(λ,sell,temp=temp)*1e6*L/λ
def coherencelength(Δλ,λ): # in mm (Δλ,λ in nm)
    assert Δλ<λ, 'arguments in wrong order'
    return 1e-6*λ**2/Δλ
def wavelength2energy(λ): # returns photon energy in eV for λ in nm
    h,c = 4.1356675e-15,299792458 # eV·s,m/s
    return h*c/(λ*1e-9)
def wavelength2wavenumber(λ): # returns 𝜈 in 1/cm for λ in nm
    return 1e7/λ
def wavenumber2wavelength(𝜈): # returns λ in nm for 𝜈 in 1/cm
    return 1e7/𝜈
def wavenumberbandwidth(Δλ,λ): # returns Δ𝜈 in 1/cm for Δλ,λ in nm
    assert Δλ<λ, 'arguments in wrong order'
    return Δλ*1e7/λ**2
def frequencybandwidth(Δλ,λ): # returns df in GHz for Δλ,λ in nm, or df in Hz for Δλ,λ in m
    assert Δλ<λ, 'arguments in wrong order'
    return Δλ*299792458/λ**2 # in GHz
def wavelengthbandwidth(Δf,λ,wavenumbers=False): # Δf in GHz, λ in nm
    if wavenumbers: # assume Δf in cm⁻¹ # Δ𝜈 = Δλ*1e7/λ**2
        return 1e-7*Δf*λ**2
    return Δf/299792458*λ**2 # in nm
def wavelength2thz(λ): # returns 𝜈 in THz for λ in nm
    return 0.001*299792458/λ
def thz2wavelength(𝜈): # returns λ in nm for 𝜈 in THz
    return 0.001*299792458/𝜈
def wavelength2ghz(λ): # returns 𝜈 in GHz for λ in nm
    return 299792458/λ
def ghz2wavelength(𝜈): # returns λ in nm for 𝜈 in GHz
    return 299792458/𝜈
def wavelength2rgb(λ, gamma=0.8, colour=False): # returns (r,g,b) given λ in nm
    if colour:
        import colour
        xyz = colour.wavelength_to_XYZ(λ) # print('xyz',xyz)
        return [max(0,min(1,x)) for x in colour.XYZ_to_sRGB(xyz)]
    # http://www.physics.sfasu.edu/astro/color.html
    if 380 <= λ <= 440:
        a = 0.3 + 0.7 * (λ - 380) / 60
        R, G, B = ((-(λ - 440) / 60) * a) ** gamma, 0.0, (1.0 * a) ** gamma
    elif 440 < λ <= 490:
        R, G, B = 0.0, ((λ - 440) / 50) ** gamma, 1.0
    elif 490 < λ <= 510:
        R, G, B = 0.0, 1.0, (-(λ - 510) / 20) ** gamma
    elif 510 < λ <= 580:
        R, G, B = ((λ - 510) / 70) ** gamma, 1.0, 0.0
    elif 580 < λ <= 645:
        R, G, B = 1.0, (-(λ - 645) / 65) ** gamma, 0.0
    elif 645 < λ <= 750:
        a = 0.3 + 0.7 * (750 - λ) / 105
        R, G, B = (1.0 * a) ** gamma, 0.0, 0.0
    else:
        R, G, B = 0.0, 0.0, 0.0
    return [max(0,min(1,x)) for x in [R,G,B]]
    # wavelengths = np.arange(380, 751, 1)
    # colors = [wavelength2rgb(w) for w in wavelengths]
    # # colors = [wavelength2rgb(w,colour=1) for w in wavelengths]
    # color_array = np.array([colors for _ in range(100)])
    # plt.figure(figsize=(12, 6))
    # plt.imshow(color_array, extent=[380, 750, 0, 1], aspect='auto')
    # plt.xlabel("Wavelength (nm)")
    # plt.yticks([])
    # plt.show()
def frequency2pitch(frequency):
    midi_note = 12 * np.log2(frequency / 440) + 69 # Calculate the MIDI note number
    nearest_midi_note = round(midi_note) # Determine the nearest whole MIDI note
    cent_deviation = ((midi_note - nearest_midi_note) * 100) # Calculate the cent deviation from the nearest MIDI note
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"] # The notes in an octave
    note_index = nearest_midi_note % 12 # Calculate the note name and octave
    octave = (nearest_midi_note // 12) - 1
    note_name = notes[note_index] + str(octave)
    return note_name+f"{cent_deviation:+.2f}"
def pitch2frequency(note_with_cents):
    def split_string_float(s):
        import re
        match = re.match(r"([A-Ga-g#]+)(\d+)([+-]\d+(\.\d+)?)?", s)
        note, octave, alteration = match.groups()[0:3] if match else (s, '', '0')
        return note, int(octave), float(alteration if alteration else '0')
    note, octave, cents = split_string_float(note_with_cents)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] # MIDI note numbers for the notes
    assert note in notes, f"{note} not recognized"
    midi_note = notes.index(note) + (octave + 1) * 12
    frequency = 440 * 2 ** ((midi_note - 69) / 12) # Calculate the frequency
    frequency *= 2 ** (cents / 1200) # Adjust for cents
    return frequency
def wavelength2pitch(λ): # returns musical pitch and octave given λ in nm
    return frequency2pitch(1e9*wavelength2ghz(λ))
def pitch2wavelength(pitch): # returns optical wavelength in nm given musical pitch and octave
    return ghz2wavelength(1e-9*pitch2frequency(pitch))
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
def besselj(n,β): # sum (k=-inf to inf) of besselj(k,β)² = 1
    return scipy.special.jv(n,β)
def sidebandpower(β): # β = π*Vin/Vpi = modulation index (https://en.wikipedia.org/wiki/Frequency_modulation#Bessel_functions)
    return besselj(1,β)**2 / besselj(0,β)**2
def sidebandphase(powerratio):
    wx = np.linspace(0,scipy.special.jn_zeros(0,1)[0],10000) # np.linspace(0,2.4048,24049)
    from wavedata import Wave
    return Wave(sidebandpower(wx),wx).xaty(powerratio)
def dbm(volts,Z=50):
    return 10*np.log10(volts**2/Z/0.001)
def pidbm(vpi,Z=50): # power in dBm to get peak-to-peak π phase shift
    return 10*np.log10(125*vpi**2/Z)
def vrms(powerindbm,Z=50):
    mW = 10**(powerindbm/10)
    return np.sqrt(Z*0.001*mW)
def vpeak(powerindbm,Z=50):
    mW = 10**(powerindbm/10)
    return np.sqrt(2*Z*0.001*mW)
def vpp(powerindbm,Z=50):
    return 2*vpeak(powerindbm,Z)
def resonantfrequency(L,n=1,units=True): # L in mm, returns f in GHz
    Δf = 299792458/(2*n*L*0.001)*1e-9
    return (Δf,'GHz') if units else Δf
def thermalfreespectralrange(λ,L,sell,temp=20): # λ in nm, L in mm
    n,Δn,Δε = sellmeier.index(λ,sell,temp), sellmeier.index(λ,sell,temp+1)-sellmeier.index(λ,sell,temp), sellmeier.expansion(temp+1,sell)-sellmeier.expansion(temp,sell)
    ΔT = 0.5*λ/(1e6*L)/(Δn+n*Δε) # print('n,Δn,Δε',n,Δn,Δε)
    return ΔT # in °C
def freespectralrange(λ,L,sell): # λ in nm, L in mm
    Δλ = λ**2/(1e6*2*L*sellmeier.groupindex(λ,sell=sell))
    return Δλ
def fabryperotTvsλ(λ0=1064,d=1,n=1.5,F=40,fsrs=20,num=1001,plot=False): # λ0 in nm, d in mm, F = 4GR/(1-GR)² where G = propagation loss, R = one facet reflectivity = (n-1)²/(n+1)²
    d = round(d*1e6*n/λ0)*λ0/n/1e6
    Δλ = λ0**2/(2*d*1e6*n)
    wx = np.linspace(λ0-fsrs*Δλ,λ0+fsrs*Δλ,num)
    from wavedata import Wave
    w = Wave( 1/(1+F*sin(2*pi*d*1e6*n/wx)**2), wx )
    if plot: w.plot()
    return w
# A ≡ Tmin/Tmax = (1-GR)²/((1-GR)²+4GR) → GR = (1-√A)/(1+√A)
# B ≡ Rmin/Rmax = R(1-G)²/(1-GR)² ((1-GR)²+4GR)/(R(1-G)²+4GR)
# B = R(1-G)²/(R(1-G)²+4GR)/A → ½(G+1/G) = C ≡ 1+2/(1/(AB)-1), G = C-√(C²-1)
def fabryperotlossreflectionfactor(A):
    # A = min/max from Fabry-Perot scan
    # I(λ) = c/[(1-GR)² + 4GRsin²(2πλ/ΔλFSR)] where GR = lossreflectionfactor
    GR = (1-sqrt(A))/(1+sqrt(A))
    assert np.isclose(A,(1-GR)**2/(1+GR)**2,equal_nan=True)
    return GR
def fabryperotlossfactor(A,B):
    # A = min/max from forward transmission Fabry-Perot scan
    # B = min/max from backward reflection scan
    C = 1+2/(1/(A*B)-1)
    return C - sqrt(C*C-1)
def fabryperotreflectionfactor(A,B):
    # A = min/max from forward transmission Fabry-Perot scan
    # B = min/max from backward reflection scan
    GR = fabryperotlossreflectionfactor(A)
    G = fabryperotlossfactor(A,B)
    R = GR/G
    assert np.isclose(B,R*(1-G)**2/(1-GR)**2*((1-GR)**2+4*GR)/(R*(1-G)**2+4*GR),equal_nan=True)
    return R
def fabryperotloss(A,nn,verbose=False,dB=True):
    gr = fabryperotlossreflectionfactor(A)
    r = (nn-1)**2 / (nn+1)**2
    g = gr/r
    if verbose: print(f"A:{A:.3f}, GR:{gr:.4f}, R:{r:.4f}, G:{g:.3f}({10*np.log10(g):.2f}dB)")
    return 10*np.log10(g) if dB else g
def interferenceintensity(I1,I2,θ):
    # Iout = I1 + I2 + 2√(I1 I2)cosθ where I1 = I2 = Iin/4
    return I1 + I2 + 2*np.sqrt(I1*I2)*np.cos(θ)
def interferencevisibity(I1,I2):
    return 2*np.sqrt(I1*I2)/(I1+I2)
def NA(λ,ω,index=1): # λ in nm, ω in µm
    return λ/1000/(index*pi*ω)
def rayleighrange(λ,ω): # λ in nm, ω in µm
    return 2*pi*ω**2/(λ/1000) # in µm
def anglecouplingloss(NA,θ): # see OFR fiber formula or marcuse1977 - Loss Analysis of Single-Mode Fiber Splices
    return exp(-(θ/NA)**2)
def gapcouplingloss(λ,ω,gap): # λ in nm, ω,gap in µm # marcuse1977 - Loss Analysis of Single-Mode Fiber Splices
    b = 2*pi*ω**2/(λ/1000) # b = confocal parameter = 2 × Rayleigh range = kω²
    return 1/(1+(gap/b)**2)
def brewstersangle(n,deg=True):
    return np.arctan(n)*180/pi if deg else np.arctan(n)
def planargratingcouplerperiod(λ,sell,θ=0,sellair='air'): # λ in nm, θ = angle in radians from vertical
    # Bragg condition: (2π/λ)sinθ + 2π/Λ = 2πn/λ, ref:Micromachines 2020,11,666
    return 1e-3/(sellmeier.index(λ,sell)/λ-sin(θ)*sellmeier.index(λ,sellair)/λ) # period in µm
def EOindex(λ,sell='ktpz',temp=20,E=1,low=True): # index change corresponding to E in kV/mm (or V/µm)
    # KTP from BierleinVanherzeele89
    # LN from www.redoptronics.com/LiNbO3-crystal-electro-optical.html:
    #   r33 = 32 pm/V, r13 = r23 = 10 pm/V, r22 = -r11 = 6.8 pm/V at low frequency and r33 = 31 pm/V, r31(typo) = 8.6 pm/V, r22 = 3.4 pm/V at high electric frequency.
    if low:
        r = {'ktpz':36.3,'ktpy':15.7,'ktpx':9.5,'lnz':32,'lny':10,'lnx':10}[sell.replace('wg','')]
    else:
        r = {'ktpz':35,'ktpy':13.8,'ktpx':8.8,'lnz':32,'lny':8.6,'lnx':8.6}[sell.replace('wg','')]
    return 0.5 * sellmeier.index(λ,temp=temp,sell=sell)**3 * 1e-12*r * 1e6*E
def bulkvpi(λ,sell,gap): # gap in mm, Vpi in V·cm
    Δn = EOindex(λ,sell,temp=20,E=1,low=True)
    phase = 2*np.pi*Δn*1e7/λ
    return 1000*gap*pi/phase # in V·cm
def bulkcapacitance(sell,A,d=1): # A in mm², d in mm, C = ε ε0 A / d
    ε = {'lnz':43,'lnx':28,'ktp':13}[sell]
    ε0 = 8.854 # pF/m
    return ε*ε0*1e-3*A/d # in pF
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
def braggperiod(λ,sell,m): # λ in nm, m = bragg order
    return λ/(2000*sellmeier.index(λ,sell)/m) # bragg wavelength in nm
def sinc(x):
    return np.sinc(x/pi)
def gauss(x,sigma=1,fwhm=None):
    if fwhm is not None: return np.exp( -4*log(2)*x**2/fwhm**2 )
    return np.exp( -x**2 / sigma**2 / 2 )
def sbend(x,L,w):
    return np.where( x<0, 0, np.where( L<x, w, (1-np.cos(np.pi*x/L)) * w/2 ))
def sbendroc(L,w):
    return 2/np.pi**2 * L**2/w

if __name__ == '__main__':
    print(phaseshift(1064,1.064))
    print(sbendroc(2450,127-6))
