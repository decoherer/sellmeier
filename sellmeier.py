#!/usr/bin/env python
# coding: utf8
import numpy as np
from numpy import sqrt,pi,nan,inf,sign,abs,exp,log,sin,cos
import scipy, scipy.optimize, functools
from ridgesellmeier import *
from optical import *
from qpm import Qpm

# Lithium Niobate nonlinear coefficients:
#  ChatGPT d11=6.8, d22=9.3, d31=6.6, d32=8.5, d33=30.9
#  INRAD d22=2.4, d31=-4.52, d33=31.5
#  pmoptics d33=34.4, d31=d15=5.95, d22=3.07
# Potassium Titanyl Phosphate (KTP) has the following non-zero second-order nonlinear coefficients:
#  ChatGPT d11=d22=3.0, d31=d32=3.5, d33=16.9
#  Northrop-Grumman d31=6.5, d32=5.0, d33=13.7, d24=7.6, d15=6.1
#  msesupplies d31=2.54, d32=4.35, d33=16.9, d24=3.64, d15=1.91
# laser components d31=2.54, d32=4.35, d33=16.9, d24=3.64, d15=1.91

def groupindex(λ,sell,temp=20,Δλ=0.1): # ng = n(λ) - λ dn/dλ = n(ω) + ω dn/dω
    Δn = index(λ+Δλ/2,sell,temp=temp) - index(λ-Δλ/2,sell,temp=temp) # print('n,λ,Δn,Δλ',index(λ,sell,temp=temp),λ,Δn,Δλ)
    return index(λ,sell,temp=temp) - λ*Δn/Δλ
def groupvelocity(λ,sell,temp=20): # in units of c
    return 1/groupindex(λ,sell=sell,temp=temp)
def inversegroupvelocity(λ,sell,temp=20): # seconds per meter
    return 1/299792458*1/groupvelocity(λ,sell,temp)
def groupvelocitydelay(λ,sell,temp=20): # seconds per meter
    return inversegroupvelocity(λ,sell+'z',temp) - inversegroupvelocity(λ,sell+'y',temp)
    # return 1/299792458*(1/groupvelocity(λ,sell+'z',temp)-1/groupvelocity(λ,sell+'y',temp))
def shggroupvelocitypsdelaytime(λ,sell,temp=20,L=10): # L in mm, delay in picoseconds
    u1,u3 = [inversegroupvelocity(w,sell,temp) for w in (λ,λ/2)]
    return 1e12 * (u3-u1) * L/1000
def compensationlength2psdelay(λ,L=10,sell1='ktpz',sell2='ktpy',temp=20): # L in mm, returns delay time in ps
    u1,u2 = [inversegroupvelocity(λ,sell,temp) for sell in (sell1,sell2)]
    return 1e12 * (u1-u2) * L/1000
def psdelay2compensationlength(λ,dt,sell1='ktpz',sell2='ktpy',temp=20): # dt in ps, returns Lc in mm
    u1,u2 = [inversegroupvelocity(λ,sell,temp) for sell in (sell1,sell2)]
    return 1e-9 * dt / (u1-u2)
def interp(x,nx,ny):
    if nx[1]<nx[0]: return np.interp(-x,-nx,ny)
    return np.interp(x,nx,ny,left=None,right=None) # interp will not work if not strictly increasing
def qpmwavelengths(w1,w2,w3=None):
    assert (w1 is not None) or (w3 is not None and w2 is not None)
    if w3 is not None and w1 is None:
        w1 = 1./(1./np.abs(w3)-1./np.abs(w2))
    if w3 is not None and w2 is None:
        w2 = 1./(1./np.abs(w3)-1./np.abs(w1))
    else:
        w2 = w2 if w2 is not None else np.abs(w1)
        w3 = 1./(1./np.abs(w1)+1./np.abs(w2))
    return w1,w2,w3
def polingperiodktpwg(w1,w2,temp,npy,npz,Type='zzz',qpmargs=None):
    w1,w2,w3 = qpmwavelengths(w1,w2)
    if qpmargs is not None:
        assert npy is None and npz is None, "can't use npy,npz with qpmargs in polingperiodktpwg"
        import modes
        return modes.qpm.Λ(w1,w2,temp=temp,Type=Type,verbose=False,**qpmargs)
    kwg = 1/polingperiod(w1,w2,sell='ktpwg',Type=Type,temp=temp)
    knp = 1/polingperiod(npy,npz,sell='ktpwg',Type='yzy',temp=20)
    a = interp(2*w3,dkx,dkii if Type in [2,'yzy'] else dki)
    # if Type in [1,'yyz']: a = interp(2*w3,dkx,dkii)/2 + interp(2*w3,dkx,dki)/2 # rough estimate (actually worse for BCT1710-A26,27)
    if Type in [1,'yyz']: a = 0
    return 1/(kwg - a*knp)
def polingperiod(w1=None,w2=None,sell='ktpwg',Type=0,temp=20,npy=None,npz=None,
    w3=None,qpmargs=None,Λ=None,returntuple=False,show=False):
    assert not isinstance(w2, str), 'check order of arguments in polingperiod(w1,w2,sell)'
    if show:
        pp = polingperiod(w1,w2,sell,Type,temp,npy,npz,w3,qpmargs,Λ,returntuple,show=False)
        print(f"Λ = {pp:g} µm")
        return pp
    # type 0 sfg period in um, w1,w2 in nm, temp in °C
    if Λ is not None: # solve for an input argument given poling period
        wx = np.linspace(600,2600,2001)
        if qpmargs is not None: raise NotImplementedError
        if w1 is None: # print('solving for w1')
            def f(x): return 1/polingperiod(x,w2,sell,Type,temp,npy=npy,npz=npz) - 1/Λ
        elif w1==-np.inf: # print('solving for w1')
            def f(x): return 1/polingperiod(-x,w2,sell,Type,temp,npy=npy,npz=npz) - 1/Λ
        elif w2 is None: # print('solving for w2')
            def f(x): return 1/polingperiod(w1,x,sell,Type,temp,npy=npy,npz=npz) - 1/Λ
        elif temp is None: # print('solving for temp')
            def f(x): return 1/polingperiod(w1,w2,sell,Type,x,npy=npy,npz=npz) - 1/Λ
            wx = np.linspace(-100,500,301)
        elif npy is None: # print('solving for npy')
            assert 'ktpwg'==sell
            def f(x): return 1/polingperiod(w1,w2,sell,Type,temp,npy=x,npz=npz) - 1/Λ
        elif npz is None: # print('solving for npz')
            assert 'ktpwg'==sell
            def f(x): return 1/polingperiod(w1,w2,sell,Type,temp,npy=npy,npz=x) - 1/Λ
        else:
            assert 0, 'wrong arguments in polingperiod'
        wy = f(wx)
        zero_crossings = np.where(np.diff(np.signbit(wy)))[0] # https://stackoverflow.com/a/29674950
        values = tuple(scipy.optimize.brentq(f,wx[z],wx[z+1]) for z in zero_crossings)
        # if debug:
        #     print('zero_crossings,values',zero_crossings,values) #wx[z],wy[z],f(wx[z]),wx[z+1],wy[z+1],f(wx[z+1]))
        #     from wavedata import Wave
        #     Wave(np.clip(1/(wy+1/Λ),-100,100),wx).plot(waves=[Wave(Λ,wx)])
        if 0==len(zero_crossings): return np.nan
        if not 1==len(zero_crossings) and not returntuple:
            print(f'multiple values found for poling period Λ={Λ:g}µm, values:',values) # print('returntuple',returntuple,'values',values)
        return values if returntuple else values[0]
    w1,w2,w3 = qpmwavelengths(w1,w2,w3)
    npz = npz if npz is not None else npy
    if npy is not None or qpmargs is not None:
        if 'lnwg'==sell:
            import modes
            return modes.qpm.Λ(w1,w2,temp=temp,sell='ln',Type=Type,verbose=False,**qpmargs)
        assert 'ktpwg'==sell
        return polingperiodktpwg(w1,w2,temp,npy,npz,Type,qpmargs)
    if 0==Type:
        return (1-expansion(temp,sell))/( index(w3,sell,temp)/w3 - index(w2,sell,temp)/w2 - index(w1,sell,temp)/w1 )/1000
    if 'zzz'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'z',temp)/w3 - index(w2,sell+'z',temp)/w2 - index(w1,sell+'z',temp)/w1 )/1000
    if 'z0z'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'z',temp)/w3 - 1/w2 - index(w1,sell+'z',temp)/w1 )/1000
    if 'y0z'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'z',temp)/w3 - 1/w2 - index(w1,sell+'y',temp)/w1 )/1000
    if 1==Type or 'yyz'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'z',temp)/w3 - index(w2,sell+'y',temp)/w2 - index(w1,sell+'y',temp)/w1 )/1000
    if 2==Type or 'yzy'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'y',temp)/w3 - index(w2,sell+'z',temp)/w2 - index(w1,sell+'y',temp)/w1 )/1000
    if 'zyy'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'y',temp)/w3 - index(w2,sell+'y',temp)/w2 - index(w1,sell+'z',temp)/w1 )/1000
    if 'yyy'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'y',temp)/w3 - index(w2,sell+'y',temp)/w2 - index(w1,sell+'y',temp)/w1 )/1000
    if 'xxz'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'z',temp)/w3 - index(w2,sell+'x',temp)/w2 - index(w1,sell+'x',temp)/w1 )/1000
    if 'xzx'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'x',temp)/w3 - index(w2,sell+'z',temp)/w2 - index(w1,sell+'x',temp)/w1 )/1000
    if 'zxx'==Type:
        return (1-expansion(temp,sell))/( index(w3,sell+'x',temp)/w3 - index(w2,sell+'x',temp)/w2 - index(w1,sell+'z',temp)/w1 )/1000
    a1,a2,a3 = Type if 3==len(Type) else Type.replace('-',' ').split()
    return (1-expansion(temp,sell))/( index(w3,sell+a3,temp)/w3 - index(w2,sell+a2,temp)/w2 - index(w1,sell+a1,temp)/w1 )/1000
    # assert 0, f'polingperiod(Type={Type}) not implemented'
pp = np.vectorize(polingperiod)
def polingperiod2wavelength(w1=None,w2=None,sell='ktpwg',Type=0,temp=20,npy=None,npz=None,*,Λ): # Λ is a required keyword argument 
    return polingperiod(w1,w2,sell,Type,temp=temp,npy=npy,npz=npz,Λ=Λ)
def polingperiod2temperature(w1,w2=None,sell='ktpwg',Type=0,npy=None,npz=None,w3=None,*,Λ):
    w1,w2,_ = qpmwavelengths(w1,w2,w3)
    return polingperiod(w1,w2,sell,Type,temp=None,npy=npy,npz=npz,Λ=Λ)
def polingperiod2nonpoled(w1,w2=None,sell='ktpwg',Type=0,temp=20,npy=None,npz=None,w3=None,*,Λ):
    w1,w2,_ = qpmwavelengths(w1,w2,w3)
    return polingperiod(w1,w2,sell,Type,temp=temp,npy=npy,npz=npz,Λ=Λ)
def expansion(temp,sell):
    sell = sell.replace('wg','')
    if 'ln' in sell and sell[-1] in 'xy':
        return 5e-6 * (temp-20)  ## ranges from 2e-6 to 7e-6, see Browder77,Howlander06
    if 'ln' in sell:
        return 15.4e-6 * (temp-25) + 5.3e-9 * (temp-25)**2 ## Jundt97
    if 'lt' in sell:
        return 1.6e-5 * (temp-20) + 7e-9 * (temp-20)**2
    if 'rta' in sell:
        return 15.1e-6 * (temp-20)
    if 'rtp' in sell:
        return expansion(temp,'rta')
    if 'ktp' in sell or any(s in sell for s in ['eman','kat','fan','van','frad','che']):
        ##alpha = 11e-6 * (temp-20)
        return 6.7e-6 * (temp-25) + 11e-9 * (temp-25)**2  ## x-prop emanueli03
    if 'kta' in sell:
        return 7.6e-6 * (temp-25) + 8.4e-9 * (temp-25)**2  ## x-prop emanueli03
    assert 20==temp, 'no thermal expansion defined for '+sell+' at temp = '+str(temp)
    # if not 20==temp: print('zero thermal expansion',sell)
    return 0
def sellmeier(x,a,b,c,d):
    return sqrt(a+b/(1-c/x**2)-d*x**2)
def sellmeier2(x,a,b,c,d):
    return sqrt(a+b/(x**2-c)-d*x**2)
def sellmeier3(x,a,b,c,d,e,f):
    return np.sqrt(np.abs(1+a/(1-b/x**2)+c/(1-d/x**2)+e/(1-f/x**2)))
def sellmeier4(x,a,b,c,d,e,f):
    return np.sqrt(np.abs(a+b/(1-c/x**2)+e/(1-f/x**2)-d*x**2))
def sellmeier6(x,a,b,c,d,e,pz,qz):
    return np.sqrt(np.abs(a+b/(1-c/x**pz)+d/(1-e/x**qz)))
def sellmeier7(x,a,b,c,e,f):
    return np.sqrt(a+b/(x**2-c)+e/(x**2-f))
def sellmeier5(x,T,a,b,c,d,e,f,g,h,bt,ct):
    return sqrt(a+(b+bt*(T+273.15)**2)/(x**2-(c+ct*(T+273.15)**2)**2)+d*x**2+e/(x**2-f**2)+g/(x**2-h**2))
def sellmeier8(x,a,b,c):
    return a+b/x**2+c/x**4
def sellmeier9(x,b1,b2,b3,c1,c2,c3):
    return sqrt(1+b1/(1-c1/x**2)+b2/(1-c2/x**2)+b3/(1-c3/x**2))
def sellmeier10(x,a,b,c,e,f,g):
    return np.sqrt(np.abs(a+b/(x**2-c**2)+e/(x**2-f**2)-g*x**2))
def sellmeier11(x,T,a1,a2,a3,a4,a5,a6,b1,b2,b3,b4):
    f = (T-24.5)*(T+570.82)
    return sqrt( a1 + b1*f + (a2+b2*f)/(x**2-(a3+b3*f)**2) + (a4+b4*f)/(x**2-(a5)**2) - a6*x**2 )
def sellmeier12(x,a,b,c,d,e):
    return sqrt( a + b/(1-c/x**2) + d/(1-e/x**2) )
def sellmeier13(x,a,b,c,d,e,f):
    return sqrt( a + b*x**2 + c/x**2 + d/x**2 + e/x**2 + f/x**2 )
def sellmeier14(x,a,b,c,e,f,g,h):
    return sqrt(a+b/(x**2-c)+e/(x**2-f)+g/(x**2-h))
def sellmeier15(x,a,b,c):
    return sqrt(a+1/x**2+b/(x**2-c))
def sellmeier16(x,a,b,c,f,g):
    return a + b*x + c*x**2 + f*np.exp(-(x-350)/g)
def sellmeier17(x,a,b):
    return a*x**2/(x**2-b)
def sellmeier18(x,a,b,c):
    return a + b*x + c*x**2
def sellmeierkato2002z(x): ## sellmeierkato2002z(x) = sqrt(4.59423+0.06206/(x**2-0.04763)+110.80672/(x**2-86.12171))
    return sqrt(4.59423+0.06206/(x**2-0.04763)+110.80672/(x**2-86.12171))
def sellmeierkato2002y(x): ## sellmeierkato2002y(x) = sqrt(3.45018+0.04341/(x**2-0.04597)+16.98825/(x**2-39.43799))
    return sqrt(3.45018+0.04341/(x**2-0.04597)+16.98825/(x**2-39.43799))
def sellmeierkato2002x(x):
    return sqrt(3.29100+0.04140/(x**2-0.03978)+9.65522/(x**2-31.45571))
def nzmglnridge(x): #make/o/n=401 nzmglnridge; setscale/i x,250,4250,nzmglnridge; nzmglnridge = 3.7813e-05-8.2301e-08*x-1.1761e-09*x^2 // 10x10um,10um cut MgLN ridge on SiO2
    return 3.7813e-05-8.2301e-08*x-1.1761e-09*x**2
def nzmglnhalfridge(x): #make/o/n=401 nzmglnhalfridge; setscale/i x,250,4250,nzmglnhalfridge; nzmglnhalfridge = 3.805e-05-8.2578e-08*x-1.083e-09*x^2 // 10x10um,5um cut MgLN ridge on SiO2
    return 3.805e-05-8.2578e-08*x-1.083e-09*x**2
def ridgesellmeier(λ,width,depth,sell):
    ...
def ridgesellmeier(width,depth,sell): # returns fit coefficients
    ...

def index(w,sell,temp=20,warn=True): # x is wavelength in nm
    # np.warnings.filterwarnings('ignore')
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    if warn and not np.all(np.logical_or(np.isnan(w),np.logical_and(100<np.abs(w),np.abs(w)<10000))):
        # raise ValueError('Wavelength out of range, w = %g nm' % w)
        a = np.array(w).flatten()
        print('Warning: sellmeier.index wavelength out of range', a if len(a)<6 else f"{a[:2]}..{a[-2:]}" )
    x = np.abs(w)/1000.  # change x to um
    temp = np.array(temp) # list becomes np array
    n = None
    if sell=="pm1550z" or sell=="pm1550":
        n = index(x*1000,"sio2",temp,warn=warn)
    if sell=="pm1550y" or sell=="pm1550x":
        n = index(x*1000,"pm1550z",temp,warn=warn) - 0.00035*2.25/1.1674743331937738*1335.279956/2100 # sign might be wrong
    if sell=="pm850z" or sell=="pm850":
        n = index(x*1000,"sio2",temp,warn=warn)
    if sell=="pm850y" or sell=="pm850x":
        n = index(x*1000,"pm850z",temp,warn=warn) - 0.00035*2.25/1.1674743331937738 # sign might be wrong
    if sell=="michelsonz":
        n = 2
    if sell=="michelsony" or sell=="michelsonx":
        n = 1
    if sell=="air" or sell=="airz" or sell=="airy" or sell=="airx":
        n = 1
    if sell=="lnwg" or sell=="lnwgz":
        n = index(x*1000,"ln",temp,warn=warn) + interp(x*1000,dnzlnx,dnzlny)
    if sell=="lnwgy" or sell=="lnwgx":
        assert 0, 'no type II lnwg'
    if sell=="mglnwg" or sell=="mglnwgz":
        n = index(x*1000,"mgln",temp,warn=warn) + interp(x*1000,dnzlnx,dnzlny)
    if sell=="mglnwgy" or sell=="mglnwgx":
        assert 0, 'no type II mglnwg'
    if sell=="mglnridgewg" or sell=="mglnridgewgz":
        n = index(x*1000,"mgln",temp,warn=warn) + nzmglnridge(x*1000)
    if sell=="mglnhalfridgewg" or sell=="mglnhalfridgewgz":
        n = index(x*1000,"mgln",temp,warn=warn) + nzmglnhalfridge(x*1000)
    if sell=="lnridgewg" or sell=="lnridgewgz":
        n = index(x*1000,"ln",temp,warn=warn) + nzmglnridge(x*1000)
    if sell=="lnhalfridgewg" or sell=="lnhalfridgewgz":
        n = index(x*1000,"ln",temp,warn=warn) + nzmglnhalfridge(x*1000)
    if sell=="mglnridgewgy":
        n = index(x*1000,"mglny",temp,warn=warn) + nzmglnridge(x*1000)
    if sell=="mglnhalfridgewgy":
        n = index(x*1000,"mglny",temp,warn=warn) + nzmglnhalfridge(x*1000)
    if sell=="lnridgewgy":
        n = index(x*1000,"lny",temp,warn=warn) + nzmglnridge(x*1000)
    if sell=="lnhalfridgewgy":
        n = index(x*1000,"lny",temp,warn=warn) + nzmglnhalfridge(x*1000)
    if "ridge" in sell and n is None:
        assert 'wg' in sell, f'"{sell}" is wrong format (e.g. mglnhalfridge3x7wgz)'
        import ridgesellmeier
        func = ridgesellmeier.indexfunc(sell)
        bulk = sell.split('ridge')[0].replace('half','')
        n = index(x*1000,bulk,temp,warn=warn) + func(x*1000)
    if sell=="gaymglnwg" or sell=="gaymglnwgz":
        n = index(x*1000,"gaymgln",temp,warn=warn) + interp(x*1000,dnzlnx,dnzlny)
    if sell=="ktpwg" or sell=="ktpwgz" or sell=="katwg" or sell=="katwgz":
        n = index(x*1000,"ktp",temp,warn=warn) + interp(x*1000,dnzx,dnz)
    if sell=="ktpwgy" or sell=="katwgy":
        n = index(x*1000,"ktpy",temp,warn=warn) + interp(x*1000,dnyx,dny)
    if sell=="ktpwgsurfacez":
        n = index(x*1000,"ktp",temp,warn=warn) + nsurffunc(x*1000,*dnsz_coef)
    if sell=="ktpwgsurfacey":
        n = index(x*1000,"ktpy",temp,warn=warn) + nsurffunc(x*1000,*dnsy_coef)
    if sell=="ktpwgx" or sell=="katwgx":
        n = index(x*1000,"ktpx",temp,warn=warn) + interp(x*1000,dnyx,dny)
    if sell=="ktphalfwg" or sell=="ktphalfwgz":
        n = index(x*1000,"ktp",temp,warn=warn) + 0.5*interp(x*1000,dnzx,dnz)
    if sell=="ktphalfwgy":
        n = index(x*1000,"ktpy",temp,warn=warn) + 0.5*interp(x*1000,dnyx,dny)
    if sell=="ktphalfwgx":
        n = index(x*1000,"ktpx",temp,warn=warn) + 0.5*interp(x*1000,dnyx,dny)
    if sell=="ktpcalwg" or sell=="ktpcalwgz":
        n = index(x*1000,"ktpz",temp,warn=warn) + sellmeier16(x*1000,26.7947e-3,-1.09737e-5,2.29268e-9,2.24565e-2,44.62477)
        ## Callahan,Safak,Battle,Roberts,Kartner2014, see also dnsz above
        # def sellmeier16(x,a,b,c,f,g): return a + b*x + c*x**2 + f*np.exp(-(x-350)/g)
    if sell=="ktpcalwgy":
        n = index(x*1000,"ktpy",temp,warn=warn) + sellmeier16(x*1000,29.0816e-3,-6.5850e-6,2.13894e-9,9.60547e-3,39.20047)
        ## Callahan,Safak,Battle,Roberts,Kartner2014, see also dnsy above
        # def sellmeier16(x,a,b,c,f,g): return a + b*x + c*x**2 + f*np.exp(-(x-350)/g)
    if sell=="eman" or sell=="emanz":
        n=sellmeier7(x, 4.59423, 0.06206, 0.04763, 110.80672, 86.12171)   ##Kato02
        n = n + (  9.9587e-6 + 9.9228e-6/(x) - 8.9603e-6/(x)**2 + 4.1010e-6/(x)**3 )*(temp-25)            ##Emanueli03
        n = n + ( -1.1882e-8 + 10.459e-8/(x) - 9.8136e-8/(x)**2 + 3.1481e-8/(x)**3 )*(temp-25)**2      ##Emanueli03
    if sell=="emanx":
        n=sellmeier7(x, 3.29100, 0.04140, 0.03978, 9.35522, 31.45571)  ##Kato02
        n = n + (temp-20) * ( 0.1717e-5/(x)**3 -0.5353e-5/(x)**2 +0.8416e-5/(x) +0.1627e-5 )  ## no nx for Emanueli03 
    if sell=="emany":
        n=sellmeier7(x, 3.45018, 0.04341, 0.04597, 16.98825, 39.43799)  ##Kato02
        n = n + (  6.2897e-6 + 6.3061e-6/(x) - 6.0629e-6/(x)**2 + 2.6486e-6/(x)**3 )*(temp-25)            ##Emanueli03
        n = n + ( -.14445e-8 + 2.2244e-8/(x) - 3.5770e-8/(x)**2 + 1.3470e-8/(x)**3 )*(temp-25)**2      ##Emanueli03
    if sell=="ktp" or sell=="ktpz" or sell=="kat" or sell=="katz":
        n=sellmeier7(x, 4.59423, 0.06206, 0.04763, 110.80672, 86.12171)  ##Kato&Takaoka02,  .44<x<3.55
        n = n + (temp-20) * ( 0.9221e-5/(x)**3 -2.9220e-5/(x)**2 +3.6677e-5/(x) -0.1897e-5 )  ##temp dep (Kato z, .53<x<1.57)
    if sell=="ktpx" or sell=="katx":
        n=sellmeier7(x, 3.29100, 0.04140, 0.03978, 9.35522, 31.45571)  ##Kato&Takaoka02
        n = n + (temp-20) * ( 0.1717e-5/(x)**3 -0.5353e-5/(x)**2 +0.8416e-5/(x) +0.1627e-5 )  ##temp dep (Kato x, .43<x<1.58)
    if sell=="ktpy" or sell=="katy":
        n=sellmeier7(x, 3.45018, 0.04341, 0.04597, 16.98825, 39.43799)  ##Kato&Takaoka02
        n = n + (temp-20) * ( 0.1997e-5/(x)**3 -0.4063e-5/(x)**2 +0.5154e-5/(x) +0.5425e-5 )  ##temp dep (Kato y, .43<x<1.58)
    if sell=="fan" or sell=="fanz":
        n=sellmeier4(x, 2.25411, 1.06543, 0.05486, 0.02140, 0, 0)  ##Fan (<1um)
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann)
        ##n = ( (x<.4) ? index(400,temp,sell) - ( (index(410,temp,sell)-index(400,temp,sell))/(.01)*(.4-x) ) : n )
    if sell=="fanx":
        n=sellmeier4(x, 2.16747, 0.83733, 0.04611, 0.01713, 0, 0)  ##Fan x
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="fany":
        n=sellmeier4(x, 2.19229, 0.83547, 0.04970, 0.01621, 0, 0)  ##Fan y
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="fanwg" or sell=="fanwgz":
        n = index(x*1000,"fan",temp,warn=warn) + interp(x*1000,dnzx,dnz)
    if sell=="fanwgy":
        n = index(x*1000,"fany",temp,warn=warn) + interp(x*1000,dnyx,dny)
    if sell=="fanwgx":
        n = index(x*1000,"fanx",temp,warn=warn) + interp(x*1000,dnyx,dny)
    if sell=="frad" or sell=="fradz":
        n=sellmeier4(x, 2.12725, 1.18431, 0.0514852, 0.00968956, 0.6603, 100.00507)  ##KTP Fradkin (>1um)
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann)
        ##n = ( (x<.4) ? index(400,temp,sell) - ( (index(410,temp,sell)-index(400,temp,sell))/(.01)*(.4-x) ) : n )
    if sell=="van" or sell=="vanz":
        n=sellmeier4(x,  2.3136, 1.00012, 0.23831**2, 0.01679, 0, 0)  ##HT-KTP  Vanherzeele 88 (same as Cheng et al 94)
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann)
    if sell=="vanx":
        n=sellmeier4(x,  2.1146, 0.89188, 0.20861**2, 0.01320, 0, 0)  ##HT-KTP  Vanherzeele 88 (same as Cheng et al 94)
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="vany":
        n=sellmeier4(x,  2.1518, 0.87862, 0.21801**2, 0.01327, 0, 0)  ##HT-KTP  Vanherzeele 88 (same as Cheng et al 94)
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="vanwg" or sell=="vanwgz":
        n = index(x*1000,"van",temp,warn=warn) + interp(x*1000,dnzx,dnz)
    if sell=="vanwgy":
        n = index(x*1000,"vany",temp,warn=warn) + interp(x*1000,dnyx,dny)
    if sell=="vanwgx":
        n = index(x*1000,"vanx",temp,warn=warn) + interp(x*1000,dnyx,dny)
    if sell=="criktp":
        n=sellmeier2(x, 3.313400, 0.056940, 0.059410, 0.016713) ##KTP Cristal
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann)
    if sell=="crirtp":
        n=sellmeier4(x, 2.27723, 1.1103, 0.23454**2, 0.01995, 0, 0)  ##RTP Cristal, same as Cheng94
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="chex":
        n=sellmeier4(x, 2.1146, 0.89188, 0.20861**2, 0.0132, 0, 0)  ##KTP Cheng94 (same as HT-KTP  Vanherzeele 88)
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="chey":
        n=sellmeier4(x, 2.1518, 0.87862, 0.21801**2, 0.01327, 0, 0)  ##KTP Cheng94 (same as HT-KTP  Vanherzeele 88)
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="che" or sell=="chez":
        n=sellmeier4(x, 2.3136, 1.00012, 0.23831**2, 0.01679, 0, 0)  ##KTP Cheng94 (same as HT-KTP  Vanherzeele 88)
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="rtpx":
        n=sellmeier4(x, 2.15559, 0.93307, 0.20994**2, 0.01452, 0, 0)  ##RTP Cheng94
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="rtpy":
        n=sellmeier4(x, 2.38494, 0.73603, 0.23891**2, 0.01583, 0, 0)  ##RTP Cheng94
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="rtp" or sell=="rtpz":
        n=sellmeier4(x, 2.27723, 1.1103, 0.23454**2, 0.01995, 0, 0)  ##RTP Cheng94
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="rtpmikx":
        n=sellmeier7(x, 4.65575, 0.04068, 0.04750, 204.2586, 130.7684)       ##RTP Mikami09
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="rtpmiky":
        n=sellmeier7(x, 4.76892, 0.04490, 0.05130, 221.3309, 134.2832)       ##RTP Mikami09
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="rtpmik" or sell=="rtpmikz":
        n=sellmeier7(x, 7.97109, 0.06079, 0.05968, 1234.6913, 269.8094)      ##RTP Mikami09
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="rtax":
        n=sellmeier4(x, 2.22681, 0.99616, 0.21423**2, 0.01369, 0, 0)  ##RTA Cheng94
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="rtay":
        n=sellmeier4(x, 1.97756, 1.25726, 0.20448**2, 0.00865, 0, 0)  ##RTA Cheng94
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="rta" or sell=="rtaz":
        n=sellmeier4(x, 2.28779, 1.20629, 0.23484**2, 0.01583, 0, 0)  ##RTA Cheng94
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="ktax":
        n=sellmeier4(x, 2.11055, 1.03177, 0.21088**2, 0.01064, 0, 0)  ##KTA Cheng94
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="ktay":
        n=sellmeier4(x, 2.38888, 0.779, 0.23784**2, 0.01501, 0, 0)  ##KTA Cheng94
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="kta" or sell=="ktaz":
        n=sellmeier4(x, 2.34723, 1.10111, 0.24016**2, 0.01739, 0, 0)  ##KTA Cheng94
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="ctax":
        n=sellmeier4(x, 2.34498, 1.04863, 0.22044**2, 0.01483, 0, 0)  ##CTA Cheng94
        n = n + (temp-20) * ( 1.427e-6/(x)**3 -4.735e-6/(x)**2 +8.711e-6/(x) +0.952e-6 )  ##temp dep (Wiechmann x)
    if sell=="ctay":
        n=sellmeier4(x, 2.7444, 0.70733, 0.26033**2, 0.01526, 0, 0)  ##CTA Cheng94
        n = n + (temp-20) * ( 4.269e-6/(x)**3 -14.761e-6/(x)**2 +21.232e-6/(x) -2.113e-6 )  ##temp dep (Wiechmann y)
    if sell=="cta" or sell=="ctaz":
        n=sellmeier4(x, 2.53666, 1.106, 0.24988**2, 0.01711, 0, 0)  ##CTA Cheng94
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="dmirtp":
        n=sellmeier4(x, 2.77339, 0.63961, 0.08151, 0.02237, 0, 0)  ##RTP Dmitriev
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="fenrta":
        n=sellmeier4(x, 2.18962, 1.30103, 0.22809**2, 0.01390, 0, 0)  ##RTA Fenimore
        n = n + (temp-20) * ( -76.34e-6/(x)**3 +257.19e-6/(x)**2 -237.97e-6/(x) )  ##RTA temp dep Karlsson
    if sell=="fevrta":
        n=sellmeier6(x, 2.5902, 0.8983, 0.072667, 0.7528, 68.6252, 1.9851, 2.0099)  ##RTA Feve
        n = n + (temp-20) * ( -76.34e-6/(x)**3 +257.19e-6/(x)**2 -237.97e-6/(x) )  ##RTA temp dep Karlsson
    if sell=="fevkta":
        n=sellmeier6(x, 2.1931, 1.2382, 0.059171, 0.5088, 53.2898, 1.8920, 2.0000)  ##KTA Feve
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="crikta":
        n=sellmeier4(x, 2.14786, 1.29559, 0.22719**2, 0.01436, 0, 0)  ##KTA Cristal
        n = n + (temp-20) * ( 12.415e-6/(x)**3 -44.414e-6/(x)**2 +59.129e-6/(x) -12.101e-6 )  ##temp dep (Wiechmann KTP)
    if sell=="ktpsurfz":
        n = nsurffunc(1000*x,26.7694730652534,-10.9737456325554,2.29268132448848,0,0.022459517196379,44.624769075701)
        assert 20==temp, "ktpsurfz temperature dependence not defined"
    if sell=="ktpsurfy":
        n = nsurffunc(1000*x,29.0815525865133,-6.58500255570493,2.13893700130814,0,0.00960546909966567,39.2004747033197)
        assert 20==temp, "ktpsurfy temperature dependence not defined"
    if sell=="slt":
        n = sellmeier5(x, temp, 4.502483, 0.007294, 0.185087, -0.02357, 0.073423, 0.199595, 0.001, 7.99724, 3.483933e-8, 1.607839e-8)  ##Bruner03 SLT
    if sell=="mgslt":
        n = sellmeier5(x, temp, 4.524925, 0.007687, 0.266361, -0.022721, 0.070618, 0.19041, 1.434624, 6.90558, 3.814896e-8, 1.605983e-8)  ##Weng08 MgSLT
    if sell=="clt":
        n = sellmeier5(x, temp, 4.514261, 0.011901, 0.110744, -0.02323, 0.076144, 0.195596, 0, 0, 1.82194e-8, 1.5662e-8)  ##Bruner03 CLT
    if sell=="abeltx" or sell=="abelty":
        n = sellmeier2(x, 4.51224, 0.0847522, 0.19876**2, 0.0239046 )  ##Abedin96 LT no
        assert 20==temp, "temperature dependence of lt not defined"
    if sell=="abeltz":
        n = sellmeier2(x, 4.52999, 0.0844313, 0.20344**2, 0.0237909 )  ##Abedin96 LT ne
        assert 20==temp, "temperature dependence of lt not defined"
    if sell=="ln" or sell=="lnz" or sell=="cln" or sell=="clnz":
        n = sellmeier3(x,2.9804,0.02047,0.5981,0.0666,8.9543,416.08)  ##ZelmonSmallJundt97 LN ne
        n = n + index(1000*x,"gaymglnz",temp,warn=warn) - index(1000*x,"gaymglnz",20,warn=warn) ## using MgLN for temperature dependence
    if sell=="lnx" or sell=="lny" or sell=="clnx" or sell=="clny":
        n = sellmeier3(x,2.6734,0.01764,1.2290,0.05914,12.614,474.6)  ##ZelmonSmallJundt97 LN no
        n = n + index(1000*x,"gaymglny",temp,warn=warn) - index(1000*x,"gaymglny",20,warn=warn) ## using MgLN for temperature dependence
    if sell=="gaymgln" or sell=="gaymglnz":  ##  Gayer08 better for 1064->1571+3297, worse than Zelmon for 1064
        n = sellmeier11(x,temp, 5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2, 2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4 )  ## Gayer08 5% MgO:LN ne (d33)
    if sell=="gaymglnx" or sell=="gaymglny":
        n = sellmeier11(x,temp, 5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2, 7.941e-7, 3.134e-8, -4.641e-9, -2.188e-6 )  ## Gayer08 5% MgO:LN no (d22)
    if sell=="zelmgln" or sell=="zelmglnz" or sell=="lenmgln":
        n = sellmeier3(x,2.2454,0.01242,1.3005,0.05313,6.8972,331.33)  ##ZelmonSmallJundt97 5% MgO:LN ne
        n = n + index(1000*x,"gaymglnz",temp,warn=warn) - index(1000*x,"gaymglnz",20,warn=warn)
    if sell=="zelmglnx" or sell=="zelmglny":
        n = sellmeier3(x,2.4272,0.01478,1.4617,0.05612,9.6536,371.216)  ##ZelmonSmallJundt97 5% MgO:LN no
        n = n + index(1000*x,"gaymglny",temp,warn=warn) - index(1000*x,"gaymglny",20,warn=warn)
    if sell=="yam":
        n = sellmeier3(x,2.22428221,.00759751,1.320888873,.059146547,6.88350261,331.000137)  ##Yamaju 5% MgO:LN ne (n)
        n = n + index(1000*x,"gaymglnz",temp,warn=warn) - index(1000*x,"gaymglnz",20,warn=warn)
    if sell=="mgln" or sell=="mglnz" or sell=="mglne" or sell=="umemgln" or sell=="umemglnz":
        #n = sellmeier2(x,4.54514,0.096471,0.043763,0.021502) if x<=2.7 else sellmeier14(x,24.6746,0.0456,2.280,19166.65,953.52,1.0103,45.86)  # UmemuraMatsuda16 5% MgO:LN
        n = np.where(x<=2.7, sellmeier2(x,4.54514,0.096471,0.043763,0.021502), sellmeier14(x,24.6746,0.0456,2.280,19166.65,953.52,1.0103,45.86))
        # dn/dt = (0.4175/x**3 + -0.6643/x**2 + 0.9036/x + 3.5332 + -0.0744*x)*1e-5*(1+0.00276*20)
        n = n + (0.4175/x**3 + -0.6643/x**2 + 0.9036/x + 3.5332 + -0.0744*x)*1e-5*( (temp-20) + 0.00276/2 * (temp-20)*(temp-20) )
    if sell=="mglnx" or sell=="mglny" or sell=="mglno" or sell=="umemglnx" or sell=="umemglny":
        n = sellmeier7(x,19.5542,0.11745,0.04557,8132.545,554.57)   # UmemuraMatsuda16 5% MgO:LN
        # dn/dT = (0.4519/x**4 + -2.1143/x**3 + 4.0283/x**2 + -2.9264/x + 1.0908) * 1e-5 * (1 + 0.00216*temp)*(temp-20)
        n = n + (0.4519/x**4 + -2.1143/x**3 + 4.0283/x**2 + -2.9264/x + 1.0908)*1e-5*( (temp-20) + 0.00216/2 * (temp-20)**2 )
    ##  ne**2 = 5.35583 + 4.629 \times 10**{-7} f + {0.100473 + 3.862 \times 10**{-8} f \over \x**2 - (0.20692 - 0.89 \times 10**{-8} f)**2 }
    ##  + { 100 + 2.657 \times 10**{-5} f \over \x**2 - (11.34927 )**2 } - 1.5334 \times 10**{-2} \x**2
    ##  f = (T-24.5) \times (T+570.82) valid from 20-250 degrees Celsius for wavelengths from 0.4 to 5 um [Jundt97]
    ##  ne**2 = 5.39121 + 4.968 \times 10**{-7} f + {0.100473 + 3.862 \times 10**{-8} f \over \x**2 - (0.20692 - 0.89 \times 10**{-8} f)**2 }
    ##  + { 100 + 2.657 \times 10**{-5} f \over \x**2 - (11.34927 )**2 } - (1.544 \times 10**{-2} + 9.62119 \times 10**{-10} \x) \x**2
    ##  valid for T = 25 to 180 degrees Celsius, for wavelengths between 2.8 and 4.8 um [LH Deng et al. Optics Communications 268 (2006)]
    if sell=="lnrouwg":  ## sqrt(a+b/(x**2-c)-d*x**2)
        n = sellmeier2(x, 4.945, 0.1354, 0.2324**2, 0.0278 )  ##Roussev06thesis wg ne
        assert 20==temp, "lnrouwg temperature dependence of ln not defined"
    if sell=="lnsurfx":  ## sqrt(a+b/(x**2-c)-d*x**2)
        n = sellmeier2(x, 5.063e-3, 1.294e-3, 0.217**2, 0 )  ##Lenzini15 surface delta-n x-cut ne
        assert 20==temp, "lnsurfx temperature dependence of ln not defined"
    if sell=="lnsurfz":  ## sqrt(a+b/(x**2-c)-d*x**2)
        n = sellmeier2(x, 4.646e-3, 9.632e-4, 0.272**2, 0 )  ##Lenzini15 surface delta-n z-cut ne
        assert 20==temp, "lnsurfz temperature dependence of ln not defined"
    if sell=="lnjun":  ## sqrt(a+b/(x**2-c**2)+e/(x**2-f**2)-g*x**2)
        n = sellmeier10(x, 5.35583, 0.100473, 0.20692, 100, 11.34927, 1.5334e-2 ) ## Jundt97 LN ne
        assert 20==temp, "lnjun temperature dependence of ln not defined"
    if sell=="lnstrakey":  # Strake88 index-to-concentration ratio d_o(λ)
        n = sellmeier17(x,0.67,0.13)
        assert 20==temp, "lnstrakey temperature dependence of ln not defined"
    if sell=="lnstrakez":  # Strake88 index-to-concentration ratio d_e(λ)
        n = sellmeier17(x,0.839,0.0645)
        assert 20==temp, "lnstrakez temperature dependence of ln not defined"
    if sell=="lnfouchet0y":  # Fouchet87 surface index dispersion relation
        n = sellmeier18(x,+6.53e-2,-3.15e-2,+7.09e-3)
        assert 20==temp, "lnfouchet0y temperature dependence of ln not defined"
    if sell=="lnfouchet1y":  # Fouchet87 surface index dispersion relation
        n = sellmeier18(x,+0.478,+0.464,-0.348)
        assert 20==temp, "lnfouchet1y temperature dependence of ln not defined"
    if sell=="lnfouchet0z":  # Fouchet87 surface index dispersion relation
        n = sellmeier18(x,+0.385,-0.430,+0.171)
        assert 20==temp, "lnfouchet0z temperature dependence of ln not defined"
    if sell=="lnfouchet1z":  # Fouchet87 surface index dispersion relation
        n = sellmeier18(x,+9.13,+3.85,-2.49)
        assert 20==temp, "lnfouchet1z temperature dependence of ln not defined"

    if sell=="lnganguly0y":  # Ganguly surface index dispersion relation
        n = sellmeier18(x,+6.53e-2,-3.15e-2,+7.09e-3)
        assert 20==temp, "lnganguly0y temperature dependence of ln not defined"
    if sell=="lnganguly1y":  # Ganguly surface index dispersion relation
        n = sellmeier18(x,+0.478,+0.464,-0.348)
        assert 20==temp, "lnganguly1y temperature dependence of ln not defined"
    if sell=="lnganguly0z":  # Ganguly surface index dispersion relation
        n = sellmeier18(x,+0.385,-0.430,+0.171) # # <--- +1.171 is only difference with Fouchet87
        assert 20==temp, "lnganguly0z temperature dependence of ln not defined"
    if sell=="lnganguly1z":  # Ganguly surface index dispersion relation
        n = sellmeier18(x,+9.13,+3.85,-2.49)
        assert 20==temp, "lnganguly1z temperature dependence of ln not defined"

    if sell=="sin" or sell=="sinz" or sell=="siny" or sell=="siliconnitride":
        n = (1+3.0249/(1-(0.1353406/x)**2)+40314/(1-(1239.842/x)**2))**.5 # https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke
    if sell=="si" or sell=="siz" or sell=="siy":
        n =  sellmeier9(x,10.6684293,0.0030434748,1.54133408,0.301516485**2,1.13475115**2,1104**2)
        assert np.all(1.357<=x), 'silicon sellmeier out of range: '+str(x)
        # https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg
        # n^2-1=\frac{10.6684293?^2}{ λ^2-0.301516485^2}+\frac{0.0030434748 λ^2}{ λ^2-1.13475115^2}+\frac{1.54133408 λ^2}{ λ^2-1104^2} # 
        # def sellmeier9(x,b1,b2,b3,c1,c2,c3): return sqrt(1+b1/(1-c1/x**2)+b2/(1-c2/x**2)+b3/(1-c3/x**2))
    if sell=="asi" or sell=="asiz" or sell=="asiy":
        n = np.real(pierceamorphoussilicon(x))
    if sell=="sichan" or sell=="sichanz" or sell=="sichany": # Chandler-Horowitz and Amirtharaj 2005
        n = sellmeier15(x,11.67316,0.004482633,1.108205**2)
    if sell=="sio2" or sell=="sio2z" or sell=="sio2y" or sell=="sio2cvi":
        n = sellmeier9(x,6.96166300e-1,4.07942600e-1,8.97479400e-1,4.67914826e-3,1.35120631e-2,97.9340025)
    if sell=="sio2schott":
        n = sellmeier9(x,6.70710810E-01,4.33322857E-01,8.77379057E-01,4.49192312E-03,1.32812976E-02,9.58899878E+01)
    if sell=="quartz" or sell=="quartze" or sell=="quartzz": ##ne2=2.3849-0.01259?2+0.01079/?2+1.6518x10-4/?4-1.9474x10-6/?6+9.3648x10-8/?8
        n = sellmeier13(x, 2.3849, -0.01259, 0.01079, 1.6518e-4, -1.9474e-6, 9.3648e-8)
    if sell=="quartzo" or sell=="quartzx" or sell=="quartzy": ##no2=2.3573-0.01170?2+0.01054/?2+1.3414x10-4/?4-4.4537x10-7/?6+5.9236x10-8/?8 ##http:##www.newlightphotonics.com/v1/quartz-properties.html
        n = sellmeier13(x, 2.3573, -0.01170, 0.01054, 1.3414e-4, -4.4537e-7, 5.9236e-8)
    if sell=="calciteo" or sell=="calcitez" or sell=="calcite": ##no2=2.69705+0.0192064/(λ2-0.01820)-0.0151624λ2 ##http://www.redoptronics.com/Calcite-crystal.html # http://www.newlightphotonics.com/Birefringent-Crystals/Calcite-Crystals
        n = sellmeier2(x, 2.69705, 0.0192064, 0.01820, 0.0151624)
    if sell=="calcitee" or sell=="calcitey": ##nne2=2.18438+0.0087309/(λ2-0.01018)-0.0024411λ2
        n = sellmeier2(x, 2.18438, 0.0087309, 0.01018, 0.0024411)
    if sell=="calcite2o" or sell=="calcite2z" or sell=="calcite2": #no2=1+0.73358749+0.96464345*lambda^2/(lambda^2-.0194325203)+1.82831454*lambda^2/(lambda^2-120) # Kristina Meier 2/28/19
        n = sellmeier12(x,1+0.73358749,0.96464345,.0194325203,1.82831454,120)
    if sell=="calcite2e" or sell=="calcite2y": #ne2=1+0.35859695+0.8242783*lambda^2/(lambda^2-.0106689543)+0.14429128*lambda^2/(lambda^2-120)
        n = sellmeier12(x,1+0.35859695,0.8242783,.0106689543,0.14429128,120)
    if sell=="bk7" or sell=="bk7cvi":
        n = sellmeier9(x,1.03961212,2.31792344e-1,1.01046945,6.00069867e-3,2.00179144e-2,1.03560653e2)
    if sell=="noa":
        n = sellmeier8(x, 1.5375, 8290.45e-6, -2.11046e-4)  ##Norland NOA 61
    if sell=="yvo4" or sell=="yvo4e" or sell=="yvo4z": ## extraordinary
        n = sellmeier2(x, 4.59905, 0.110534, 0.04813, 0.012267612) ##yttrium orthovanadate wiki
    if sell=="yvo4o" or sell=="yvo4y": ## ordinary
        n = sellmeier2(x, 3.77834, 0.069736, 0.04724, 0.0108133) ##yttrium orthovanadate wiki
    if sell=="ndyvo4" or sell=="ndyvo4e" or sell=="ndyvo4z": ## extraordinary
        n = sellmeier12(x, 2.7582, 1.853, 0.056986, 3.0749, 195.06) ## Zelmon10
    if sell=="ndyvo4o" or sell=="ndyvo4y": ## ordinary
        n = sellmeier12(x, 2.3409, 1.4402, 0.04825, 1.8698, 171.27) ## Zelmon10
    if sell=="al2o3": # https://refractiveindex.info/tmp/data/main/Al2O3/Malitson-o.html
        n = (1+1.4313493/(1-(0.0726631/x)**2)+0.65054713/(1-(0.1193242/x)**2)+5.3414021/(1-(18.028251/x)**2))**.5
    x*=1000
    assert np.all(n!=None), f'{sell} index not defined'
    if warn and not np.all(np.logical_or(np.isnan(n),n>=1)) and not sell in ['ktpsurfz','ktpsurfy','lnsurfz','lnsurfx','michelsony','michelsonx','lnstrakey','lnstrakez','lnfouchet0y','lnfouchet1y','lnfouchet0z','lnfouchet1z','lnganguly0y','lnganguly1y','lnganguly0z','lnganguly1z']:
        # raise ValueError('index is less than one for '+str(w)+'~'+str(temp)+'~'+sell)
        a = np.array(w).flatten()
        print('warning, index is less than one for '+(str(a) if len(a)<6 else f"{a[:2]}..{a[-2:]}")+'~'+str(temp)+'~'+sell)
    return n

sellmeierlist = ['lnwg','lnwgz','mglnwg','mglnwgz','mglnridgewg','mglnridgewgz','mglnhalfridgewg','mglnhalfridgewgz','lnridgewg','lnridgewgz','lnhalfridgewg','lnhalfridgewgz','mglnridgewgy','mglnhalfridgewgy','lnridgewgy','lnhalfridgewgy','gaymglnwg','ktpwg','ktpwgz','ktpwgy','ktpwgx','eman','emanz','emanx','emany','ktp','ktpz','kat','katz','ktpx','katx','ktpy','katy','fan','fanz','fanx','fany','frad','fradz','van','vanz','vanx','vany','criktp','crirtp','chex','chey','che','chez','rtpx','rtpy','rtp','rtpz','rtax','rtay','rta','rtaz','ktax','ktay','kta','ktaz','ctax','ctay','cta','ctaz','dmirtp','fenrta','fevrta','fevkta','crikta','slt','mgslt','clt','abeltx','abelty','abeltz','ln','lnz','cln','clnz','lnx','lny','clnx','clny','gaymgln','gaymglnz','gaymglnx','gaymglny','mgln','mglnz','mglne','zelmgln','zelmglnz','lenmgln','mglnx','mglny','mglno','zelmglnx','zelmglny','yam','lnrouwg','lnsurfx','lnsurfz','lnjun','sio2','sio2cvi','sio2schott','quartz','quartze','quartzz','quartzo','quartzy','bk7','bk7cvi','noa','yvo4','yvo4e','yvo4z','yvo4o','yvo4y','ndyvo4','ndyvo4e','ndyvo4z','ndyvo4o','ndyvo4y']
polingperiodlist = ['lnwg','mglnwg','mglnridgewg','mglnhalfridgewg','lnridgewg','lnhalfridgewg','gaymglnwg','ktpwg','eman','ktp','kat','fan','van','kta','ln','cln','gaymgln','mgln','zelmgln']
# lnwg, mglnwg
dnzlny  = np.array([0.0649733,0.0383078,0.0297231,0.0251477,0.0222051,0.0201054,0.0185018,0.0172158,0.0161455,0.0152282,0.0144231,0.0137025,0.013047,0.0124427,0.0118793,0.011349,0.0108461,0.0103661,0.00990548,0.00946157,0.00903228,0.00861598,0.00821138,0.0078175,0.00743356,0.00705898,0.00669333,0.00633627,0.0059876,0.00564716,0.00531488,0.00499074,0.00467476,0.00436699,0.00406754])
dnzlnx = np.array([300+i*50 for i in range(len(dnzlny))])
# ktpsurf
dnsz_coef = [26.7694730652534,-10.9737456325554,2.29268132448848,0,0.022459517196379,44.624769075701]
dnsy_coef = [29.0815525865133,-6.58500255570493,2.13893700130814,0,0.00960546909966567,39.2004747033197]
#make/o/n=(4000-300+1) ni,nii,nsi,nsii; setscale/i x,300,4000,ni,nii,nsi,nsii
def nsurffunc(x,a,b,c,d,f,g): return (a*1e-3) + (b*1e-6)*x + (c*1e-9)*x**2 + (d*1e-12)*x**3 + f*np.exp(-(x-350)/g)
def dnsz(x): return nsurffunc(x,*dnsz_coef)
def dnsy(x): return nsurffunc(x,*dnsy_coef)
def pierceamorphoussilicon(x): # https://refractiveindex.info/?shelf=main&book=Si&page=Pierce
    def deal(n,aa):
        return [np.array(aa[i::n]) for i in range(n)]
    λ,n,k = deal(3,[1.033E-1,3.27E-1,7.26E-1,1.078E-1,3.63E-1,8.47E-1,1.127E-1,3.92E-1,9.46E-1,1.181E-1,4.23E-1,1.04E+0,1.240E-1,4.59E-1,1.14E+0,1.305E-1,
        4.97E-1,1.24E+0,1.378E-1,5.43E-1,1.35E+0,1.459E-1,5.97E-1,1.47E+0,1.550E-1,6.60E-1,1.60E+0,1.653E-1,7.35E-1,1.74E+0,1.771E-1,8.32E-1,1.89E+0,1.907E-1,
        9.51E-1,2.07E+0,2.066E-1,1.11E+0,2.28E+0,2.254E-1,1.35E+0,2.51E+0,2.480E-1,1.69E+0,2.76E+0,2.583E-1,1.86E+0,2.85E+0,2.695E-1,2.07E+0,2.93E+0,2.818E-1,
        2.30E+0,2.99E+0,2.952E-1,2.56E+0,3.04E+0,3.100E-1,2.87E+0,3.06E+0,3.263E-1,3.21E+0,3.00E+0,3.444E-1,3.55E+0,2.88E+0,3.543E-1,3.73E+0,2.79E+0,3.647E-1,
        3.90E+0,2.66E+0,3.875E-1,4.17E+0,2.38E+0,4.133E-1,4.38E+0,2.02E+0,4.428E-1,4.47E+0,1.64E+0,4.769E-1,4.49E+0,1.28E+0,4.960E-1,4.47E+0,1.12E+0,5.166E-1,
        4.46E+0,9.69E-1,5.636E-1,4.36E+0,6.90E-1,6.199E-1,4.23E+0,4.61E-1,6.526E-1,4.17E+0,3.63E-1,6.888E-1,4.09E+0,2.71E-1,7.293E-1,4.01E+0,1.99E-1,7.749E-1,
        3.93E+0,1.36E-1,8.266E-1,3.86E+0,8.12E-2,8.856E-1,3.77E+0,4.01E-2,9.538E-1,3.68E+0,0.00E+0,1.033E+0,3.61E+0,0.00E+0,1.127E+0,3.57E+0,0.00E+0,1.240E+0,
        3.54E+0,0.00E+0,1.378E+0,3.50E+0,0.00E+0,1.550E+0,3.48E+0,0.00E+0,1.771E+0,3.45E+0,0.00E+0,2.066E+0,3.44E+0,0.00E+0])
    return np.interp(x,λ,n+1j*k)
# ktpwg
dnz = np.array([0.081609152,0.069072403,0.059035599,0.050996944,0.044555377,0.039390355,0.035114542,0.031669326,0.028887415,0.026634645,0.024803359,0.02330943,0.022083759,0.021072598,0.020232713,0.019532176,0.01894374,0.018441586,0.018008651,0.017631108,0.017297665,0.016999999,0.016730638,0.016484203,0.016256234,0.01604297,0.015841959,0.015650559,0.015467147,0.015290286,0.015118651,0.01495166,0.01478825,0.014628,0.014470444,0.014315051,0.014161839,0.014010255,0.013860269,0.013711731,0.013564358,0.013418374,0.013273377,0.013129791,0.012988581,0.012848243,0.012709077,0.012570725,0.012433335,0.01229689,0.012161202,0.01202662,0.011892776,0.011759845,0.011627819,0.011496508,0.011366286,0.011236769,0.011108147,0.010980418,0.010853379,0.01072744,0.010602185,0.010477823,0.010354352,0.010231555,0.010109866,0.0099888491,0.0098687224,0.0097494861,0.0096309111,0.0095134629,0.0093966695,0.0092807692,0.0091657611,0.0090513984,0.0089381756,0.0088255936,0.0087139048,0.0086031053,0.0084929382,0.0083839279,0.0082755461,0.0081680566,0.0080614584,0.0079554776,0.0078506609,0.0077464581,0.0076431422,0.007540714,0.0074388892,0.0073382431,0.0072381953,0.0071390336,0.0070407572,0.0069430685,0.0068465592,0.0067506325,0.0066555906,0.0065614297,0.006467842,0.0063754357,0.0062835952,0.006192633,0.0061025475,0.0060130181,0.0059246775,0.0058368878,0.0057499669,0.005663909,0.0055783927,0.0054940647,0.0054102717,0.0053273393,0.0052452646,0.0051637143,0.0050833495,0.0050035021,0.0049245013,0.0048463438,0.0047686924,0.0046922285,0.0046162629,0.0045411317,0.0044668298,0.004393016,0.0043203863,0.0042482382,0.004176911,0.0041064019,0.0040363632,0.0039674849,0.0038990695,0.0038338122,0.0037701533,0.0037069432,0.0036448755,0.0035832468,0.0035224082,0.003462354,0.0034027249,0.0033442162,0.0032861221,0.0032287838,0.0031721962,0.0031160086,0.0030609213,0.0030062238,0.0029522618,0.0028990291,0.0028461711,0.002794368,0.0027429289,0.0026921928,0.002642154,0.0025924644,0.0025437952,0.0024954651,0.0024478089,0.0024008222,0.0023541588,0.0023084716,0.0022630973,0.0022183622,0.0021742603,0.0021304563,0.0020875931,0.0020450177,0.0020030409,0.0019616575,0.0019205471,0.0018803311,0.0018403784,0.0018009912,0.0017621646,0.0017235863,0.0016858414,0.0016483355,0.0016113619,0.0015749155,0.0015386943,0.0015032537,0.0014680292,0.0014332976,0.0013990548,0.0013650146,0.0013317011,0.0012985814,0.0012659188,0.0012337092,0.0012016811,0.0011703324,0.0011391572,0.0011083955,0.0010780434,0.0010478526,0.0010189399,0.00099018659,0.00096397731,0.00094033137,0.00091686106,0.00089394109,0.00087117893,0.00084875297,0.00082665478,0.00080468942,0.00078321906,0.00076186511,0.00074079185,0.00071999151,0.00069928344,0.00067900162,0.00065879646,0.00063882192,0.0006190706,0.00059937342,0.00058003329,0.00056073273,0.00054161431,0.00052267086,0.00050374557,0.0004851261,0.00046651106,0.00044802565,0.00042966357,0.00041128497,0.0003931555,0.00037499637,0.00035692492,0.00033893515,0.00032089581,0.00030302975,0.00028510159,0.00026722642,0.00024939855,0.00023148935,0.00021369458,0.00019580638,0.00017792349,0.0001600403,0.00014204507,0.00012412733,0.0001060858,8.8006949e-05,6.9885384e-05,5.1621882e-05,3.3382632e-05,1.4989951e-05,4.5974239e-06,2.3090847e-06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
dnzx = np.array([300+i*10 for i in range(len(dnz))])
dny = np.array([0.054086104,0.047032565,0.041556828,0.037303731,0.033998068,0.031426601,0.029305317,0.027631978,0.026306273,0.025249638,0.024400216,0.023712736,0.023149308,0.022682512,0.022290677,0.021956466,0.021667833,0.021413863,0.021187173,0.020981777,0.020792687,0.020616919,0.020451112,0.02029337,0.020142041,0.019995594,0.019853463,0.019714458,0.019578174,0.019444155,0.019311853,0.01918133,0.019052034,0.018923961,0.018796977,0.018670827,0.018545723,0.018421315,0.018297698,0.018174835,0.018052554,0.017931111,0.017810207,0.017689962,0.017570367,0.017451277,0.017332958,0.017215125,0.017097918,0.016981333,0.016865222,0.016749861,0.016634969,0.016520683,0.016407004,0.016293783,0.016181322,0.016069829,0.015958956,0.015848687,0.015738871,0.015629828,0.015521234,0.015413246,0.015305864,0.015198922,0.015092758,0.014987029,0.014881913,0.014777404,0.014673324,0.014570031,0.014467161,0.014364902,0.01426325,0.014162012,0.014061579,0.013961553,0.013862139,0.013763333,0.01366493,0.01356734,0.013470148,0.013373565,0.013277589,0.013182004,0.013087246,0.012992869,0.012899103,0.012805941,0.012713156,0.012621207,0.012529628,0.012438657,0.01234829,0.012258285,0.012169124,0.012080319,0.011992117,0.011904517,0.011817265,0.011730873,0.011644823,0.011559371,0.011474513,0.011389991,0.011306331,0.011223,0.011140263,0.011058117,0.010976291,0.010895334,0.010814691,0.010734634,0.010655164,0.010575998,0.010497699,0.010419698,0.010342282,0.010265448,0.010188905,0.010113229,0.010037838,0.0099630225,0.0098887794,0.0098148109,0.0097417114,0.0096688801,0.009596616,0.0095249154,0.0094534745,0.0093829008,0.0093125794,0.0092428178,0.009173613,0.0091046514,0.0090365503,0.0089686848,0.0089013688,0.0088345986,0.0087680547,0.0087023694,0.008636903,0.0085719759,0.0085075842,0.008443404,0.0083800787,0.0083169574,0.0082543595,0.0081922822,0.0081303986,0.0080693616,0.0080085136,0.0079481797,0.0078883581,0.0078287143,0.0077699074,0.0077112699,0.007653133,0.0075954911,0.0075380104,0.0074813603,0.007424864,0.0073688547,0.0073133288,0.0072579472,0.0072033815,0.0071489541,0.0070950021,0.0070415237,0.0069881724,0.0069356216,0.0068831919,0.0068312204,0.0067797028,0.0067282966,0.006677683,0.0066271755,0.006577109,0.006527483,0.0064779506,0.0064291959,0.0063805291,0.0063322932,0.0062844856,0.0062367558,0.006189784,0.006142884,0.0060963975,0.0060503203,0.0060043056,0.0059590354,0.005914066,0.0058708987,0.005828165,0.0057855146,0.0057436344,0.0057018315,0.0056604478,0.0056194765,0.0055785757,0.0055384305,0.0054983487,0.0054586688,0.005419387,0.0053801597,0.005341663,0.0053032152,0.0052651549,0.0052274792,0.0051898421,0.0051529203,0.0051160301,0.0050795102,0.0050433576,0.0050072288,0.0049717925,0.0049363752,0.0049013109,0.0048665991,0.0048318957,0.0047978628,0.0047638332,0.0047301408,0.0046967827,0.0046634199,0.0046307086,0.0045979866,0.0045655835,0.0045334962,0.0045013912,0.0044699125,0.0044384105,0.0044072084,0.0043763048,0.0043453691,0.0043150461,0.0042846855,0.0042546038,0.0042247982,0.0041949484,0.0041656871,0.0041363775,0.0041073249,0.0040785293,0.004049676,0.0040214029,0.0039930679,0.0039649704,0.0039371094,0.0039091795,0.0038818063,0.0038543588,0.0038271381,0.0038001416,0.0037730644,0.003746503,0.003719857,0.0036934258,0.0036672084,0.003640899,0.0036150799,0.0035891647,0.003563453,0.0035379424,0.0035123306,0.0034871867,0.0034619393,0.0034368727,0.0034119885,0.0033869937,0.0033624501,0.003337794,0.0033133088,0.0032889948,0.0032645622,0.0032405604,0.0032164382,0.0031924688,0.0031686504,0.0031447073,0.0031211805,0.0030975267,0.0030740188,0.0030506558,0.0030271614,0.003004051,0.0029808078,0.0029576966,0.0029347164,0.0029116003,0.0028888674,0.0028659969,0.0028432419,0.0028206035,0.0027978241,0.0027754097,0.002752854,0.0027304071,0.0027080697,0.0026855876,0.0026634466,0.0026411605,0.0026189731,0.0025968854,0.0025746513,0.0025965611,0.0026185918,0.0026407426,0.0026630149,0.0026854069,0.00270792,0.0027305547,0.0027533083,0.0027761839,0.0027991801,0.0028222962,0.0028455332,0.0028688919,0.0028923696,0.0029159691,0.0029396894,0.0029635294,0.0029874917,0.0030115733,0.0030357761,0.0030600985,0.0030845429,0.003109107,0.0031337931,0.0031585989,0.0031835255,0.003208572,0.0032337403,0.0032590285,0.0032844385,0.0033099684,0.0033356191,0.0033613897,0.0033872821,0.0034132942,0.0034394283,0.0034656823,0.0034920569,0.0035185514,0.0035451679,0.003571904,0.0035987622,0.0036257401,0.0036528388,0.0036800574,0.0037073977,0.0037348589,0.0037624401,0.003790142,0.0038179648,0.0038459084,0.0038739718,0.0039021571,0.0039304621,0.0039588879,0.0039874348,0.0040161023,0.0040448899,0.0040737991,0.0041028284,0.0041319784,0.0041612489,0.0041906405,0.0042201523,0.0042497856])
dnyx = np.array([300+i*10 for i in range(len(dny))])
sinchwhm = 1.39155737825151004649626429454656317830085754394531250 # half width half max of sinc function
# polingperiodktpwg
# dki = difference in k (=2pi/period) for type i SHG interaction between 5um and 6um deep, 4um wide waveguide, as function of pump wavelength (from nppredictioncomparison.pxp) # dkii = same for type ii # dkx = pump wavelength
dki = np.array([0,0,0.011621666,0.17914113,0.3193219,0.43593392,0.53333497,0.61297661,0.67873436,0.73236549,0.77772838,0.81555527,0.845182,0.86857671,0.88540995,0.89754152,0.90557665,0.90989524,0.91000509,0.90765297,0.90242589,0.89546245,0.88544494,0.87353307,0.85945207,0.84488785,0.82802522,0.80840755,0.78919542,0.76785719,0.74539155,0.72174203,0.69621438,0.66980761,0.64236921,0.61368603,0.58369756,0.55245167,0.52050859,0.48894924,0.45536336,0.4212124,0.38625079,0.34999937,0.31263638,0.27372408,0.23499447,0.19256514,0.14847715,0.10311889,0.057827551,0.011165355,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
dkii = np.array([0.77646583,0.95606303,1.0834198,1.1716065,1.2311177,1.2679832,1.2887056,1.2955557,1.2935179,1.2838825,1.2681495,1.2493047,1.2274544,1.2024751,1.1774651,1.1501468,1.1237457,1.0951746,1.0673534,1.0390772,1.0106604,0.98289025,0.95500118,0.92769909,0.89979255,0.87286603,0.84653288,0.81954122,0.79344887,0.76737982,0.7419337,0.7167325,0.69181049,0.66679204,0.64214867,0.61827189,0.59446001,0.57065469,0.54675484,0.52395862,0.50075865,0.47840074,0.45571157,0.43314555,0.41115043,0.38927171,0.36761513,0.34557718,0.32360351,0.3017396,0.28016695,0.25876504,0.23802613,0.21742013,0.19703321,0.17686136,0.15726107,0.13790983,0.11863098,0.099713162,0.081053622,0.063093767,0.044903044,0.02733542,0.010339052,0])
dkx = np.array([700,720,740,760,780,800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,1200,1220,1240,1260,1280,1300,1320,1340,1360,1380,1400,1420,1440,1460,1480,1500,1520,1540,1560,1580,1600,1620,1640,1660,1680,1700,1720,1740,1760,1780,1800,1820,1840,1860,1880,1900,1920,1940,1960,1980,2000])
# def qpmspectrum(λfwhm,λ0,Λ0,Λf,η=None):
#     ...
def polingperiodbandwidths(w1,w2=None,sell='ktpwg',Type='zzz',temp=20,w3=None,npy=None,npz=None,L=10,kind=None,
    getcurve=False,plot=False,quiet=True,debug=False,getfreq=False):
    # returns bandwidths in nm (or GHz for getfreq=True)
    w1,w2,w3 = qpmwavelengths(w1,w2,w3)
    kinds = [kind.lower()] if kind is not None else ['sfg1','sfg2','shg','dc1','dc2','dfg1','dfg2','temp']
    bwparam = {'sfg1':w1,'sfg2':w2,'shg':w1,'dc1':w1,'dc2':w2,'dfg1':w3,'dfg2':w3,'temp':temp}
    def p1(x): # w2 fixed
        return polingperiod( w1=w1+x, w2=w2, sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def p2(x): # w1 fixed
        return polingperiod( w1=w1, w2=w2+x, sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def pshg(x):
        return polingperiod( w1=w1+x, w2=w2+x, sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def pdc1(x):
        return polingperiod( w1=w1+x, w2=sign(w2)*1/(1/w3-1/np.abs(w1+x)), sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def pdc2(x):
        return polingperiod( w1=sign(w1)*1/(1/w3-1/np.abs(w2+x)), w2=w2+x, sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def pdfg1(x): # w1 fixed
        return polingperiod( w1=w1, w2=sign(w2)*1/(1/(w3+x)-1/np.abs(w1)), sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def pdfg2(x): # w2 fixed
        return polingperiod( w1=sign(w1)*1/(1/(w3+x)-1/np.abs(w2)), w2=w2, sell=sell, Type=Type, temp=temp, npy=npy, npz=npz )
    def ptemp(x):
        return polingperiod( w1=w1, w2=w2, sell=sell, Type=Type, temp=temp+x, npy=npy, npz=npz )
    ppfunctionselector = {'sfg1':p1,'sfg2':p2,'shg':pshg,'dc1':pdc1,'dc2':pdc2,'dfg1':pdfg1,'dfg2':pdfg2,'temp':ptemp}
    bw = {}
    for k in kinds:
        # dk1 = 2*pi/p1 at halfmax: sinc(dk1*L/2)**2 = 0.5  ->  sinc( 2*pi*(1/p1-1/p0) *L/2)**2 = 0.5  ->  L*pi*(1/p1-1/p0) = 1.3915575  ->  1/p1 = +1.3915575/L/pi + 1/p0
        # 1/p± = ±1.3915575/L/pi + 1/p0 # need to find dw for which 1/p0 - 1/period(w1+dw) ± 1.3915575/L/pi = 0
        pp = ppfunctionselector[k]
        p0 = pp(0)
        def f(x):
            return np.abs( 1/p0 - 1/pp(x) ) - sinchwhm/pi/(L*1000.)
        # in this function, f=0 at the halfmax points (x for which sinc**2=0.5), so the first negative root and the first positive root of f will give us the bandwidth 
        # to use brentq to find the roots of f, first we need to find a wavelength or temp for which f changes sign, first in positive direction (xp) then in negative direction (xn)
        def firstcrossing(func,xs): # returns first x for which func(x) is different sign than func(0) given list of x's
            s0 = sign(func(0))
            for x in xs:
                try:
                    if not s0==sign(func(x)): return x
                except ValueError: break
            return inf*sign(x)
        xs = 10**np.arange(-2,5,0.2) # xs = 10**np.arange(-2,5,1.) # ie. xs = [0.01,0.1,1,10,100,1000]
        xn,xp = firstcrossing(f,-xs), firstcrossing(f,xs)
        xn,xp = scipy.optimize.brentq(f,xn,0),scipy.optimize.brentq(f,0,xp)
        bw[k] = xp-xn
        if getcurve or plot or debug:
            def sinc(x):
                return np.sinc(x/np.pi)
            def g(x):
                return sinc( np.pi*L*1e3*( 1/pp(x) - 1/pp(0) ))**2
            ux = np.linspace(25*xn,25*xp,10001)
            u = g(ux)
            if not debug:
                uxx = ux + bwparam[k]
                # from wavedata import Wave
                # w = Wave(u,uxx,f'L = {L:.3g}mm\n$Δλ_{{FWHM}}$ = {bw[k]:.3g}nm')
                # w.plot(x='λ (nm)' if not 'temp'==k else 'temperature (°C)',y='relative efficiency',
                #     seed=w1*w2,save=f'{w1}+{w2} {k} bandwidth plot, L={L:.3g}mm')
                # return w
                if not quiet: print('  %s bandwidth: %.3f'%(kind,polingperiodbandwidths(w1,w2,sell,Type,temp,L=L,kind=kind,npy=npy,npz=npz)))
                if plot:
                    import matplotlib
                    import matplotlib.pyplot as plt
                    plt.plot(uxx,u)
                    plt.xlabel('λ (nm)' if not 'temp'==k else 'temperature (°C)')
                    plt.ylabel('relative efficiency')
                    plt.show()
                if getcurve:
                    return (u,uxx)
            else:
                u = -1+2*u
                def fp(x): return 1/p0 - 1/pp(x) + sinchwhm/pi/(L*1000.)
                def fn(x): return 1/p0 - 1/pp(x) - sinchwhm/pi/(L*1000.)
                up,un = 10000*fp(ux),10000*fn(ux)
                up,un = np.where(up>0,np.clip(up,-1,1),0), np.where(un<0,np.clip(un,-1,1),0)
                def step(x): return np.heaviside(x,0.5)
                u1,u2 = step(ux-xp),step(xn-ux)
                from wavedata import Wave
                Wave.plots(*[Wave(ui,ux) for ui in [u,u,up,un,u1,u2]])
    if getfreq:
        from optical import frequencybandwidth
        bw = {x:(frequencybandwidth(bw[x],bwparam[x]) if x in ['sfg1','sfg2','shg','dc1','dc2','dfg1','dfg2'] else bw[x]) for x in bw}
    bw['period'] = p0
    if kind: return bw[kinds[0]]
    return bw
for a in ['sfg1','sfg2','shg','dc1','dc2','dfg1','dfg2','temp']: # enable vectorized behavior using this syntax: polingperiodbandwidths.shg(1064,sell='ktp')
    setattr(polingperiodbandwidths, a, np.vectorize( functools.partial(polingperiodbandwidths,kind=a), cache=True) )
def pairrate(w1,w2=None,temp=20,sell='ktpwg',Type='zzz',w3=None,npy=None,npz=None,L=10,ηSHG=100,nW=False,pernm=False,units=False): # L in mm, ηSHG = SHG equivalent con. eff. in %/W/cm²
    bw1 = polingperiodbandwidths(w1,w2=w2,sell=sell,Type=Type,temp=temp,w3=w3,npy=npy,npz=npz,L=L,kind='dc1',quiet=True)
    bw2 = polingperiodbandwidths(w1,w2=w2,sell=sell,Type=Type,temp=temp,w3=w3,npy=npy,npz=npz,L=L,kind='dc2',quiet=True)
    # print(f'Δλ = {bw1}, {bw2} nm')
    rateinHzpermWpernm = (ηSHG/100) * (L/10)**2 * ((w2 if 2==pernm else w1)*1e-3)**(-2) * 299792458
    if not nW:
        if pernm:
            return (rateinHzpermWpernm,'Hz/mW/nm') if units else rateinHzpermWpernm
        return (rateinHzpermWpernm * bw1,'Hz/mW') if units else rateinHzpermWpernm * bw1
    h,c = 6.62607015e-34,299792458
    photonenergy = h*c/(w1*1e-9) # x2 for each photon in pair????
    rateinnWpermWpernm = rateinHzpermWpernm * photonenergy * 1e9 # rate in nW/mW
    if pernm:
        return (rateinnWpermWpernm,'nW/mW/nm') if units else rateinnWpermWpernm
    return (rateinnWpermWpernm * bw1,'nW/mW') if units else rateinnWpermWpernm * bw1
def amplitudemodulator(λ=810,sell='ktpwg',temp=25,L=10,E=0,vf=0.5,θ0=pi/4,φ0=0,norm=False,getphase=False):
    # L in mm, E in 1kV/mm, vf = vertical fraction, θ0 = output polarizer angle
    from sellmeier import EOindex,expansion,index
    nz,ny = index(λ,sell+'z',temp=temp),index(λ,sell+'y',temp=temp)
    dnz,dny = EOindex(λ,sell=sell+'z',temp=temp,E=E),EOindex(λ,sell=sell+'y',temp=temp,E=E)
    φz,φy,φ0z,φ0y = [2*pi*(n+dn)*L*(1+expansion(temp,sell))*1e6/λ for n,dn in [(nz,dnz),(ny,dny),(nz,0),(ny,0)]]
    φ = (φz-φ0z) - (φy-φ0y) - φ0 if norm else φz - φy - φ0
    if getphase:
        return φ
    Ez,Ey = np.sqrt(vf),np.sqrt(1-vf)
    ### formula with no phase delay:
    # Ein = Ez+1j*Ey
    # Eout = (Ez+1j*Ey) * (cos(θ0)-1j*sin(θ0)) # along axis of polarizer, Re = ||, Im = ⊥
    # Ereal = Ez*cos(θ0) + Ey*sin(θ0) # after polarizer, keep only real part
    ### formula with phase delay:
    # Ereal = Ez*cos(θ0)*cos(ωt) + Ey*sin(θ0)*cos(ωt-φ) ≡ Acos(ωt)+Bcos(ωt-φ)
    # I = [Acos(ωt)+Bcos(ωt-φ)]² time averaged
    # I = A²cos²(ωt) + B²cos²(ωt-φ) + 2ABcos(ωt)cos(ωt-φ) time averaged
    # I = A²cos²(ωt) + B²cos²(ωt-φ) + 2ABcos(ωt)(cos(ωt)*cos(φ)+sin(ωt)*sin(φ)) time averaged (cos(A-B)=cosAcosB+sinAsinB)
    # I = A²cos²(ωt) + B²cos²(ωt-φ) + 2ABcos²(ωt)cos(φ) time averaged
    # I = A² + B² + 2ABcos(φ)
    A,B = Ez*cos(θ0),Ey*sin(θ0)
    return A**2 + B**2 + 2*A*B*cos(φ)

def jsiplot(w1=1550,w2=1550,temp=20,sell='ktpwg',Type='yzy',lengthinmm=10,dw=None,num=2001,dt=0,
        schmidt=False,minschmidt=False,schmidtres=400,filter=None,filtershape=None,rcavity=0,apodized=False,indistinguishability=True,
        swapaxes=False,intensity=True,plot=True,plotschmidt=0,get='',qpmargs=None,plotargs={}): # dt = ΔtFWHM in ns
    assert not (get and swapaxes)
    from sellmeier import pulsebandwidth,sinc
    f1,f2 = filter if isinstance(filter,(list,tuple)) else (filter,filter)
    def bw(n):
        ppbw = polingperiodbandwidths(w1,w2,sell=sell,Type=Type,temp=temp,L=lengthinmm,kind=f'dc{n}')
        pulsebw = pulsebandwidth(dt,w1 if 1==n else w2) if dt else 0
        fbw = f1 if 1==n else f2
        if fbw is None: return 8*max(pulsebw,ppbw)
        return min( 8*max(pulsebw,ppbw), 4*fbw if filtershape=='gaussian' else fbw)
    dw1,dw2 = dw if isinstance(dw,(list,tuple)) else (dw,dw)
    dw1,dw2 = dw1 if dw1 is not None else bw(1), dw2 if dw2 is not None else bw(2)
    # dw = dw if dw is not None else 4*pulsebandwidth(dt,w1)
    # dw = 2*filter if filter is not None else dw
    # filter = filter if filter is not None else dw
    # dw1,dw2 = dw if isinstance(dw,(list,tuple)) else (dw,dw)
    # dw1,dw2 = 2*f1 if f1 is not None else dw1, 2*f2 if f2 is not None else dw2
    # x0,x1,y0,y1 = w1-dw1/2,w1+dw1/2,w2-dw2/2,w2+dw2/2
    x0,x1,y0,y1 = w1-max(dw1,f1 if f1 else 0)/2,w1+max(dw1,f1 if f1 else 0)/2,w2-max(dw2,f2 if f2 else 0)/2,w2+max(dw2,f2 if f2 else 0)/2
    def period(x,y):
        return polingperiod(w1=x,w2=y,sell=sell,Type=Type,temp=temp,qpmargs=qpmargs)
    p0 = period(w1,w2)
    def tophat(dx,width):
        return (-width/2 < dx) * (dx < width/2)
    def gaussian(dx,fwhm):
        return np.exp(np.log(0.5)*dx**2/(fwhm/2.)**2)
    filtershape = gaussian if filtershape=='gaussian' else tophat
    def filterfunc(x,y):
        fx = 1 if f1 is None else filtershape(x-w1,f1)
        fy = 1 if f2 is None else filtershape(y-w2,f2)
        return fx*fy # 1 if filter is None else 1 * filtershape(x-w1,f1) * filtershape(y-w2,f2)
    def phasematching(x,y):
        if apodized:
            if 2==apodized:
                return sinc( 2*pi * (1/period(x,y)-1/p0) * lengthinmm*1e3/2 )**2
            return exp(-0.193*( 2*pi * (1/period(x,y)-1/p0) * lengthinmm*1e3/2 )**2)  # sinc(ΔkL/2) ≈ exp(-0.193(ΔkL/2)²) smith,mahou,..walmsley2009 p7 # gaussian
        return sinc( 2*pi * (1/period(x,y)-1/p0) * lengthinmm*1e3/2 )
    def energy(x,y): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/π = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # print('ΔfFWHM (GHz)=',ΔfFWHM*1e-9) # I = exp(-4*log(2)*(f-f0)**2/ΔfFWHM**2)
        return exp(-2*log(2)*((1/abs(w1)-1/abs(x)+1/abs(w2)-1/abs(y))*1e9*299792458/ΔfFWHM)**2)
    def cavity(x,y):
        if not rcavity:
            return 1
        φx = index(x,sell+Type[0])*2*pi*2*lengthinmm*1e6*(1/x-1/w1)
        φy = index(y,sell+Type[1])*2*pi*2*lengthinmm*1e6*(1/y-1/w2)
        # return np.abs((1-rcavity)**2/(1-rcavity*exp(1j*φx))/(1-rcavity*exp(1j*φy)))
        return np.abs((1-rcavity)**2/(1-rcavity*exp(1j*φx))/(1-rcavity*exp(1j*φy)))
    x,y = np.linspace(x0,x1,num),np.linspace(y0,y1,num)
    yy,xx = np.meshgrid(y,x) # yy,xx = np.mgrid[y0:y1:(num*1j), x0:x1:(num*1j)] # x,y are reversed!
    from wavedata import Wave2D
    zz = phasematching(xx,yy) * energy(xx,yy) * filterfunc(xx,yy) * cavity(xx,yy) if dt else phasematching(xx,yy) * cavity(xx,yy)
    w = Wave2D(zz.T,xs=y,ys=x) if swapaxes else Wave2D(zz,xs=x,ys=y)
    if get=='energy': return Wave2D(energy(xx,yy).T,xs=y,ys=x)
    if get=='phasematching': return Wave2D(phasematching(xx,yy).T,xs=y,ys=x)
    if minschmidt:
        assert not get and not plotschmidt and not rcavity
        def f(a):
            w,dt = jsiplot(w1=w1,w2=w2,temp=temp,sell=sell,Type=Type,lengthinmm=lengthinmm,dw=dw,num=num,dt=a[0],schmidt=False,minschmidt=False,
                filter=filter,filtershape=filtershape,schmidtres=schmidtres,apodized=apodized,swapaxes=swapaxes,intensity=0,plot=0,qpmargs=qpmargs)
            return schmidtnumber(schmidtdecomposition(w,schmidtres))
        result = scipy.optimize.minimize(f, [dt if dt else 0.001], method='Nelder-Mead', options={'disp':True},tol=0.001) # remove method, more robust?
        dt = result.x[0]
        return jsiplot(w1=w1,w2=w2,temp=temp,sell=sell,Type=Type,lengthinmm=lengthinmm,dw=dw,num=num,dt=dt,schmidt=True,minschmidt=False,
            filter=filter,schmidtres=schmidtres,apodized=apodized,swapaxes=swapaxes,intensity=intensity,plot=plot,qpmargs=qpmargs,plotargs=plotargs)
    xlabel,ylabel = (f"λ{'s' if Type[0]==Type[1] else Type[0]} (nm)",f"λ{'i' if Type[0]==Type[1] else Type[1]} (nm)")
    if swapaxes: xlabel,ylabel = ylabel,xlabel
    xlim,ylim = (w1-dw1/2,w1+dw1/2),(w2-dw2/2,w2+dw2/2)
    if swapaxes: xlim,ylim = ylim,xlim
    def diffmax(w):
        return max([u.diff().abs().max() for u in w.xwaves()] + [u.diff().abs().max() for u in w.ywaves()])
    def perimetermax(w):
       return w.abs().perimeter().max()/w.abs().max()
    if diffmax(w)>0.1:
        print(f'warning, jsiplot diff amplitude greater than 10% max, use finer grid: {100*diffmax(w):g}%')
    if perimetermax(w)>0.01:
        print(f'warning, jsiplot edge amplitude greater than 1% max, use expanded grid: {100*perimetermax(w):g}%')
    name = f"{'jsi' if intensity else 'jsa'} {Type} {'backward'+str(-w1) if w1<0 else str(w1)}+{'backward'+str(-w2) if w2<0 else str(w2)} {sell} {lengthinmm}mm {dt*1000:g}pspump"
    name = name+f" {rcavity:g}R {dw}nm" if rcavity else name
    name = name if filter is None else name+f', {f1,f2}nm filters'
    name = name+" apodized" if apodized else name
    if schmidt or minschmidt:
        modes = schmidtdecomposition(w,schmidtres)
        K = np.abs(schmidtnumber(modes))
        print(f'τ = {dt}, K = {K}, purity = {100/K}%, {np.abs(sum([mode[0]**2 for mode in modes]))}')
        if plotschmidt:
            wws = schmidtmodes(modes,w.xs,w.ys)
            for i,ww in enumerate(wws[:plotschmidt]):
                ww.plot(x=xlabel,y=ylabel,xlim=xlim,ylim=ylim,legendtext=f'Schmidt mode {i}\nSchmidt coefficient {modes[i][0]**2:.3g}',
                    save=name+f' schmidt mode {i}',colormesh=1,show=0)
            sum(wws[:plotschmidt]).plot(x=xlabel,y=ylabel,xlim=xlim,ylim=ylim,save=name+f' sum of {plotschmidt} schmidt modes',colormesh=1,show=0)
        legendtext = f"L = {lengthinmm}mm\nΛ = {abs(p0):.1f}µm\npulse FWHM = {1000*dt:g}ps\nK = {K:.3f}\npurity = {100/K:.1f}%"
        if w1==w2 and indistinguishability:
            legendtext += f"\nI = {100*w.indistinguishability():.1f}%"
        plotargs={'legendtext':legendtext,**plotargs}
    if get=='jsa': return w
    if get=='jsi': return w**2
    if get=='purity': return 1/K
    if get=='K': return K
    if intensity: w = w**2
    if plot: 
        w.plot(x=xlabel,y=ylabel,xlim=xlim,ylim=ylim,save=name,colormesh=1,**plotargs)
    if schmidt or minschmidt:
        w.K,w.purity,w.modes = K,1/K,modes
    return w,dt
# def apodizedjsiplot(w1=1550,w2=1550,temp=20,sell='ktpwg',Type='yzy',lengthinmm=10,dw=1,num=2001,dt=0,schmidt=False,minschmidt=False,schmidtres=400,
#     filter=None,swapaxes=False,intensity=True,plot=True,plotschmidt=0,qpmargs=None,plotargs={}): # dt = ΔtFWHM in ns
#     # modified for apodization
#     filter = dw if filter is None else filter
#     f1,f2 = filter if isinstance(filter,(list,tuple)) else (filter,filter)
#     dw1,dw2 = dw if isinstance(dw,(list,tuple)) else (dw,dw)
#     x0,x1,y0,y1 = w1-max(dw1,f1)/2,w1+max(dw1,f1)/2,w2-max(dw2,f2)/2,w2+max(dw2,f2)/2
#     def period(x,y):
#         return polingperiod(w1=x,w2=y,temp=temp,sell=sell,Type=Type,qpmargs=qpmargs)
#     p0 = period(w1,w2)
#     def filterfunc(x,y):
#         def tophat(x,x0,width): return (-width/2 < x-x0) * (x-x0 < width/2)
#         return 1 if filter is None else 1 * tophat(x,w1,f1) * tophat(y,w2,f2)
#     def jsa(x,y):
#         # return sinc( 2*pi * (1/period(x,y)-1/p0) * lengthinmm*1e3/2      *0.5)**2
#         # gaussian # 
#         return exp(-0.193*( 2*pi * (1/period(x,y)-1/p0) * lengthinmm*1e3/2 )**2)  # sinc(ΔkL/2) ≈ exp(-0.193(ΔkL/2)²) smith,mahou,..walmsley2009 p7
#     def energy(x,y): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/pi = 0.4413 (Seigman p334)
#         ΔfFWHM = 2*log(2)/pi/(dt*1e-9) # print('ΔfFWHM (GHz)=',ΔfFWHM*1e-9,'at FWHM 0.5=',exp(-4*log(2)*(0.5)**2/1**2),'Λ=',p0)
#         return exp(-2*log(2)*((1/abs(w1)-1/abs(x)+1/abs(w2)-1/abs(y))*1e9*299792458/ΔfFWHM)**2) # exp(-4*log(2)*(f-f0)**2/ΔfFWHM**2)
#     x,y = np.linspace(x0,x1,num),np.linspace(y0,y1,num)
#     yy,xx = np.meshgrid(y,x) # yy,xx = np.mgrid[y0:y1:(num*1j), x0:x1:(num*1j)] # x,y are reversed!
#     zz = jsa(xx,yy) * energy(xx,yy) * filterfunc(xx,yy) if dt else jsa(xx,yy)
#     from wavedata import Wave2D
#     w = Wave2D(zz.T,xs=y,ys=x) if swapaxes else Wave2D(zz,xs=x,ys=y)
#     if minschmidt:
#         def f(a):
#             w = apodizedjsiplot(w1=w1,w2=w2,temp=temp,sell=sell,Type=Type,lengthinmm=lengthinmm,dw=dw,num=num,dt=a[0],schmidt=False,minschmidt=False,schmidtres=schmidtres,
#                 filter=filter,swapaxes=swapaxes,intensity=0,plot=0,qpmargs=qpmargs)
#             return schmidtnumber(schmidtdecomposition(w,schmidtres))
#         result = scipy.optimize.minimize(f, [dt if dt else 0.001], method='Nelder-Mead', options={'disp':True},tol=0.001) # remove method, more robust?
#         dt = result.x[0]
#         return apodizedjsiplot(w1=w1,w2=w2,temp=temp,sell=sell,Type=Type,lengthinmm=lengthinmm,dw=dw,num=num,dt=dt,schmidt=True,minschmidt=False,schmidtres=schmidtres,
#             filter=filter,swapaxes=swapaxes,intensity=intensity,plot=plot,qpmargs=qpmargs,plotargs=plotargs)
#     xlabel,ylabel = (f"λ{'s' if Type[0]==Type[1] else Type[0]} (nm)",f"λ{'i' if Type[0]==Type[1] else Type[1]} (nm)")
#     if swapaxes: xlabel,ylabel = ylabel,xlabel
#     xlim,ylim = (w1-dw1/2,w1+dw1/2),(w2-dw2/2,w2+dw2/2)
#     if swapaxes: xlim,ylim = ylim,xlim
#     name = f"apodized {'jsi' if intensity else 'jsa'} {Type} {'backward'+str(-w1) if w1<0 else str(w1)}+{'backward'+str(-w2) if w2<0 else str(w2)} {sell} {lengthinmm}mm {dt*1000:.1f}pspump"
#     name = name if filter is None else name+f', {f1,f2}nm filters'
#     if schmidt or minschmidt:
#         w.modes = schmidtdecomposition(w,schmidtres)
#         w.K = schmidtnumber(w.modes)
#         print(f'τ = {dt}, K = {w.K}, purity = {100/w.K}%, {sum([mode[0]**2 for mode in w.modes])}')
#         if plotschmidt:
#             wws = schmidtmodes(w.modes,w.xs,w.ys)
#             for i,ww in enumerate(wws[:plotschmidt]):
#                 ww.plot(x=xlabel,y=ylabel,xlim=xlim,ylim=ylim,legendtext=f'Schmidt mode {i}\nSchmidt coefficient {w.modes[i][0]**2:.3g}',save=name+f' schmidt mode {i}',colormesh=1,show=0)
#             sum(wws[:plotschmidt]).plot(x=xlabel,y=ylabel,xlim=xlim,ylim=ylim,save=name+f' sum of {plotschmidt} schmidt modes',colormesh=1,show=0)
#         legendtext = f"L = {lengthinmm}mm\nΛ = {abs(p0):.1f}µm\npulse FWHM = {1000*dt:g}ps\nK = {w.K:.3f}\npurity = {100/w.K:.1f}%"
#         plotargs={'legendtext':legendtext,**plotargs}
#     if intensity: w = w**2
#     if plot: w.plot(x=xlabel,y=ylabel,xlim=xlim,ylim=ylim,save=name,colormesh=1,**plotargs)
#     return w,dt
def schmidtmodes(modes,xs,ys,freqspace=True):
    from wavedata import Wave2D
    sxys = [(fx(1/xs),fy(1/ys)) if freqspace else (fx(xs),fy(ys)) for m,fx,fy in modes]
    return [ m*Wave2D( sx[:,np.newaxis]*sy, xs, ys ) for m,(sx,sy) in zip([mode[0] for mode in modes],sxys) ]
def schmidtnumber(modes):
    modes = map(lambda m:m[0],modes) if hasattr(modes[0],'__len__') else modes
    return 1/sum([np.abs(m)**4 for m in modes])
def schmidtdecomposition(ww,res=100,keep=None,freqspace=True):
    import pyqentangle # don't use pyqentangle==3.1.7 (3.1.0 is ok)
    f,x0,x1,y0,y1 = ww, ww.xs[0], ww.xs[-1], ww.ys[0], ww.ys[-1]
    if freqspace: f,x0,x1,y0,y1 = lambda x,y:ww(1/x,1/y), 1/ww.xs[-1], 1/ww.xs[0], 1/ww.ys[-1], 1/ww.ys[0]
    modes = pyqentangle.continuous_schmidt_decomposition(f,x0,x1,y0,y1, nb_x1=res, nb_x2=res, keep=keep) # basically a wrapper around numpy.linalg.svd, may be easier to use that directly?
    return modes

def jsicompare(w1=-1550,w2=810,temp=20,sell='ktpwg',Type='yzy',lengthinmm=10,dw=0.5,num=201,dt=0.02,qpmargs=None): # dt = ?tFWHM in ns
    Δfs = polingperiodbandwidths(w1,w2,sell=sell,Type=Type,temp=temp,L=lengthinmm,getfreq=True) # ; print(Δfs['dc1'],Δfs['dc2']) # print('df',Δfs['dc1'],'dt',2*log(2)/pi/(Δfs['dc1']))
    print('dt',dt,'Δt',2*log(2)/(Δfs['dc1']))
    x0,x1,y0,y1 = w1-dw,w1+dw,w2-dw,w2+dw
    def period(x,y): return polingperiod(w1=x,w2=y,sell=sell,Type=Type,temp=temp,qpmargs=qpmargs)
    p0,p00 = period(w1,w2),period(w2,w1) # print(p0,p00)
    def jsi(x,y,p): return sinc( 2*pi * (1/period(x,y)-1/p) * lengthinmm*1e3/2 )**2
    def energy(x,y): # time-bandwidth product for intensity ΔtFWHM*ΔfFWHM = 2ln2/pi = 0.4413 (Seigman p334)
        ΔfFWHM = 2*log(2)/pi/(dt*1e-9)
        return exp(-4*log(2)*((1/w1-1/x+1/w2-1/y)*1e9*299792458/ΔfFWHM)**2) # exp(-4*log(2)*(f-f0)**2/ΔfFWHM**2)
    x,y = np.linspace(x0,x1,num),np.linspace(y0,y1,num)
    xx,yy = np.meshgrid(x,y)
    xx00,yy00 = np.meshgrid(y,x)
    zz = jsi(xx,yy,p0) * energy(xx,yy)
    zz00 = jsi(xx00,yy00,p00) * energy(xx00,yy00)
    def plotz(z):
        import matplotlib.pyplot as plt
        plt.figure()
        color_map = plt.cm.Spectral_r
        plt.imshow(z.T,extent=[y0,y1,x0,x1],origin='lower',interpolation='bilinear',cmap=color_map,vmin=0,vmax=1)
        plt.xlabel(f'λ (nm)')
        plt.ylabel(f'λ (nm)')
        plt.ticklabel_format(style='plain',useOffset=False)
        plt.savefig('out')
        plt.show()
    plotz(zz);plotz(zz00);plotz(abs(zz-zz00))
    print(.005**2*zz.sum(),.005**2*zz00.sum(),.005**2*abs(zz-zz00).sum())
    return zz
def qpmplot(Λs=None,wp=[405,532,780],temp=20,sell='ktpwg',Type='yzy',npy=None,npz=None,
    w2onxaxis=1,x0=600,x1=2000,dx=None,prop=(1,1),qpmargs=None,**plotargs):
    from wavedata import wrange
    def pperiod(x,y):
        if qpmargs is not None:
            @np.vectorize
            def qpmΛ(x,y):
                from modes import qpm
                return qpm(λ1=x,λ2=y,**qpmargs).Λ
            return qpmΛ(x,y) if not w2onxaxis else qpmΛ(y,x)
        return (polingperiod(w1=x,w2=y,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz) if not w2onxaxis else 
                polingperiod(w1=y,w2=x,sell=sell,Type=Type,temp=temp,npy=npy,npz=npz))
    x,y = (np.linspace(x0,x1,(x1-x0)//5+1),np.linspace(x0,x1,(x1-x0)//5+1)) if dx is None else (wrange(x0,x1,dx),wrange(x0,x1,dx))
    x,y = x*prop[0],y*prop[1]
    yy,xx = np.meshgrid(y,x)
    zz = 1./pperiod(xx,yy)
    from wavedata import Wave2D,Wave
    sig,idl = f"λ{'s' if Type[0]==Type[1] else Type[0]}",f"λ{'i' if Type[0]==Type[1] else Type[1]}"
    xlabel,ylabel = (f"{sig} (nm)",f"{idl} (nm)")
    if w2onxaxis: xlabel,ylabel = ylabel,xlabel
    colors = {405:(36864,14592,58880),488:(0,26112,39168),532:(0,39168,0),560:(0,39168,0),600:(39168,26112,0),633:(52224,0,0),650:(52224,14592,14592),775:(65280,26112,26112),780:(65280,26112,26112),800:(65280,26112,26112),976:(65280,32768,32768),1064:(65280,40832,40832),1310:(65280,52224,52224),1550:(65280,58752,58752)}
    def rgbtostr(ns): return '#'+''.join([f'{n:04x}'[:2] for n in ns])
    def pumpw2(w1,wp):
        w2 = 1/np.where(w1>wp,1/wp-1/w1,nan) # w2 = np.where(w1>wp,1/(1/wp-1/w1),nan) # 'RuntimeWarning: divide by zero encountered in true_divide'  why!?
        w2 = np.where(w2>x0,w2,nan)
        return np.where(w2<x1,w2,nan)
    pumpwaves,pumpcolors = [np.sign(y[0])*Wave(pumpw2(x,float(w)),x,name=f'{w}nm'+(n==0)*' SFG') for n,w in enumerate(wp)],[rgbtostr(colors[w]) for w in wp]
    colorlines = [{'xdata':[x0*prop[0],x1*prop[0]], 'ydata':[x0*prop[1],x1*prop[1]], 'color':'gray','linestyle':(0, (1, 1))}] # + plotargs.pop('lines',[])
    # pwaves = [w1vsw2(Λ=p,temp=temp,sell=sell,Type=Type,npy=npy,npz=npz,w2onxaxis=w2onxaxis,x0=x0,x1=x1).rename(f'{abs(p):.1f}µm')
    #      for p in Λs] if Λs is not None else [] # on w1vsw2 error, check that x0,x1 range is large enough
    def contourwave(invΛ,zz):
        from skimage import measure
        cs = measure.find_contours(zz,invΛ, fully_connected='low', positive_orientation='low')
        from wavedata import Wave
        wx,wy = Wave(x),Wave(y)
        return [Wave( wy(c[:,1]), wx(c[:,0]) ) for c in cs][0]
    pwaves = [contourwave(1/p,zz).rename(f'{abs(p):.2f}µm')
         for p in Λs] if Λs is not None else []
    from plot import plot
    save = f"phase matching plot, {sell} {Type}{f' backward {sig}'*(prop[0]<1)}{f' backward {idl}'*(prop[1]<1)}"
    import matplotlib.pyplot as plt
    zz = np.log(np.abs(zz))
    zz -= np.nanmin(zz)
    if 'ktp'==sell[:3] and not Type=='zzz': zz = zz**3
    if Type=='zzz': zz = zz**(1/3)
    ww = Wave2D(zz,xs=x,ys=y)
    plot(waves=pwaves+pumpwaves,
        c=['k' for p in pwaves]+pumpcolors,i=['2' for p in pwaves]+['1' for p in pumpwaves],
        lines=colorlines,image=ww,x=xlabel,y=ylabel,legendtext='',
        xlim=(x0*prop[0],x1*prop[0]),ylim=(x0*prop[1],x1*prop[1]),corner='lower right',
        colormap=getattr(plt.cm,'terrain_r'),fontsize=9,save=save,**plotargs)
    return ww
def w1vsw2old(w1=None,w2=None,Λ=None,w2onxaxis=True,x0=440,x1=3000,num=101,**ppargs):
    print('w1,w2 order is backwards for yzy')
    from skimage import measure
    def period(x,y):
        return polingperiod(w1=y,w2=x,**ppargs) if w2onxaxis else polingperiod(w1=x,w2=y,**ppargs)
    x,y = np.linspace(x0,x1,num),np.linspace(x0,x1,num)
    yy,xx = np.meshgrid(y,x)
    zz = 1./period(xx,yy)
    z0 = 1./period(w1,w2) if w1 is not None else 1/Λ
    contours = measure.find_contours(zz,z0, fully_connected='low', positive_orientation='low')
    from wavedata import Wave
    wx,wy = Wave(x),Wave(y)
    ws = [Wave( wy(c[:,1]), wx(c[:,0]) ) for c in contours]
    return ws[0]
def w1vsw2(*args,**kwargs):
    return phasematchcurve(*args,**kwargs)
def phasematchcurve(w1=None,w2=None,Λ=None,x0=500,x1=3000,dx=5,prop=None,plot=False,qpmargs=None,**ppargs):
    prop = prop if prop else (1,1) if Λ else (sign(w1),sign(w2))
    # option for ±halfmax of phasematchcurve?
    def pperiod(x,y):
        if qpmargs is not None:
            @np.vectorize
            def qpmΛ(x,y):
                from modes import qpm
                return qpm(λ1=x,λ2=y,**qpmargs).Λ
            return qpmΛ(x,y)
        return polingperiod(w1=x,w2=y,**ppargs)
    from wavedata import wrange,Wave2D
    x,y = (np.linspace(x0,x1,(x1-x0)//5+1),np.linspace(x0,x1,(x1-x0)//5+1)) if dx is None else (wrange(x0,x1,dx),wrange(x0,x1,dx))
    x,y = x*prop[0],y*prop[1]
    yy,xx = np.meshgrid(y,x)
    zz = 1./pperiod(xx,yy)
    z0 = 1./pperiod(w1,w2) if w1 is not None else 1/Λ
    from skimage import measure
    contours = measure.find_contours(zz,z0, fully_connected='low', positive_orientation='low')
    # convert to single contour separated by nans
    px,py = np.arange(len(x)),np.arange(len(y))
    cx,cy = [],[]
    for contour in contours:
        cx = np.concatenate((cx,contour[:,1],[nan]))
        cy = np.concatenate((cy,contour[:,0],[nan]))
    cx,cy = np.interp(cx[:-1],px,x),np.interp(cy[:-1],py,y) # convert array index to wavelength
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.plot(cx,cy,linewidth=2)
        plt.axis((y.min(),y.max(),x.min(),x.max()))
        plt.xlabel(f'$λ_2$ (nm)'); plt.ylabel(f'$λ_1$ (nm)')
        plt.ticklabel_format(style='plain',useOffset=False)
        plt.savefig('phasematchcurve.png', bbox_inches='tight')
        plt.minorticks_on()
        plt.grid(True, which='major', linestyle='-', color='0.85')
        plt.grid(True, which='minor', linestyle='--', color='0.85')
        plt.show()
    return list(cy),list(cx)
def phasematchplot(w1=None,w2=None,Λ=None,returnwave=False,w2onxaxis=False,x0=440,x1=3000,num=101,xscale=None,yscale=None,save='out',**kwargs):
    from skimage import measure
    def period(x,y):
        return polingperiod(w1=x,w2=y,**kwargs) if not w2onxaxis else polingperiod(w1=y,w2=x,**kwargs)
    y0,y1 = x0,x1
    x,y = (np.linspace(x0,x1,num),np.linspace(y0,y1,num)) if ('qpmargs' not in kwargs) else (np.linspace(500,1800,27),np.linspace(500,1800,27)) # (np.linspace(600,1800,7),np.linspace(600,1800,7))
    if xscale is not None: x = xscale
    if yscale is not None: y = yscale
    xx,yy = np.meshgrid(x,y)
    zz = 1./period(xx,yy)
    z0 = 1./period(w1,w2) if w1 is not None else 1/Λ
    # plt.imshow(zz,extent=[x0,x1,y0,y1],origin='lower',interpolation='bilinear',cmap=plt.cm.gray)
    px,py = np.arange(len(x)),np.arange(len(y))
    contours = measure.find_contours(zz,z0, fully_connected='low', positive_orientation='low')
    # convert to single contour separated by nans
    cx,cy = [],[]
    for contour in contours:
        cx = np.concatenate((cx,contour[:,1],[nan]))
        cy = np.concatenate((cy,contour[:,0],[nan]))
    cx,cy = np.interp(cx,px,x),np.interp(cy,py,y) # convert array index to wavelength
    if returnwave:
        from wavedata import Wave
        return Wave(cy[:-1][::-1],cx[:-1][::-1])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.plot(cy,cx,linewidth=2)
    plt.axis((y.min(),y.max(),x.min(),x.max()))
    plt.xlabel(f'λs (nm)'); plt.ylabel(f'λi (nm)')
    plt.ticklabel_format(style='plain',useOffset=False)
    plt.savefig(save+'.png', bbox_inches='tight')
    plt.show()
def fixedpumpdc(λp,Λ0=None,λ0=None,λ1=None,temp=20,sell='mglnridgewg',Type='yzy',returnwaves=False):
    from wavedata import Wave
    Λ0 = Λ0 if Λ0 is not None else polingperiod(2*λp,2*λp,temp=temp,sell=sell,Type=Type)
    λ0 = λ0 if λ0 is not None else int(2*λp-100)
    λ1 = λ1 if λ1 is not None else int(2*λp+100)
    wi = np.linspace(λ0,λ1,λ1-λ0+1)
    ws = np.linspace(1/(1/λp-1/λ0),1/(1/λp-1/λ1),λ1-λ0+1)
    # print(wi)
    # Λ = polingperiod(725,wi,temp=temp,sell=sell,Type=Type)
    # Wave.plots(Wave(Λ,wi),Wave(Λ0,wi))
    # @np.vectorize
    def λi(λs):
        # Λ = polingperiod(wi,λs,temp=temp,sell=sell,Type=Type) ## 
        Λ = polingperiod(λs,wi,temp=temp,sell=sell,Type=Type)
        return Wave(1/Λ,wi).xaty(1/Λ0)
    w2 = Wave([λi(w) for w in ws],ws)
    λps = 1/(1/w2+1/ws)
    ww = [Wave(ws,λps,'ws'), Wave(w2,λps,'w2'), Wave([min(ws),max(ws)],[λp,λp],name='λp')]
    if returnwaves:
        return ww
    def showinterpolation():
        Wave.plots(*ww,x='λps')
        return ''
    showinterpolation()
    # assert min(λps)<λp<max(λps), f'min(λps),λp,max(λps):{min(λps),λp,max(λps)} expand λ0,λ1 or use closer pump'+showinterpolation()
    λ1,λ2 = Wave(ws,λps)(λp),Wave(w2,λps)(λp)
    return λ1,λ2
def fixedpumpdc3b(λp,Λ0,λa0=None,λa1=None,λb0=None,λb1=None,temp=20,sell='ktpwg',Type='yzy',npy=None,debug=0):
    from wavedata import Wave
    λa0,λa1 = λa0 if λa0 is not None else λp*2-200,λa1 if λa1 is not None else λp*2+200
    λb0,λb1 = λb0 if λb0 is not None else λp*2-200,λb1 if λb1 is not None else λp*2+200
    wix,wsx = np.linspace(λb0,λb1,λb1-λb0+1),np.linspace(λa0,λa1,λa1-λa0+1)
    @np.vectorize
    def λi(λs):
        Λ = polingperiod(λs,wix,temp=temp,sell=sell,Type=Type,npy=npy)
        return Wave(1/Λ,wix).xaty(1/Λ0)
    w2 = λi(wsx)
    λps = 1/(1/w2+1/wsx)
    if debug:
        Wave.plots(Wave(wsx,λps),Wave(w2,λps))
    λ1,λ2 = Wave(wsx,λps)(λp),Wave(w2,λps)(λp)
    return λ1,λ2 # np.asscalar(np.asarray(λ1)),np.asscalar(np.asarray(λ2))
def loadsfgscan(file,skip=None,returnce=True,pumpince=False):
    def findlineinfile(s,file):
        with open(file,'r') as f:
            for n,line in enumerate(f):
                if line.startswith(s):
                    return n
        return None
    # from optical import loadfile
    def loadfile(file,dtype=None,delimiter='\t',skip=0):
        return np.array(np.genfromtxt(file, dtype=dtype, delimiter=delimiter, names=True, skip_header=skip).tolist())
    if skip is None:
        skip = findlineinfile('Wavelength',file)
        assert skip, "didn't find start of data"
    wavelength,pumpin,pumpout,shg = loadfile(file,skip=skip).T
    if returnce:
        from wavedata import Wave
        if pumpince:
            return Wave(100*shg/pumpin**2,wavelength)
        return Wave(100*shg/pumpout**2,wavelength)
    return wavelength,pumpin,pumpout,shg
def loadshgscan(file,skip=None,returnce=True,pumpince=False):
    def findlineinfile(s,file):
        with open(file,'r') as f:
            for n,line in enumerate(f):
                if line.startswith(s):
                    return n
        return None
    # from optical import loadfile
    def loadfile(file,dtype=None,delimiter='\t',skip=0):
        return np.array(np.genfromtxt(file, dtype=dtype, delimiter=delimiter, names=True, skip_header=skip).tolist())
    if skip is None:
        skip = findlineinfile('# Wavelength (nm), Corrected Input Power (mW), Corrected Output Power (mW), Corrected SHG Power (uW), Corrected Efficiency (%/W)',file)
        assert skip, "didn't find start of data"
    lf = loadfile(file,skip=skip,delimiter=',').T
    wavelength,pumpin,pumpout,shg,ce = lf
    # assert all(ce==100*shg/pumpout**2)
    assert np.allclose(0,ce-100*shg/pumpout**2)
    return wavelength,pumpin,pumpout,shg,ce
def qpmphasematchcontours(Λ,wp=(405,532,780),x0=800,x1=1600,dx=100,qpmargs=None,ppargs=None):
    def pp(x,y):
        from modes import qpm
        @np.vectorize
        def qpmΛ(x,y):
            return qpm(λ1=x,λ2=y,**qpmargs).Λ
        return qpmΛ(x,y) if qpmargs is not None else polingperiod(w1=x,w2=y,**ppargs) # if w2onxaxis else polingperiod(w1=y,w2=x,**ppargs)
    from wavedata import wrange,Wave2D
    x,y = wrange(x0,x1,dx),wrange(x0,x1,dx)
    yy,xx = np.meshgrid(y,x)
    zz = 1./pp(yy,xx)
    Λ = pp(*Λ) if hasattr(Λ,'__len__') else Λ
    def contourwave(invΛ):
        from skimage import measure
        cs = measure.find_contours(zz,invΛ, fully_connected='low', positive_orientation='low')
        from wavedata import Wave
        wx,wy = Wave(x),Wave(y)
        return [Wave( wy(c[:,1]), wx(c[:,0]) ) for c in cs][0] if cs else Wave()
    return contourwave(1./Λ).rename(f"{Λ:g}µm")

if __name__ == '__main__':
    def test():
        print(index(1064,'ktp'))
        print(index(1064,'ktpwg'))
        print(polingperiod(1064,1064,'ktp'))
        print(polingperiod(1064,1064,'ktpwg'))
        print(polingperiodbandwidths(1064,1064,'ktpwg'))
    # test()
