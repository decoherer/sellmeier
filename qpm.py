from sellmeiers import qpmwavelengths,polingperiod,polingperiodbandwidths
import numpy as np
class Qpm():
    """docstring for Qpm"""
    def __init__(self,w1=None,w2=None,**kwargs):
        # w1=None,w2=None,temp=20,sell='ktpwg',Type='zzz',w3=None,npy=None,npz=None,qpmargs=None,Λ=None
        # type 0 sfg period in um, w1,w2 in nm, temp in °C
        aa = ['w1','w2','temp','sell','Type','w3','npy','npz','qpmargs','Λ']
        self.__dict__ = {k:None for k in aa}
        # self.__dict__['w1'],self.__dict__['w2'] = w1,w2
        self.__dict__.update(kwargs)
        def setdefault(s,x): # equivalent to self.s = x if self.s is None else self.s
            if getattr(self,s) is None: setattr(self,s,x)
        for s,x in zip('w1 w2 temp Type npz'.split(),[w1,w2,20,'zzz',self.npy]):
            setdefault(s,x)
        self.w1,self.w2,self.w3 = qpmwavelengths(self.w1,self.w2,self.w3)
        setdefault('Λ',self.period())
    @property
    def λp(self):
        return self.w3
    def period(self,w1=None,w2=None,sell=None,Type=None,temp=None,
            w3=None,npy=None,npz=None,qpmargs=None):
        w1 = w1 if w1 is not None else self.w1
        w2 = w2 if w2 is not None else self.w2
        sell = sell if sell is not None else self.sell
        Type = Type if Type is not None else self.Type
        temp = temp if temp is not None else self.temp
        npy = npy if npy is not None else self.npy
        npz = npz if npz is not None else self.npz
        qpmargs = qpmargs if qpmargs is not None else self.qpmargs
        # print('w1,w2,sell,Type,temp,npy,npz,qpmargs',w1,w2,sell,Type,temp,npy,npz,qpmargs)
        return polingperiod(w1,w2,sell,Type,temp,npy,npz,qpmargs)
    def bw(self,L=10,kind=None,**kwargs):
        Δλ = polingperiodbandwidths(self.w1,self.w2,sell=self.sell,Type=self.Type,L=L,kind=kind,**kwargs)
        print(f" Δλ{kind}: {Δλ:.2g}nm" if kind else Δλ)
        return Δλ
    def dT(self,kind,verbose=True):
        ΔΛΔλ = self.dΛ(kind,verbose=1)
        ΔΛΔT = self.period(temp=self.temp+1) - self.period()
        ΔTΔλ = ΔΛΔλ/ΔΛΔT
        if verbose:
            print(f"ΔΛ/ΔT: {ΔΛΔT:g}µm/°C\nΔT/Δλ: {ΔTΔλ:g}°C/nm")
        return ΔTΔλ
    def dΛ(self,kind,verbose=True):
        w1,w2 = ((self.w1+1,self.w2+1) if 'shg'==kind else 
                 (self.w1+1,self.w2) if 'sfg1'==kind else 
                 (self.w1,self.w2+1) if 'sfg2'==kind else (None,None))
        ΔΛ = self.period(w1,w2) - self.period()
        if verbose:
            print(f"ΔΛ/Δλ: {ΔΛ:g}µm/nm")
        return ΔΛ
    def dcwavelengths(self,λp=None,Λ0=None,temp=None,λa0=None,λa1=None,λb0=None,λb1=None,Δλ=300,alt=False):
        from waves import Wave
        λp = λp if λp is not None else self.λp
        Λ0 = Λ0 if Λ0 is not None else self.Λ
        temp = temp if temp is not None else self.temp
        λa0,λa1 = λa0 if λa0 is not None else self.w1-Δλ,λa1 if λa1 is not None else self.w1+Δλ
        λb0,λb1 = λb0 if λb0 is not None else self.w2-Δλ,λb1 if λb1 is not None else self.w2+Δλ
        λa0,λb0 = max(λa0,λp+1),max(λb0,λp+1)
        # test and print warning for λp < λdegen /2?
        def rint(x): return int(np.round(x))
        wix,wsx = np.linspace(λb0,λb1,rint(λb1-λb0+1)),np.linspace(λa0,λa1,rint(λa1-λa0+1))
        @np.vectorize
        def λi(λs):
            Λ = polingperiod(λs,wix,self.sell,self.Type,temp,self.npy,self.npz,self.qpmargs)
            return Wave(1/Λ,wix).xaty(1/Λ0)
        w2 = λi(wsx) # Wave(w2,wsx).plot()
        λps = 1/(1/w2+1/wsx) # Wave.plots(Wave(wsx,λps),Wave(w2,λps))
        λ1,λ2 = (Wave(wsx[::-1],λps[::-1])(λp),Wave(w2[::-1],λps[::-1])(λp)) if alt else (Wave(wsx,λps)(λp),Wave(w2,λps)(λp))
        return λ1,λ2 # np.asscalar(np.asarray(λ1)),np.asscalar(np.asarray(λ2))
    def λdc(self,*args,**kwargs):
        return self.dcwavelengths(*args,**kwargs)
    ### return λ given temp
    ### return Λ vs temp given fixed λ
    ### return Δn vs depth given fixed λnp
    def λvariation(Λ,Λ0,dΛ):
        return λ+(Λ-Λ0)/dΛ
    # λ1,λ2 = q.λdc(λp,Λ0,temp,λa0,λa1,λb0,λb1,Δλ)
    # todo:
    # o specifying
    #  - if properly specified, return missing value, e.g. period
    #  - if underspecified, return 2D plot (or waterfall for 3D)
    #  - if overspecified, return error in each dimension
    # o generic plots, e.g. neff vs npy, mfd1x vs period
    # waveguide depth corresponding to polingperiod
if __name__ == '__main__':
    q = Qpm(1064,sell='ktpwg')
    print(q.Λ)
    q.bw()
    # print(q.λdc(531))