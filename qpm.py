import numpy as np
from wavedata import Wave,wrange,dotdict
import sellmeier

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
        self.w1,self.w2,self.w3 = sellmeier.qpmwavelengths(self.w1,self.w2,self.w3)
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
        return sellmeier.polingperiod(w1,w2,sell,Type,temp,npy,npz,qpmargs)
    def bw(self,L=10,kind=None,**kwargs):
        Δλ = sellmeier.polingperiodbandwidths(self.w1,self.w2,sell=self.sell,Type=self.Type,L=L,kind=kind,**kwargs)
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
    def dcwavelengths(self,λp=None,Λ0=None,temp=None,λa0=None,λa1=None,λb0=None,λb1=None,Δλ=300,alt=False,debug=False):
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
            Λ = sellmeier.polingperiod(λs,wix,self.sell,self.Type,temp,self.npy,self.npz,self.qpmargs)
            return Wave(1/Λ,wix).xaty(1/Λ0,i1d=False)
        w2 = λi(wsx) # Wave(w2,wsx).plot()
        λps = 1/(1/w2+1/wsx)
        λ1,λ2 = (Wave(wsx[::-1],λps[::-1]).atx(λp,monotonicx=0),Wave(w2[::-1],λps[::-1]).atx(λp,monotonicx=0)) if alt else (Wave(wsx,λps).atx(λp,monotonicx=0),Wave(w2,λps).atx(λp,monotonicx=0))
        if debug:
            Wave.plots(Wave(wsx,λps,'wsx'),Wave(w2,λps,'w2'),lines=[λp],legendtext=f"λ1:{λ1:g}\nλ2:{λ2:g}")
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
    # reconcile polingperiod sell vs modesolver sell (bulk,material,...?)
    # how to return a modified instance?
    # return list of modified instances when input arg is a list?
    # make subclass of np.array() so that [qpm0,qpm1,qpm2].method() = [qpm0.method(),qpm1.method(),qpm2.method()]?


class Qpmlnwg():
    """docstring for Qpmlnwg"""
    # qdl = qpm(1550,810,8,1.9,18,9.50,sell='ln',res=0.2)
    # md = modesolver(λ=1550,width=8,depth=1.9,ape=23.5,rpe=24.5,sell='ln',apetemp=320,rpetemp=300,res=0.2,limits=(-15,15,-20,2))
    # Qpmdata

    # TODO: switch to implicit args:
    #   qpmargs = 'λ1 λ2 w d a r sell Type modes apetemp rpetemp method verbose'.split()
    #   nonqpmargs = 'temp'
    #   'mode,nummodes,planar,gap,sellcover,nair,braggfraction,bendradius,taper,res,limits,apetemp,rpetemp,ridgewidth,ridgedepth,slantx,xcut,rotate'
    def __init__(self,λ1=None,λ2=None,w=8,d=1.9,a=23.5,r=24.5,aa=0,temp=20,atemp=320,rtemp=300,**qpmargs):
        λ2 = λ2 if λ2 is not None else λ1
        self.λ1,self.λ2,self.w,self.d,self.a,self.r,self.aa,self.temp,self.atemp,self.rtemp = λ1,λ2,w,d,a,r,aa,temp,atemp,rtemp
        self.sell,self.Type = 'ln','zzz'
        # self.__dict__.update(qpmargs)
        self.qpmargs = dotdict(qpmargs)
        _,_,self.λ3 = sellmeier.qpmwavelengths(λ1,λ2)
    def __call__(self,λ1=None,λ2=None,w=None,d=None,a=None,r=None,aa=None,temp=None,atemp=None,rtemp=None):
        λ1,λ2,w,d,a,r,aa,temp,atemp,rtemp = [x if x is not None else getattr(self,s) for x,s in zip((λ1,λ2,w,d,a,r,aa,temp,atemp,rtemp),'λ1 λ2 w d a r aa temp atemp rtemp'.split())]
        return Qpmlnwg(λ1,λ2,w,d,a,r,aa,temp,atemp,rtemp,**self.qpmargs)
    def sellmeierΛ(self,λ1=None,λ2=None,temp=None):
        λ1 = λ1 if λ1 is not None else self.λ1
        λ2 = λ2 if λ2 is not None else self.λ2
        temp = temp if temp is not None else self.temp
        return sellmeier.polingperiod(λ1,λ2,sell='lnwg',Type=self.Type,temp=temp)
    def qd(self,λ1=None,λ2=None,w=None,d=None,a=None,r=None,aa=None):
        from modes import qpm,Qpmdata
        # λ1,λ2,w,d,a,r,aa = [x if x is not None else getattr(self,s) for x,s in zip((λ1,λ2,w,d,a,r,aa),'λ1 λ2 w d a r aa'.split())]
        λ1,λ2,w,d,a,r,aa = self.λ1,self.λ2,self.w,self.d,self.a,self.r,self.aa
        sell,Type,atemp,rtemp = self.sell,self.Type,self.atemp,self.rtemp
        assert 0==aa # print(self.idstr(),self.qpmargs)
        return qpm(λ1,λ2,w,d,a,r,sell=sell,Type=Type,modes=(0,0,0),temp=20,apetemp=atemp,rpetemp=rtemp,method=None,verbose=False,**self.qpmargs)
    def md(self,num):
        return self.qd().mds[num-1]
    def Λ2Δλ1(self,Λ,Λ0=None,shg=False):
        Λ0 = Λ0 if Λ0 is not None else self.Λ()
        if shg:
            dinvΛdλ = 1/sellmeier.polingperiod(self.λ1+1,self.λ2+1,self.sell+'wg',self.Type) - 1/sellmeier.polingperiod(self.λ1,self.λ2,self.sell+'wg',self.Type)
        else:
            dinvΛdλ = 1/sellmeier.polingperiod(self.λ1+1,self.λ2,self.sell+'wg',self.Type) - 1/sellmeier.polingperiod(self.λ1,self.λ2,self.sell+'wg',self.Type)
        return (1/Λ - 1/Λ0)/dinvΛdλ
    def λ(self,num):
        return [self.λ1,self.λ2,self.λ3][num-1]
    def λs(self):
        return [self.λ1,self.λ2,self.λ3]
    def η(self):
        qd = self.qd()
        return qd.deadsfgce
    def Λ(self,temp=None):
        qd = self.qd()
        temp = temp if temp is not None else self.temp
        return 1/(1/qd.Λ - 1/self.sellmeierΛ(temp=20) + 1/self.sellmeierΛ(temp=temp))
    def dcwavelengths(self,λp=None,Λ0=None,temp=None,λa0=None,λa1=None,λb0=None,λb1=None,Δλ=200,alt=False,debug=False):
        λp = λp if λp is not None else self.λ3
        Λ0 = Λ0 if Λ0 is not None else self.Λ()
        temp = temp if temp is not None else self.temp
        λa0,λa1 = λa0 if λa0 is not None else self.λ1-Δλ,λa1 if λa1 is not None else self.λ1+Δλ
        λb0,λb1 = λb0 if λb0 is not None else self.λ2-Δλ,λb1 if λb1 is not None else self.λ2+Δλ
        λa0,λb0 = max(λa0,λp+1),max(λb0,λp+1)
        # test and print warning for λp < λdegen /2?
        wix,wsx = np.linspace(λb0,λb1,int(np.round(λb1-λb0+1))),np.linspace(λa0,λa1,int(np.round(λa1-λa0+1)))
        @np.vectorize
        def λi(λs):
            # Λ = sellmeier.polingperiod(λs,wix,self.sell,self.Type,temp,self.npy,self.npz,self.qpmargs)
            # sellmeier.polingperiod(λ1,λ2,sell='lnwg',Type=Type,temp=temp)
            Λ = 1/( 1/self.Λ() - 1/self.sellmeierΛ() + 1/self.sellmeierΛ(λs,wix) )
            # Λ = self(λ1=λs,λ2=wix).Λ()
            return Wave(1/Λ,wix).xaty(1/Λ0,i1d=False)
        w2 = λi(wsx) # Wave(w2,wsx).plot()
        λps = 1/(1/w2+1/wsx)
        λ1,λ2 = (Wave(wsx[::-1],λps[::-1]).atx(λp,monotonicx=0),Wave(w2[::-1],λps[::-1]).atx(λp,monotonicx=0)) if alt else (Wave(wsx,λps).atx(λp,monotonicx=0),Wave(w2,λps).atx(λp,monotonicx=0))
        if debug:
            Wave.plots(Wave(wsx,λps,'wsx'),Wave(w2,λps,'w2'),lines=[λp],legendtext=f"λ1:{λ1:g}\nλ2:{λ2:g}")
        return λ1,λ2 # np.asscalar(np.asarray(λ1)),np.asscalar(np.asarray(λ2))
    def λdc(self,*args,**kwargs):
        return self.dcwavelengths(*args,**kwargs)
    def Δλvswidth(self,ws=None,plot=False,save=None,**plotargs):
        v = self.Λvswidth(ws=ws)
        x0 = v.quadminloc()
        q0 = self(w=int(round(x0)))
        return Wave([self.Λ2Δλ1(Λ,Λ0=q0.Λ()) for Λ in v],v.x)
    def sfgdesigndoc(self,dw=0.5,prs=None,show=True,fork=True,**plotargs):
        from pypowerpoint import Presentation,addadvrslide
        prs = prs if prs is not None else Presentation()
        titles = [
            f"{self.λ1}+{self.λ2}→{self.λ3:.0f} poling period vs width",
            f"{self.λ1}+{self.λ2}→{self.λ3:.0f} conversion efficiency vs width",
            f"{self.λ1}+{self.λ2}→{self.λ3:.0f} phasematching",
            f"MFD vs width",
            f"Transmission vs width" ]
        subtitles = ['','','','','']
        pngs = [
            self.Λvswidth(ws=wrange(6,12,0.5),temps=[20,40] if 20==self.temp else [20,self.temp],show=show,fork=fork,**plotargs),
            self.ηvswidth(ws=wrange(6,12,0.5),show=show,fork=fork,**plotargs),
            self.phasematchingcurves(show=show,fork=fork,**plotargs),
            self.mfdsvswidth(ws=wrange(1,6,dw),show=show,fork=fork,**plotargs),
            self.transmissionsvswidth(ws=wrange(1,6,dw),show=show,fork=fork,**plotargs) ]
        print(pngs)
        from time import sleep
        sleep(20)
        for title,subtitle,pngfile in zip(titles,subtitles,pngs):
            addadvrslide(prs,title,subtitle,pngfile)
        file = f"figs/lnwg sfg design parameters {self.idstr(1)}.pptx"
        prs.save(file)
        return prs
    def phasematchingcurve(self,Λ0=None,temp=None,δλ=None,Δλ1=None,Δλ2=None,exact=False,plot=False,save=None,**plotargs):
        save = save if save is not None else f"phasematching λ1 vs λ2, {self.idstr()}"
        Λ0 = Λ0 if Λ0 is not None else self.Λ()
        δλ = δλ if δλ is not None else 25 if exact else 10
        Δλ1 = Δλ1 if Δλ1 is not None else 200
        Δλ2 = Δλ2 if Δλ2 is not None else Δλ1
        temp = temp if temp is not None else self.temp
        λ1,λ2 = self.λ1,self.λ2
        @np.vectorize
        def pperiod(x,y):
            return self(λ1=x,λ2=y).Λ(temp=temp) if exact else 1/(1/self(λ1=λ1,λ2=λ2).Λ(temp=temp)
            - 1/sellmeier.polingperiod(λ1,λ2,sell='lnwg',Type=self.Type,temp=temp)
            + 1/sellmeier.polingperiod( x, y,sell='lnwg',Type=self.Type,temp=temp))
        x,y = wrange(λ1-0.5*Δλ1,λ1+0.5*Δλ1,δλ),wrange(λ2-0.5*Δλ2,λ2+0.5*Δλ2,δλ)
        yy,xx = np.meshgrid(y,x)
        zz = 1/pperiod(xx,yy)
        z0 = 1/Λ0
        from skimage import measure
        contours = measure.find_contours(zz,z0, fully_connected='low', positive_orientation='low')
        # convert to a single contour line separated by nans
        px,py = np.arange(len(x)),np.arange(len(y))
        cx,cy = [],[]
        for contour in contours:
            cx = np.concatenate((cx,contour[:,1],[np.nan]))
            cy = np.concatenate((cy,contour[:,0],[np.nan]))
        cx,cy = np.interp(cx[:-1],px,x),np.interp(cy[:-1],py,y) # convert array index to wavelength
        u = Wave(cy,cx)
        if plot:
            Wave.plots(u,x='$λ_1$ (nm)',y='$λ_2$ (nm)',grid=1,c='1',xlim='f',ylim='f',aspect=1 if self.λ1==self.λ2 else None,save=save,seed=int(self.λ1+self.λ2),**plotargs)
        return u
    def phasematchingcurves(self,Λ0=None,temps=None,δλ=None,Δλ1=None,Δλ2=None,exact=False,plot=True,save=None,**plotargs):
        Λ0 = Λ0 if Λ0 is not None else self.Λ()
        temps = [20,40] if 20==self.temp else [20,self.temp]
        save = save if save is not None else f"phasematching λ1 vs λ2, {','.join([f'{t:g}C' for t in temps])}, {self.idstr()}"
        us = [self.phasematchingcurve(Λ0=Λ0,temp=temp,δλ=δλ,Δλ1=Δλ1,Δλ2=Δλ2,exact=exact,plot=False,save=save,**plotargs).rename(f"{temp}°C") for temp in temps]
        if plot:
            Wave.plots(*us,x='$λ_1$ (nm)',y='$λ_2$ (nm)',grid=1,xlim='f',ylim='f',aspect=1 if self.λ1==self.λ2 else None,save=save,seed=int(self.λ1+self.λ2),legendtext=f"Λ = {Λ0:.3f}µm",**plotargs)
            return f"figs/{save}.png"
        return us
    def Λvswidth(self,ws=None,temps=None,plot=False,save=None,**plotargs):
        save = save if save is not None else f"Λ vs width, {self.idstr()}"
        ws = ws if ws is not None else np.linspace(6,12,7)
        us = [Wave([self(w=w).Λ(temp=temp) for w in ws],ws,f"{temp}°C" if temps is not None else "").setplot(c=i) for i,temp in enumerate(temps if temps is not None else [None])]
        u0s = [u.quadminloc(aswave=1).setplot(m='o',l=' ',c=i).rename(f"Λ={u.quadmin():.3f}µm") for i,u in enumerate(us)]
        if plot or plotargs:
            Wave.plots(*us,*u0s,x='width (µm)',y='Λ (µm)',legendtext=f"noncrit width {us[0].quadminloc():.1f}µm",corner='upper right',save=save,seed=int(self.λ1+self.λ2),showseed=1,**plotargs)
            return f"figs/{save}.png"
        return us if temps is not None else us[0].setplot(c=None)
    def ηvswidth(self,ws=None,noncrit=True,plot=False,save=None,**plotargs):
        save = save if save is not None else f"η vs width, {self.idstr()}"
        ws = ws if ws is not None else np.linspace(6,12,7)
        u = Wave([self(w=w).η() for w in ws],ws)
        u0 = u.quadmaxloc(aswave=1).setplot(m='o',l=' ').rename(f"{u.quadmax():.0f}%/W/cm² peak")
        if plot or plotargs:
            if noncrit:
                x0 = self.Λvswidth(ws).quadminloc()
                u1 = Wave([u(x0)],[x0]).setplot(m='o',l=' ',mf='w').rename(f"{u(x0):.0f}%/W/cm² noncrit")
                Wave.plots(u,u0,u1,x='width (µm)',y='$η_\mathrm{{SFG}}$ (%/W/cm²)',c='111',corner='lower center',save=save,seed=int(self.λ1+self.λ2),**plotargs)
            else:
                Wave.plots(u,u0,x='width (µm)',y='η (%/W/cm²)',c='11',corner='lower center',save=save,seed=int(self.λ1+self.λ2),**plotargs)
            return f"figs/{save}.png"
        return u
    def mfdvswidth(self,num=1,axis=0,ws=None,plot=False,**plotargs):
        axis = 0 if 'x'==axis else 1 if 'y'==axis else axis
        assert num in (1,2,3) and axis in (0,1)
        ws = ws if ws is not None else np.linspace(1,5,11)
        u = Wave([self(w=w).md(num).mfdx if 0==axis else self(w=w).md(num).mfdy for w in ws],ws)
        # u0 = u.quadminloc(aswave=1).setplot(m='o',l=' ').rename(f"{u.quadmin():.0f}µm min")
        if plot or plotargs:
            return Wave.plots(u,x='width (µm)',y=f"MFD{'xy'[axis]} (µm)",c='00',grid=1,xlim='f',ylim=(0,0.6*abs(self.qpmargs.limits[2])),**plotargs)
        return u
    def mfdsvswidth(self,ws=None,plot=False,save=None,**plotargs):
        save = save if save is not None else f"MFD vs width, {self.idstr()}"
        us = [self.mfdvswidth(num,axis,ws=ws).rename(f"{'xy'[axis]} {self.λ(num):.0f}nm").setplot(c=num+1,l='0' if 0==axis else '1') for num in (1,2,3) for axis in (0,1)]
        if plot or plotargs:
            Wave.plots(*us,x='width (µm)',y=f'MFD (µm)',c='00',grid=1,xlim='f',ylim=(0,0.6*abs(self.qpmargs.limits[2])),corner='upper right',save=save,seed=int(self.λ1+self.λ2),**plotargs)
            return f"figs/{save}.png"
        return us
    def transmissionvswidth(self,num=1,ws=None,fiber=None,plot=False,**plotargs):
        assert num in (1,2,3)
        ws = ws if ws is not None else np.linspace(1,5,11)
        mds = [self(w=w).md(num) for w in ws]
        fiber = fiber if fiber is not None else mds[0].fiber()
        u = Wave([100*md.fibercoupling(fiber=fiber) for md in mds],ws)
        u0 = u.quadmaxloc(edgemax=True,aswave=1).setplot(m='o',l=' ').rename(f"{u.quadmax(edgemax=True):.1f}% peak")
        if plot or plotargs:
            Wave.plots(u,u0,x='width (µm)',y=f"MFD{'xy'[axis]} (µm)",c='00',xlim='f',ylim=(0,None),**plotargs)
        return u,fiber
    def transmissionsvswidth(self,ws=None,plot=False,save=None,**plotargs):
        save = save if save is not None else f"transmission vs width, {self.idstr()}"
        us,fibers = zip(*[self.transmissionvswidth(num,ws=ws) for num in (1,2,3)])
        us = [u.rename(f"{self.λ(num):.0f}nm, PM{f}").setplot(c=num+1) for num,u,f in zip((1,2,3),us,fibers)]
        u0s = [u.quadmaxloc(edgemax=True,aswave=1).setplot(m='o',l=' ',c=num+1).rename(f"{u.quadmax(edgemax=True):.1f}% at {u.quadmaxloc(edgemax=True,):.1f}µm") for num,u in zip((1,2,3),us)]
        if plot or plotargs:
            Wave.plots(*us,*u0s,x='width (µm)',y=f'fiber coupling (%)',c='00',xlim='f',ylim=(50,100),corner='lower right',save=save,seed=int(self.λ1+self.λ2),**plotargs)
            return f"figs/{save}.png"
        return us
    def idstr(self,short=False):
        λ1,λ2,w,d,a,r,aa = self.λ1,self.λ2,self.w,self.d,self.a,self.r,self.aa
        s = f"{λ1:g}+{λ2:g}" if short else f"{λ1:g}+{λ2:g} {w:g}w {d:g}sa {a:g}a"
        s += f" {r:g}r"*bool(r)+f" {aa:g}a2"*bool(aa)
        s += f" {self.atemp:g}°a"*bool(320!=self.atemp)+f" {self.rtemp:g}°r"*bool(300!=self.rtemp)
        return s

# η vs width plots for DOE 7120
def ηplot0(s='',plotit=True,**args):
    args = args if args else dict(depth=1.9,ape=23.5,rpe=24.5,sell='ln',res=0.2,apetemp=320,rpetemp=300)
    def peak(w,min=False):
        x = w.minloc() if min else w.maxloc()
        return Wave([0,w.min() if min else w.max()],[x,x])#.setplot(l='1')
    def sfgvswidth(λ1,λ2):
        widths = wrange(4,15,0.2)
        limits = (-15,15,-20,2)
        print('λ1,λ2',λ1,λ2)
        from wavedata import track
        from modes import qpm
        qds = [qpm(λ1,λ2,width=wi,**args,limits=limits) for wi in track(widths)]
        wη = Wave([qd.deadshgce for qd in qds],widths,f"{λ1}+{λ2}nm")
        wΛ = Wave([qd.Λ for qd in qds],widths,f"{λ1}+{λ2}nm")
        # qqs = [qpm(None,λ2,width=wi,**args,limits=limits,Λ=wΛ.min()) for wi in widths] # wrong
        qqs = [qpm(None,None if λ2==λ1 else λ2,width=wi,**args,limits=limits,Λ=wΛ.min()) for wi in track(widths)]
        wλ = Wave([qq.λ1 for qq in qqs],widths,f"{λ1}+{λ2}nm")
        return wη,wΛ,wλ-wλ.max()
    def doplot():
        λs = [(810,810),(1064,1064),(1300,1300),(1550,1550),(1950,1950),]
        # λs += [(1550,810),(1950,730),(1623,1350),(1550,1950),(1950,1550)] # first one tunable
        # λs += [(1550,810),(1950,730),(1623,1350),(1950,1550)]
        λs += [(1550,810),(1950,730),(1623,1350),(1930,1310),(1950,1550)]
        if '810'==s: λs = [(810,810)]
        us,vs,ws = zip(*[sfgvswidth(λ1,λ2) for λ1,λ2 in λs])
        if plotit:
            Wave.plots(*us,*[peak(u) for u in us],x='width (µm)',y='SHG equivalent η (%/W/cm²)',xlim=(4,15),ylim=(0,None),grid=1,showseed=1,seed=22,scale=(1.5,2),
                c='0123456789',l='00000111110000011111',corner='upper right',darken=1,
                save='shg equivalent conversion efficiency vs width, with dead layer, EXB '+s)
            Wave.plots(*reversed(vs),*reversed([peak(v,min=1) for v in vs]),x='width (µm)',y='Λ (µm)',xlim=(4,15),ylim=(0,None),grid=1,showseed=1,seed=22,scale=(1.5,2),
                c='9876543210',l='11111000001111100000',corner='upper right',darken=1,
                save='poling period vs width, with dead layer, EXB '+s)
            ws = [w.rename(f"{λ1}+{λ2}{'*'*(λ1!=λ2)}nm") for w,(λ1,λ2) in zip(ws,λs)]
            # ylim = (-1 if 'zoom 1'==s else -5 if 'zoom 5'==s else None,0 if s.startswith('zoom') else None)
            ylim = (-5,1)
            Wave.plots(*ws,x='width (µm)',y='Δλ (nm)',xlim=(4,15),ylim=ylim,grid=1,showseed=1,seed=22,scale=1.5,
                c='0123456789',l='0000011111',corner='lower right',darken=1,legendtext='*=fixed',
                save='wavelength vs width, with dead layer, EXB '+s)
            dλs = [0.5*sellmeier.polingperiodbandwidths(λ1,λ2,sell='lnwg',Type='zzz',kind='shg' if λ1==λ2 else 'sfg1',L=10) for λ1,λ2 in λs]
            Wave.plots(*[(w+f*dλ).rename(w.name if +1==f else '') for w,dλ in zip(ws,dλs) for f in (-1,+1)],x='width (µm)',y='Δλ (nm)',xlim=(4,15),ylim=(-5,3),grid=1,showseed=1,seed=22,scale=1.5,
                c='00112233445566778899',l='00000000001111111111',corner='lower right',darken=1,legendtext='*=fixed',
                save='delta wavelength vs width, with dead layer, EXB '+s)
            us = [(w/(2*dλ)).rename(w.name) for w,dλ in zip(ws,dλs)]
            Wave.plots(*us,x='width (µm)',y='Δλ/λFWHM',xlim=(4,15),ylim=(-10,2),grid=1,showseed=1,seed=22,scale=1.5,
                c='0123456789',l='0000011111',corner='lower right',darken=1,legendtext='*=fixed',
                save='spectral width variation vs width, with dead layer, EXB '+s)
        return us,vs
    return doplot()


if __name__ == '__main__':
    ...
    # q = Qpm(1064,sell='ktpwg')
    # print(q.Λ)
    # q.bw()
    # print(q.λdc(531))

    # q = Qpmlnwg(1550)
    # print(q().Λ())
    # print(q(1550,1550).Λ())
    # print(q(1560,1560).Λ())
    # print(q(1570,1570).Λ())
    # q.Λvswidth(plot=1)
    # q.Λvswidth(plot=1,temps=[20,40])
    # q.ηvswidth(plot=1)
    # q = Qpmlnwg(1550,810)
    # q.mfdvswidth(1,'x',plot=1)
    # q.mfdvswidth(1,'y',plot=1)
    # q.mfdvswidth(2,'x',plot=1)
    # q.mfdvswidth(2,'y',plot=1)
    # q.mfdvswidth(3,'x',plot=1)
    # q.mfdvswidth(3,'y',plot=1)
    # q.mfdsvswidth(ws=np.linspace(1,5,11),plot=1)
    # q.transmissionsvswidth(ws=np.linspace(1,5,11),plot=1)
    # q = Qpmlnwg(1550,810,res=0.2,limits=(-15,15,-20,2))
    # q.sfgdesigndoc(dw=0.1)
    # q = Qpmlnwg(1550,810,res=0.5,limits=(-30,30,-60,2))
    # q = Qpmlnwg(1550,810,res=0.2,limits=(-30,30,-40,2))
    # q.sfgdesigndoc(dw=0.5)
    # q.sfgdesigndoc(dw=0.2)

    q = Qpmlnwg(1550,810,res=0.2,limits=(-15,15,-20,2))
    # Wave.plots(*q.phasematchingcurves(exact=1),*q.phasematchingcurves(exact=0))
    # q.phasematchingcurves()
    # print(q.λdc())

    # ηplot0(depth=1.9,ape=23.5,rpe=24.5,sell='ln',res=0.2,apetemp=320,rpetemp=300,s='Nov2023')

    #TODO
    # Qpmlnwg(2239,1550).transmissionvswidth(plot=True)
    # w1vsw2 at 20,40,60C
    # wdmdesign
