import numpy as np
def sellmeier(x,a,b,c,d,e,f):
    return np.sqrt(1+a/(1-b/x**2)+c/(1-d/x**2)+e/(1-f/x**2))
def bulkLNindex(x):
    return sellmeier(x/1000,2.9804,0.02047,0.5981,0.0666,8.9543,416.08) ##ZelmonSmallJundt97 LN ne
def waveguideLNindex(x):
    dnzln  = [0.0649733,0.0383078,0.0297231,0.0251477,0.0222051,0.0201054,0.0185018,0.0172158,0.0161455,0.0152282,0.0144231,0.0137025,0.013047,0.0124427,0.0118793,0.011349,0.0108461,0.0103661,0.00990548,0.00946157,0.00903228,0.00861598,0.00821138,0.0078175,0.00743356,0.00705898,0.00669333,0.00633627,0.0059876,0.00564716,0.00531488,0.00499074,0.00467476,0.00436699,0.00406754]
    wavelengths = [300+i*50 for i in range(len(dnzln))]
    return bulkLNindex(x) + np.interp(x,wavelengths,dnzln)
def lnwgindex(width,plot=False,ref=False,res=0.1,fit=False):
    from modes import modesolver
    from waves import Wave
    from sellmeier import index
    # md = modesolver(λ=1550,width=8,depth=1.9,ape=23.5,rpe=24.5,sell='ln',apetemp=320,rpetemp=300,res=0.2,limits=(-15,15,-20,2))
    args = dict(width=width,res=res)
    wx = np.linspace(400,2000,17)
    print('width',width,'args',args)
    n1 = Wave([modesolver(x,**args,sell='ln',depth=1.9,ape=23.5,rpe=24.5,apetemp=320,rpetemp=300,limits=(-15,15,-20,2)).neff - index(x,sell) for x in wx],wx,'n1')
    if plot:
        n0 = Wave(index(wx,'lnwg')-index(wx,'ln'),wx,'n0')
        Wave.plots(n0,n1,x='λ (nm)',y='index')
    # if fit:
    #     return Wave(n1.y,n1.x/1000).quadraticfit(coef=1)
    return n1.rename(f'{width}µm') if not ref else n0.rename('reference')

if __name__ == '__main__':
    print('bulk index at 1064nm:',bulkLNindex(1064))
    print('waveguide index at 1064nm:',waveguideLNindex(1064))
    lnwgindex(8,plot=1,res=0.2)


