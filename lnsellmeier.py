import numpy as np
def sellmeier(x,a,b,c,d,e,f):
    return np.sqrt(1+a/(1-b/x**2)+c/(1-d/x**2)+e/(1-f/x**2))
def bulkLNindex(x):
    return sellmeier(x/1000,2.9804,0.02047,0.5981,0.0666,8.9543,416.08) ##ZelmonSmallJundt97 LN ne
def waveguideLNindex(x):
    dnzln  = [0.0649733,0.0383078,0.0297231,0.0251477,0.0222051,0.0201054,0.0185018,0.0172158,0.0161455,0.0152282,0.0144231,0.0137025,0.013047,0.0124427,0.0118793,0.011349,0.0108461,0.0103661,0.00990548,0.00946157,0.00903228,0.00861598,0.00821138,0.0078175,0.00743356,0.00705898,0.00669333,0.00633627,0.0059876,0.00564716,0.00531488,0.00499074,0.00467476,0.00436699,0.00406754]
    wavelengths = [300+i*50 for i in range(len(dnzln))]
    return bulkLNindex(x) + np.interp(x,wavelengths,dnzln)
if __name__ == '__main__':
    print('bulk index at 1064nm:',bulkLNindex(1064))
    print('waveguide index at 1064nm:',waveguideLNindex(1064))
