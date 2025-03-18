import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import seaborn as sns

def Dipole(Q2, MA):
    fFA0 = -1.2670
    return -fFA0*(1+Q2/MA**2)**-2

def MINERvA(f):
    with open(f) as j:
      loadedjson = json.loads(j.read())
      df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in loadedjson.items() ]))
    return df.x.to_numpy(), df.y.to_numpy()*Dipole(df.x.to_numpy(), 1.014)

def CalculateZ(q2):
    fT0 = -0.28 # GeV^2
    fTcut = 0.1764 # GeV^2, == 9mpi**2
    znum  = np.sqrt(fTcut - q2) - np.sqrt(fTcut - fT0)
    zden  = np.sqrt(fTcut - q2) + np.sqrt(fTcut - fT0)
    return znum/zden

# Copied from GENIE
def CalculateAs(A1, A2, A3, A4):
    fKmax = 4
    fFA0 = -1.2670
    
    fZ_An = [0, A1, A2, A3, A4, 0, 0, 0, 0]
    
    kp4 = fKmax + 4
    kp3 = fKmax + 3
    kp2 = fKmax + 2
    kp1 = fKmax + 1
    kp0 = fKmax + 0
    
    z0   = CalculateZ(0.)
    zkp4 = np.power(z0,kp4)
    zkp3 = np.power(z0,kp3)
    zkp2 = np.power(z0,kp2)
    zkp1 = np.power(z0,kp1)
    
    denom = \
      6. -    kp4*kp3*kp2*zkp1 + 3.*kp4*kp3*kp1*zkp2 \
         - 3.*kp4*kp2*kp1*zkp3 +    kp3*kp2*kp1*zkp4
    
    b0  = 0.;
    for ki in range(1, fKmax+1):
        b0 = b0 + fZ_An[ki]
    
    
    b0z = -fFA0
    for ki in range(1, fKmax+1):
        b0z = b0z + fZ_An[ki]*np.power(z0, ki)
    
    b1  = 0.
    for ki in range(1, fKmax+1):
        b1 = b1 + ki*fZ_An[ki]
      
    b2  = 0.
    for ki in range(1, fKmax+1):
        b2 = b2 + ki*(ki-1)*fZ_An[ki]
    
    b3  = 0.
    for ki in range(1, fKmax+1):
        b3 = b3 + ki*(ki-1)*(ki-2)*fZ_An[ki]
 
    # Assign new parameters
    fZ_An[kp4] = (1./denom) *                           \
    ( (b0-b0z)*kp3*kp2*kp1                                   \
    + b3*( -1. + .5*kp3*kp2*zkp1 - kp3*kp1*zkp2              \
         +.5*kp2*kp1*zkp3                               )  \
    + b2*(  3.*kp1 - kp3*kp2*kp1*zkp1                        \
         +kp3*kp1*(2*fKmax+1)*zkp2 - kp2*kp1*kp0*zkp3   )  \
    + b1*( -3.*kp2*kp1 + .5*kp3*kp2*kp2*kp1*zkp1             \
         -kp3*kp2*kp1*kp0*zkp2 + .5*kp2*kp1*kp1*kp0*zkp3)  );
    
    fZ_An[kp3] = (1./denom) *                           \
    ( -3.*(b0-b0z)*kp4*kp2*kp1                               \
    + b3*(  3. - kp4*kp2*zkp1 + (3./2.)*kp4*kp1*zkp2         \
         -.5*kp2*kp1*zkp4                               )  \
    + b2*( -3.*(3*fKmax+4) + kp4*kp2*(2*fKmax+3)*zkp1        \
         -3.*kp4*kp1*kp1*zkp2 + kp2*kp1*kp0*zkp4        )  \
    + b1*(  3.*kp1*(3*fKmax+8) - kp4*kp3*kp2*kp1*zkp1        \
    +(3./2.)*kp4*kp3*kp1*kp0*zkp2 - .5*kp2*kp1*kp1*kp0*zkp4) )
    
    fZ_An[kp2] = (1./denom) *                           \
    ( 3.*(b0-b0z)*kp4*kp3*kp1                                \
    + b3*( -3. + .5*kp4*kp3*zkp1 - (3./2.)*kp4*kp1*zkp3      \
         +kp3*kp1*zkp4                                  )  \
    + b2*(  3.*(3*fKmax+5) - kp4*kp3*kp2*zkp1                \
         +3.*kp4*kp1*kp1*zkp3 - kp3*kp1*(2*fKmax+1)*zkp4)  \
    + b1*( -3.*kp3*(3*fKmax+4) + .5*kp4*kp3*kp3*kp2*zkp1     \
    -(3./2.)*kp4*kp3*kp1*kp0*zkp3 + kp3*kp2*kp1*kp0*zkp4)  )
    
    fZ_An[kp1] = (1./denom) *                           \
    ( -(b0-b0z)*kp4*kp3*kp2                                  \
    + b3*(  1. - .5*kp4*kp3*zkp2 + kp4*kp2*zkp3              \
         -.5*kp3*kp2*zkp4                               )  \
    + b2*( -3.*kp2 + kp4*kp3*kp2*zkp2                        \
         -kp4*kp2*(2*fKmax+3)*zkp3 + kp3*kp2*kp1*zkp4)     \
    + b1*(  3.*kp3*kp2 - .5*kp4*kp3*kp3*kp2*zkp2             \
         +kp4*kp3*kp2*kp1*zkp3 - .5*kp3*kp2*kp2*kp1*zkp4)  )
    
    fZ_An[0] = (1./denom) *                                  \
    ( -6.*b0z                                                \
    + b0*(  kp4*kp3*kp2*zkp1 - 3.*kp4*kp3*kp1*zkp2           \
         +3.*kp4*kp2*kp1*zkp3 - kp3*kp2*kp1*zkp4        )  \
    + b3*( -zkp1 + 3.*zkp2 - 3.*zkp3 + zkp4               )  \
    + b2*(  3.*kp2*zkp1 - 3.*(3*fKmax+5)*zkp2                \
         +3.*(3*fKmax+4)*zkp3 - 3.*kp1*zkp4             )  \
    + b1*( -3.*kp3*kp2*zkp1 + 3.*kp3*(3*fKmax+4)*zkp2        \
         -3.*kp1*(3*fKmax+8)*zkp3 + 3.*kp2*kp1*zkp4     )  )
    return fZ_An

def poly(Q2, c0, c1, c2, c3):
    Cs = [c0, c1, c2, c3]
    z = CalculateZ(-Q2)
    ret = 0
    for ki in range(len(Cs)):
        ret += Cs[ki] * z**ki
    return ret

def FFzexp(Q2, a1, a2, a3, a4):
    z = CalculateZ(-Q2)
    rets = []
    As = CalculateAs(a1, a2, a3, a4)
    ret = 0
    for ki in range(len(As)):
        ret += As[ki] * z**ki
    return -ret

def FFzexp_partial(Q2, i, a1, a2, a3, a4):
    # note: i index will be off from a subscript bc a0 determined by sum rules
    delta = 0.001
    FitAs = np.array([a1, a2, a3, a4])
    FitAsPlsDelta = FitAs.copy()
    FitAsPlsDelta[i] = FitAsPlsDelta[i] + delta
    ret = [(FFzexp(q2, *FitAsPlsDelta) - FFzexp(q2, *FitAs))/delta for q2 in Q2]
    return np.array(ret)

def FFzexpErr(Q2, cov, fit_vals):
    As = CalculateAs(*fit_vals)
    z = CalculateZ(-Q2)
    var = 0
    for i in range(len(cov[0])):
        for j in range(len(cov[0])):
            var += FFzexp_partial(Q2, i, As[1], As[2], As[3], As[4])*FFzexp_partial(Q2, j, As[1], As[2], As[3], As[4])*cov[i][j]
    return np.sqrt(var)

# Extracts a CV and errorbars from data thief output
def ExtractResults(file, xvals):
    with open(file) as j:
        print(file)
        loadedjson = json.loads(j.read())
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in loadedjson.items() ]))
        if("minervacv" in file):
            # minerva paper (https://www.nature.com/articles/s41586-022-05478-3) 
            # special case, just fit cv from the paper to polynomial.
            minervax, minervay = MINERvA(file)
            poptcv, pcovcv = curve_fit(FFzexp, minervax, minervay)
        else:
            # Most data is from https://arxiv.org/pdf/2210.02455. Find ends of
            # the error bars, fit to polynomial, and then take the mid point. 
            # Fit the midpoint to zexp.
            popthigh, pcovhigh = curve_fit(poly, df.highx.dropna(), df.highy.dropna())
            poptlow, pcovlow = curve_fit(poly, df.lowx.dropna(), df.lowy.dropna())
            highy = poly(xvals, *popthigh)
            lowy = poly(xvals, *poptlow)
            poptcv, pcovcv = curve_fit(FFzexp, xvals, (highy+lowy)/2, sigma=(highy-lowy)/2, absolute_sigma=True)

    return FFzexp(xvals, *poptcv)

def main(output, inputs):
    xvals = np.linspace(0, 1.0, 31)
    all_ydata = []
    all_xdata = []

    for single_input in inputs:
        CV = ExtractResults(single_input, xvals)
        plt.plot(xvals, CV, linestyle='none', marker='o', label=single_input.split(".")[0])
        all_xdata.append(xvals)
        # force constraint for FA(0)
        CV[0]=1.2723
        all_ydata.append(CV)

    avg_ydata = np.sum(all_ydata, axis=0)/len(inputs)
    avg_ydataerr = np.std(all_ydata, axis=0)

    # force error for constraint for FA(0)
    avg_ydataerr[0] = 0.0023

    plt.errorbar(xvals, avg_ydata, yerr=avg_ydataerr, label="Averaged Data", marker='o')
    glob_popt, glob_pcov = curve_fit(FFzexp, xvals, avg_ydata, sigma=avg_ydataerr, absolute_sigma=True)

    param_str = "Fit Result:\n"
    for i in range(len(glob_popt)):
        param_str += r"$a_"+str(i+1)+" = "+str(round(glob_popt[i], 3))+" \pm "+str(round(np.sqrt(np.diag(glob_pcov))[i], 3))+"$\n"
    plt.text(0.0, 0.4, param_str)
    plt.xlabel(r"$Q^2 (GeV^2)$ ")
    plt.ylabel(r"$F_{A}(Q^2)$ ")
    plt.plot(xvals, FFzexp(xvals, *glob_popt), label="Fit")
    ErrorMag = FFzexpErr(xvals, glob_pcov, glob_popt)
    plt.fill_between(xvals, FFzexp(xvals, *glob_popt) - ErrorMag, FFzexp(xvals, *glob_popt) + ErrorMag, alpha=0.5)
    plt.legend()
    plt.savefig(output) 
    plt.clf()

    coeffs = ['a1', 'a2', 'a3', 'a4']
    sns.heatmap(glob_pcov, annot=True, fmt='g', xticklabels=coeffs, yticklabels=coeffs)
    plt.savefig('cov.png')
    plt.clf()

if __name__ == "__main__":
    printhelp = len(sys.argv) < 3 or sys.argv[1] == "-h"
    if printhelp:
        print("Usage: python zexpfitter.py [output.png] [inputs.json,]")
    else:
        main(sys.argv[1], sys.argv[2:])
