#from numpy import pi
dimpar = {'mB'   : 0.003,
          'muB'  : 0.015,
          'cBR'  : 0.018,
          'mA'   : 0.013,
          'muAR' : 0.008,
          'muAB' : 0.036,
          'cAR'  : 0.021,
          'cAB'  : 0.049,
          'mM'   : 0.017,
          'muM'  : 0.061,
          'cM'   : 0.078,
          'alpha': 1.0,
          'Tmean': 17.1,
          'Tmax' : 17.1,#20.5 from the paper
          }

def update_par():
    par=dimpar.copy()
    return par


def savecsvdimpar(fname,dimpar):
    import csv
    with open(fname+'.csv', 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Parameter','value'])
        writer.writeheader()
        for k in dimpar:
            writer.writerow({'Parameter': k , 'value': dimpar[k]})
    
def saveParmSet(fname,par,text=None):
    from deepdish.io import save
    if text is not None:
        par['text']=text
    save("./auto/"+fname+".hdf5",par)
def loadParmSet(fname):
    from deepdish.io import load
    return load(fname)

if __name__ == '__main__':
    par=update_par()
    print("Nondimensional:",)
    print(par)
#    print "conv P=",par['conv_P']
    saveParmSet('Beninca_set2',par)
#    import numpy as np
#    p=par
#    a = np.array([p['lamb_max'],p['lamb_min'],p['eta'],
#                  p['p'],p['nu'],p['rho'],p['kappa'],
#                  p['c'],p['zeta'],p['gamma'],
#                  p['s_wp'],p['s_fos'],p['s_fc'],
#                  p['s_h'],p['mu_s_max'],
#                  p['chi'],p['beta'],p['a'],
#                  p['omegaf'],p['delta_s']])
#    np.savetxt("./auto/tlm_parameters.txt",a.T)
##    savemat("./p2p/b2s2_parameters.mat",p)
    
# tlm_set23 is modification of set18 to try to push the bifurcations to lower 
# mm values