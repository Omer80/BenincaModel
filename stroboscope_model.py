#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:11:30 2018

@author: ohm
"""

import numpy as np
import sdeint
from scipy.signal import find_peaks
from BenincaModel import BenincaModel,Es_normal
import deepdish.io as dd
Ps_normal='auto/Beninca_set3.hdf5'

def calc_for_constant(m):
    t,sol_const = m.ode_integrate([0.1,0.1,0.1,0.1])
    return sol_const[-1]
def calc_for_oscillation_with_Ito(m,init_cond,alpha,Tmax,ito,int_finish,step):
    tspan = np.arange(0.0, int_finish+step,step)
    forcing = -1.0*m.Ft(tspan)
    parameters={}
    parameters['Tmax']=Tmax
    parameters['alpha']=alpha
    m.update_parameters(parameters)
    def G(y, t):
        return np.array([[ito,0.0,0.0,0.0],[0.0,ito,0.0,0.0],[0.0,0.0,ito,0.0],[0.0,0.0,0.0,ito]])
    result = sdeint.itoint(m.rhs_ode, G, init_cond, tspan)
    return tspan,result,forcing

def find_timeseries_peaks(fname,var_index,ito,resolution,Tmax_index):
    Tmax_array = np.linspace(17.5,25,100)
    Tmax = Tmax_array[Tmax_index-1]
    print("Calculating for Tmax=",Tmax)
    m = BenincaModel(Es=Es_normal,Ps=Ps_normal,Vs=None)
    init_cond = calc_for_constant(m)
    int_finish=100*365
    trim=20*365
    step=0.01
    print("Integrating for initial conditions:",init_cond)
    _,result,forcing=calc_for_oscillation_with_Ito(m,init_cond,Tmax,ito,int_finish,step)
    peaks_signal, _=find_peaks(result[trim:,var_index])
    peaks_forcing, _=find_peaks(forcing[trim:])
    Tmax_peaks_signal = np.ones_like(peaks_signal)*Tmax
    sfname = fname+"var_"+str(var_index)+"_"+str(Tmax_index)+".dat"
    data = np.array([Tmax_peaks_signal,
                     result[trim:,var_index][peaks_signal],
                     result[trim:,var_index][peaks_forcing]]).T
    return sfname,data

def save_find_timeseries_peaks(fname,var_index,ito,resolution,Tmax_index):
    sfname,data=find_timeseries_peaks(fname,var_index,ito,resolution,Tmax_index)
    np.savetxt(sfname,data)
    print("Saved to",sfname)

def find_min_max(tspan,result,trim):
    series = result[trim:]
    t_plot = tspan[trim:]
    series_dt = np.gradient(series)
    series_ddt = np.gradient(series_dt)
    zero_crossings_idx = np.where(np.diff(np.sign(series_dt)))[0]
    zero_crossings=series[zero_crossings_idx]
    series_ddt_zero_crossings=series_ddt[zero_crossings_idx]
    maximas = zero_crossings[np.where(series_ddt_zero_crossings<0)[0]]
    minimas = zero_crossings[np.where(series_ddt_zero_crossings>=0)[0]]
    return t_plot[zero_crossings_idx]/365,minimas,maximas

def bif_B_min_max_to_Tmax(Tmax_max,ito,resolution,fname,max_samples=20):
    step=0.1
    int_finish=int(60*365)
    trim=int(40*365)
    #finish = int(int_finish*1.0)
    m = BenincaModel(Es=Es_normal,Ps=Ps_normal,Vs=None)
    Tmax_array = np.linspace(m.p['Tmean'],Tmax_max,resolution)
    init_cond = calc_for_constant(m)
    barnicles_arr=[init_cond[0]*100.0+init_cond[1]*100.0]
    crustose_arr=[init_cond[2]*100.0]
    mussels_arr=[init_cond[3]*100.0]
    mean_B = np.zeros_like(Tmax_array)
    mean_A = np.zeros_like(Tmax_array)
    mean_M = np.zeros_like(Tmax_array)
    mean_B[0]=np.mean(init_cond[0]*100.0+init_cond[1]*100.0)
    mean_A[0]=np.mean(init_cond[2]*100.0)
    mean_M[0]=np.mean(init_cond[3]*100.0)
#    t_plot_arr=[]
    alpha=1.0
    for i,Tmax in enumerate(Tmax_array[1:]):
        print("Calculating for Tmax=",Tmax)
        tspan,result,forcing=calc_for_oscillation_with_Ito(m,init_cond,alpha,Tmax,ito,int_finish,step)
        barnicles=(result[:,0]+result[:,1])[trim:]*100.0
        crustose=(result[:,2])[trim:]*100.0
        mussels=(result[:,3])[trim:]*100.0
        mean_B[i+1]=np.mean(barnicles)
        mean_A[i+1]=np.mean(crustose)
        mean_M[i+1]=np.mean(mussels)
        t_plot,barnicles_minimas,barnicles=find_min_max(tspan,barnicles,trim)
        t_plot,crustose_minimas,crustose=find_min_max(tspan,crustose,trim)
        t_plot,mussels_minimas,mussels=find_min_max(tspan,mussels,trim)
        if len(barnicles)>max_samples:
            barnicles=np.random.choice(barnicles,max_samples)
            crustose=np.random.choice(crustose,max_samples)
            mussels=np.random.choice(mussels,max_samples)
        barnicles_arr.append(barnicles)
        crustose_arr.append(crustose)
        mussels_arr.append(mussels)
#        t_plot_arr.append(t_plot)
    data = {"Tmax":Tmax_array,
            "B":barnicles_arr,
            "C":crustose_arr,
            "M":mussels_arr,
            "B_mean":mean_B,"A_mean":mean_A,"M_mean":mean_M}
    dd.save(fname+".hdf5",data)

def bif_B_min_max_to_alpha(Tmax,ito,resolution,fname,max_samples=20):
    step=0.1
    int_finish=int(60*365)
    trim=int(40*365)
    #finish = int(int_finish*1.0)
    Ps=dd.load(Ps_normal)
    Ps['Tmax']=Tmax
    Ps['alpha']=0.0
    m = BenincaModel(Es=Es_normal,Ps=Ps,Vs=None)
    alpha_array = np.linspace(0,1.0,resolution)
    init_cond = calc_for_constant(m)
    barnicles_arr=[init_cond[0]*100.0+init_cond[1]*100.0]
    crustose_arr=[init_cond[2]*100.0]
    mussels_arr=[init_cond[3]*100.0]
    mean_B = np.zeros_like(alpha_array)
    mean_A = np.zeros_like(alpha_array)
    mean_M = np.zeros_like(alpha_array)
    mean_B[0]=np.mean(init_cond[0]*100.0+init_cond[1]*100.0)
    mean_A[0]=np.mean(init_cond[2]*100.0)
    mean_M[0]=np.mean(init_cond[3]*100.0)
#    t_plot_arr=[]
    for i,alpha in enumerate(alpha_array[1:]):
        print("Calculating for alpha=",alpha)
        tspan,result,forcing=calc_for_oscillation_with_Ito(m,init_cond,alpha,Tmax,ito,int_finish,step)
        barnicles=(result[:,0]+result[:,1])[trim:]*100.0
        crustose=(result[:,2])[trim:]*100.0
        mussels=(result[:,3])[trim:]*100.0
        mean_B[i+1]=np.mean(barnicles)
        mean_A[i+1]=np.mean(crustose)
        mean_M[i+1]=np.mean(mussels)
        t_plot,barnicles_minimas,barnicles=find_min_max(tspan,barnicles,trim)
        t_plot,crustose_minimas,crustose=find_min_max(tspan,crustose,trim)
        t_plot,mussels_minimas,mussels=find_min_max(tspan,mussels,trim)
        if len(barnicles)>max_samples:
            barnicles=np.random.choice(barnicles,max_samples)
            crustose=np.random.choice(crustose,max_samples)
            mussels=np.random.choice(mussels,max_samples)
        barnicles_arr.append(barnicles)
        crustose_arr.append(crustose)
        mussels_arr.append(mussels)
#        t_plot_arr.append(t_plot)
    data = {"alpha":alpha_array,
            "B":barnicles_arr,
            "C":crustose_arr,
            "M":mussels_arr,
            "B_mean":mean_B,"A_mean":mean_A,"M_mean":mean_M}
    dd.save(fname+".hdf5",data)


def integrate_from_steady_state(alpha,Tmax,ito,idx_finish,step,version):
    Ps=dd.load(Ps_normal)
    Ps['Tmax']=Tmax
    Ps['alpha']=0.0
    Es = Es_normal.copy()
    Es['rhs']=version
    m = BenincaModel(Es=Es,Ps=Ps,Vs=None)
    init_cond = calc_for_constant(m)
    print("Initial condition:",init_cond)
    print("Integrating with SDEINT")
    tspan,result,forcing=calc_for_oscillation_with_Ito(m,init_cond,alpha,Tmax,ito,idx_finish,step)
    forcing_tspan = m.Ft(tspan)
    return tspan,result,forcing,forcing_tspan

def plot_ito_integration(Tmax,alpha,ito,max_time,trim,step,figsize,version):
    import matplotlib.pyplot as plt
    idx_finish=int(max_time*365)
    tspan,result,forcing,forcing_tspan=integrate_from_steady_state(alpha,Tmax,ito,idx_finish,step,version)
#    idx_trim=int(trim*365)
    #finish = int(int_finish*1.0)
    fig,ax=plt.subplots(1,2,figsize=(figsize*1.618,figsize),constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax3 = plt.subplot(gs[2:,:-1])
    ax1 = plt.subplot(gs[0, :-1])#, sharey=ax3, sharex=ax3)
    ax2 = plt.subplot(gs[1, :-1])#, sharey=ax3, sharex=ax3)
    ax4 = plt.subplot(gs[:, -1])
    plot_forcing_tspan = 10.0*forcing_tspan/np.amax(forcing_tspan)
    ax1.plot(tspan/365-trim,(result[:,0]+result[:,1])*100.0,'b',label=r'Barnacles')
    ax2.plot(tspan/365-trim,result[:,2]*100.0,'g',label=r'Algae')
    ax3.plot(tspan/365-trim,result[:,3]*100.0,'r',label=r'Mussels')
    ax1.plot(tspan/365-trim,plot_forcing_tspan,'m:',label=r'forcing',lw=1)
    ax2.plot(tspan/365-trim,plot_forcing_tspan,'m:',label=r'forcing',lw=1)
    ax3.plot(tspan/365-trim,plot_forcing_tspan,'m:',label=r'forcing',lw=1)
    ax1.set_ylabel(r'Barnacles $[\%]$')
    ax1.axes.xaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticklabels([])
    #ax3.axes.xaxis.set_ticklabels(np.arange(30,51,1))
    ax2.set_ylabel(r'Algae $[\%]$')
    ax3.set_ylabel(r'Mussels $[\%]$')
    ax3.set_xlabel(r'Time $[years]$')
    ax1.set_xlim([0,trim])
    ax2.set_xlim([0,trim])
    ax3.set_xlim([0,trim])
    ax1.set_ylim([-10,110])
    ax2.set_ylim([-10,110])
    ax3.set_ylim([-10,110])
    # FFT
#    forcing = m.Ft(tspan)
    #print(len(forcing))
    trimfft = int(len(forcing)*(2.0/5.0)) # the index from which to trim the time series to clean transients
    #print(trim)
    frq = np.fft.fftfreq(forcing[-trimfft:].size,d=0.01/365)
#    fft_forcing = np.absolute((np.fft.fft(forcing[-trimfft:])))
    fft_signal_B  = np.absolute((np.fft.fft(result[-trimfft:,0]+result[-trimfft:,1])))
    fft_signal_M  = np.absolute((np.fft.fft(result[-trimfft:,-1])))
    normalize_fft = np.amax(fft_signal_B[1:])
#    ax4.plot(frq[1:],fft_forcing[1:]/np.amax(fft_forcing[1:]),'m:',label=r'forcing')
    ax4.plot(frq[1:],fft_signal_B[1:]/normalize_fft,'b',label=r'Barnacles')
    ax4.plot(frq[1:],fft_signal_M[1:]/normalize_fft,'r',label=r'Mussels')
    ax4.set_xlim([0.1,2.0])
    ax4.set_ylim([-0.01,1.2])
    ax4.set_xlabel(r'freq $[1/years]$')
    #ax[0].legend(loc='upper left')
    ax4.legend(loc='upper right')
    #ax[0].legend(loc='upper left')
    #ax[1].legend(loc='upper left')
    base_fname='results/Beninca_Ito_and_fft_Tmax{:3.2f}_ito{:5.4f}'.format(Tmax,ito).replace('.','_')
    plt.savefig(base_fname+'.pdf')
    plt.savefig(base_fname+'.png')

def main(args):
    if args.find_peaks:
        save_find_timeseries_peaks(args.fname,args.var_index,args.ito,
                              args.resolution,args.Tmax_index)
    elif args.bif_B_min_max_to_Tmax:
        bif_B_min_max_to_Tmax(args.Tmax_max,args.ito,
                              args.resolution,args.fname)
    elif args.bif_B_min_max_to_alpha:
        bif_B_min_max_to_alpha(args.Tmax,args.ito,
                               args.resolution,args.fname)
    elif args.plot_ito_integration:
        plot_ito_integration(args.Tmax,args.alpha,args.ito,
                             args.max_time,args.trim,
                             args.step,args.figsize,args.model_version)


def add_parser_arguments(parser):
    parser.add_argument("-f", "--fname",
                        type=str, nargs='?',
                        dest="fname",
                        default="Beninca_stroboscope_i_",
                        help="Save p_transition to chi in fname")
    parser.add_argument("--model_version",
                        type=str, nargs='?',
                        dest="model_version",
                        default="Beninca_forced",
                        help="Beninca model version")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        dest="verbose",
                        default=False,
                        help="Turn on debuging messages")
    parser.add_argument("--plot_ito_integration",
                        action="store_true",
                        dest="plot_ito_integration",
                        default=False,
                        help="Integrate the system with a given ito stochastic strength and plot")
    parser.add_argument("--find_peaks",
                        action="store_true",
                        dest="find_peaks",
                        default=False,
                        help="Start find_peaks function")
    parser.add_argument("--bif_B_min_max_to_Tmax",
                        action="store_true",
                        dest="bif_B_min_max_to_Tmax",
                        default=False,
                        help="Start bifurcation for B to Tmax")
    parser.add_argument("--bif_B_min_max_to_alpha",
                        action="store_true",
                        dest="bif_B_min_max_to_alpha",
                        default=False,
                        help="Start bifurcation for B to alpha")
    parser.add_argument('-i','--Tmax_index',
                        dest='Tmax_index',
                        type=int,
                        default=1,
                        help='Index to point at index at Tmax array - minimum 1')
    parser.add_argument('--var_index',
                        dest='var_index',
                        type=int,
                        default=0,
                        help='Index of variable from 0 to 3')
    parser.add_argument('--ito',
                        dest='ito',
                        type=float,
                        default=0.0,
                        help='Ito diagonal noise')
    parser.add_argument('--max_time',
                        dest='max_time',
                        type=float,
                        default=50.0,
                        help='Time for integration')
    parser.add_argument('--trim',
                        dest='trim',
                        type=float,
                        default=20.0,
                        help='Years to trim at the end of the time series')
    parser.add_argument('--step',
                        dest='step',
                        type=float,
                        default=0.01,
                        help='Step size for integration')
    parser.add_argument('--figsize',
                        dest='figsize',
                        type=float,
                        default=6.0,
                        help='figsize for ploting')
    parser.add_argument('--Tmax_max',
                        dest='Tmax_max',
                        type=float,
                        default=25.0,
                        help='Maximal Tmax in the bifurcation')
    parser.add_argument('--Tmax',
                        dest='Tmax',
                        type=float,
                        default=20.5,
                        help='Tmax in the alpha bifurcation')
    parser.add_argument('--alpha',
                        dest='alpha',
                        type=float,
                        default=1.0,
                        help='alpha parameter')
    parser.add_argument('--resolution',
                        dest='resolution',
                        type=int,
                        default=100,
                        help='Resolution of Tmax array')
    return parser

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='PROG', usage='%(prog)s [options]')
    parser = add_parser_arguments(parser)
    main(parser.parse_args())