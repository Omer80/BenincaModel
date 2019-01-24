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

def calc_for_constant(m):
    t,sol_const = m.ode_integrate([0.5,0.5,0.5,0.5])
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
    m = BenincaModel(Es=Es_normal,Ps='auto/Beninca_set1.hdf5',Vs=None)
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

def find_min_max(tspan,result,forcing,trim):
    barnicles=(result[:,0]+result[:,1])[trim:]*100.0
    t_plot = tspan[trim:]
    barnicles_dt = np.gradient(barnicles)
    zero_crossings = np.where(np.diff(np.sign(barnicles_dt)))[0]
    return t_plot[zero_crossings]/365,barnicles[zero_crossings]

def bif_B_min_max_to_Tmax(Tmax_max,ito,resolution,fname,max_samples=10):
    step=0.1
    int_finish=int(60*365)
    trim=int(40*365)
    #finish = int(int_finish*1.0)
    m = BenincaModel(Es=Es_normal,Ps='auto/Beninca_set1.hdf5',Vs=None)
    Tmax_array = np.linspace(m.p['Tmean'],Tmax_max,resolution)
    init_cond = calc_for_constant(m)
    barnicles_arr=[]
#    t_plot_arr=[]
    alpha=1.0
    for i,Tmax in enumerate(Tmax_array):
        print("Calculating for Tmax=",Tmax)
        tspan,result,forcing=calc_for_oscillation_with_Ito(m,init_cond,alpha,Tmax,ito,int_finish,step)
        t_plot,barnicles=find_min_max(tspan,result,forcing,trim)
        if len(barnicles)>max_samples:
            barnicles=np.random.choice(barnicles,max_samples)
        barnicles_arr.append(barnicles)
#        t_plot_arr.append(t_plot)
    data = {"Tmax":Tmax_array,"B":barnicles_arr}
    dd.save(fname+".hdf5",data)
    
def bif_B_min_max_to_alpha(Tmax,ito,resolution,fname,max_samples=10):
    step=0.1
    int_finish=int(60*365)
    trim=int(40*365)
    #finish = int(int_finish*1.0)
    Ps=dd.load('auto/Beninca_set1.hdf5')
    Ps['Tmax']=Tmax
    Ps['alpha']=0.0
    m = BenincaModel(Es=Es_normal,Ps=Ps,Vs=None)
    alpha_array = np.linspace(0,1.0,resolution)
    init_cond = calc_for_constant(m)
    barnicles_arr=[]
#    t_plot_arr=[]
    for i,alpha in enumerate(alpha_array):
        print("Calculating for alpha=",alpha)
        tspan,result,forcing=calc_for_oscillation_with_Ito(m,init_cond,alpha,Tmax,ito,int_finish,step)
        t_plot,barnicles=find_min_max(tspan,result,forcing,trim)
        if len(barnicles)>max_samples:
            barnicles=np.random.choice(barnicles,max_samples)
        barnicles_arr.append(barnicles)
#        t_plot_arr.append(t_plot)
    data = {"alpha":alpha_array,"B":barnicles_arr}
    dd.save(fname+".hdf5",data)

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

def add_parser_arguments(parser):
    parser.add_argument("-f", "--fname",
                        type=str, nargs='?',
                        dest="fname",
                        default="Beninca_stroboscope_i_",
                        help="Save p_transition to chi in fname")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        dest="verbose",
                        default=False,
                        help="Turn on debuging messages")
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