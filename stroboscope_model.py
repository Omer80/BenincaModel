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

def find_timeseries_peaks(fname,resolution,Tmax_index):
    Tmax_array = np.linspace(17.5,25,100)
    Tmax = Tmax_array[Tmax_index]
    print("Calculating for Tmax=",Tmax)
    m = BenincaModel(Es=Es_normal,Ps='auto/Beninca_set1.hdf5',Vs=None)
    t,sol_const = m.ode_integrate([0.5,0.5,0.5,0.5])
    init_cond = sol_const[-1]
    int_finish=100*365
    trim=20*365
    ito=0.0000
    parameters={}
    parameters['Tmax']=Tmax
    m.update_parameters(parameters)
    def G(y, t):
        return np.array([[ito,0.0,0.0,0.0],[0.0,ito,0.0,0.0],[0.0,0.0,ito,0.0],[0.0,0.0,0.0,ito]])
    step=0.01
    tspan = np.arange(0.0, int_finish+step,step)
    forcing = -1.0*m.Ft(tspan)
    print("Integrating for initial conditions:",init_cond)
    result = sdeint.itoint(m.rhs_ode, G, init_cond, tspan)
    peaks_signal, _=find_peaks(result[trim:,0])
    peaks_forcing, _=find_peaks(forcing[trim:,0])
    Tmax_peaks_signal = np.ones_like(peaks_signal)*Tmax
    sfname = fname+str(Tmax_index)+".dat"
    np.savetxt(sfname,np.array([Tmax_peaks_signal,
                                result[trim:,0][peaks_signal],
                                result[trim:,-2][peaks_forcing]]).T)
    print("Saved to",sfname)
    
def main(args):
    find_timeseries_peaks(args.fname,args.resolution,args.Tmax_index)

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
    parser.add_argument('-i','--Tmax_index',
                        dest='Tmax_index',
                        type=int,
                        default=0,
                        help='Index to point at index at Tmax array')
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