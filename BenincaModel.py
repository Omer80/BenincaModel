# -*- coding: utf-8 -*-
"""
#  BeninceModel.py
#
#  Beninca Model - DOI: 10.1073/pnas.1421968112
#
#  Copyright 2016 Omer Tzuk <omertz@post.bgu.ac.il>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
"""
__version__= 1.0
__author__ = """Omer Tzuk (omertz@post.bgu.ac.il)"""
import time
from sympy import symbols, Matrix,lambdify
#from sympy.utilities.autowrap import ufuncify
import numpy as np
#import scipy.linalg as linalg
from scipy.integrate import solve_ivp
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.optimize import root as root_ode
from scipy.fftpack import fftn, ifftn
#from scipy.stats import norm,chi,pearson3
#import scipy.sparse as sparse
#from utilities import handle_netcdf as hn
import deepdish.io as dd

Es_normal={'rhs':"Beninca_forced",
        'n':(1024,),
        'l':(256.0,),
        'bc':"neumann",
        'scipy_method':'RK45',
        #'scipy_method':'BDF',
        #'stochastic':("norm_around_p",0.03,100),
        'it':"scipy",
        'dt':0.001,
        'analyze':False,
        'verbose':True,
        'setPDE':False}

def main():
    global m,p
    m = BenincaModel(Es=Es_normal,Ps='auto/Beninca_set2.hdf5',Vs=None)
    return 0

class BenincaModel(object):
    def __init__(self,Ps,Es,Vs=None):
        if type(Ps)==str:
            self.Psfname=Ps
            self.p=dd.load(Ps)
        else:
            self.Psfname=None
            self.p = Ps
        self.setup=Es
        self.setup['nvar']=4
#        self.Vs=Vs
        self.verbose=Es['verbose']
        if self.verbose:
            start=time.time()
        self.set_equations()
        self.dt = Es['dt']
        self.time_elapsed = 0
        if self.setup['setPDE']:
            self.rhs=self.rhs_pde        
            self.p['nd']=len(Es['n'])
            if self.p['nd']==2:
                self.p['nx'],self.p['ny']=Es['n']
                self.p['lx'],self.p['ly']=Es['l']
                self.l=[self.p['lx'],self.p['ly']]
                self.n=[self.p['nx'],self.p['ny']]
                self.dg  = tuple([l/float(n) for l,n in zip(self.l,self.n)])
                self.dx  = self.dg[0]
            elif self.p['nd']==1:
                self.dg=[Es['l'][0]/float(Es['n'][0])]
                self.dx=self.dg[0]
            self.dx2 = self.dx**2
            self.dt=Es['dt']*self.dx2 / self.p['delta_s']
            self.X = np.linspace(0,Es['l'][0],Es['n'][0])
            from utilities.laplacian_sparse import create_laplacian #,create_gradient
            self.lapmat=create_laplacian(self.setup['n'],self.setup['l'], self.setup['bc'] , [1.0,self.p['delta_s'],self.p['delta_s']],verbose=self.verbose)
#            self.gradmat=create_gradient(self.setup['n'],self.setup['l'], self.setup['bc'] , [1.0,self.p['Dw'],self.p['Dh']])
            if self.verbose:
                print("Laplacian created")
        else:
            self.rhs=self.rhs_ode
        self.set_integrator()
        if Vs is not None:
            self.setup_initial_condition(Vs)
        if self.verbose:
            print("Time to setup: ",time.time()-start)
    """ Setting up model equations """
    def set_equations(self):
        B0,BA,A,M,t = symbols('B0 BA A M t')
        self.var_symbols = {'B0':B0,'BA':BA,'A':A,'M':M,'t':t}
        self.Ps_symbols={}
        for key in list(self.p.keys()):
            self.Ps_symbols[key] = symbols(key)
        p=self.Ps_symbols
        if self.setup['rhs']=="Beninca_forced":
            """ Klausmeier Model """
            from sympy.functions import cos as symcos
            Ft=(1.0+p['alpha']*(p['Tmax']-p['Tmean'])*symcos(2.0*np.pi*(t-32.0)/365.0))
            R = 1.0 - B0 - A - M
            self.dB0dt_eq = p['cBR']*(B0+BA)*R-p['cAB']*A*B0-p['cM']*M*B0-p['mB']*B0+Ft*p['mA']*BA
            self.dBAdt_eq = p['cAB']*A*B0-p['cM']*M*BA-p['mB']*BA-Ft*p['mA']*BA
            self.dAdt_eq  = p['cAR']*A*R+p['cAB']*A*B0-p['cM']*M*A-Ft*p['mA']*A
            self.dMdt_eq  = p['cM']*M*(B0+A)-Ft*p['mM']*M
            self.Ft_sym=Ft
        """ Creating numpy functions """
        self.forcing = lambdify((t,p['alpha'],p['Tmax'],p['Tmean']),self.Ft_sym)
        symeqs = Matrix([self.dB0dt_eq,self.dBAdt_eq,self.dAdt_eq,self.dMdt_eq])
        self.ode  = lambdify((B0,BA,A,M,t,p['alpha'],p['Tmax']),self.sub_parms(symeqs))
        localJac   = symeqs.jacobian(Matrix([B0,BA,A,M]))
        self.sym_localJac = localJac
        self.localJac = lambdify((B0,BA,A,M,t,p['alpha'],p['Tmax']),self.sub_parms(localJac),dummify=False)
        if self.verbose:
            self.print_equations()
            print("Local Jacobian:" ,localJac)

    """ Printing and parameters related functions """
    def print_parameters(self):
        print(self.p)
    def print_equations(self,numeric=False):
        if numeric:
            print("dB0dt = ", self.sub_all_parms(self.dB0dt_eq))
            print("dBAdt = ", self.sub_all_parms(self.dBAdt_eq))
            print("dAdt  = ", self.sub_all_parms(self.dAdt_eq))
            print("dMdt  = ", self.sub_all_parms(self.dMdt_eq))
        else:
            print("dB0dt = ", self.dB0dt_eq)
            print("dBAdt = ", self.dBAdt_eq)
            print("dAdt  = ", self.dAdt_eq)
            print("dMdt  = ", self.dMdt_eq)
    def print_latex_equations(self):
        from sympy import latex
        print("\partial_t B0 = ",latex(self.dB0dt_eq))
        print("\partial_t BA = ",latex(self.dBAdt_eq))
        print("\partial_t A  = ",latex(self.dAdt_eq))
        print("\partial_t M  = ",latex(self.dMdt_eq))
    """ Functions for use with scipy methods """
    """ Utilities """
    def Ft(self,t,**kwargs):
        if kwargs:
            self.update_parameters(kwargs)
        return self.forcing(t,self.p['alpha'],self.p['Tmax'],self.p['Tmean'])
    def sub_parms(self,eqs):
        B0,BA,A,M,t = symbols('B0 BA A M t')
        for key in list(self.p.keys()):
#            print key 
            if key!='alpha' and key!='Tmax':
                eqs=eqs.subs(self.Ps_symbols[key],self.p[key])
        return eqs
    def sub_all_parms(self,eqs):
        B0,BA,A,M,t = symbols('B0 BA A M t')
        for key in list(self.p.keys()):
            eqs=eqs.subs(self.Ps_symbols[key],self.p[key])
        return eqs

    """ Spatial functions """
    def set_integrator(self):
        if 'stochastic' not in list(self.setup.keys()):
            integrator_type = {}
            integrator_type['euler'] = self.euler_integrate
            integrator_type['scipy'] = self.ode_integrate
            integrator_type['rk4'] = self.rk4_integrate
            integrator_type['pseudo_spectral'] = self.pseudo_spectral_integrate
            try:
                self.integrator = integrator_type[self.setup['it']]
            except KeyError:
                raise  ValueError("No such integrator : %s."%self.setup['it'])
            if self.setup['it']=='pseudo_spectral':
                self.dt*=100.0
        elif 'stochastic' in list(self.setup.keys()):
#            self.setup_stochastic_rainfall()
            integrator_type = {}
            integrator_type['euler'] = self.euler_integrate_stochastic
            integrator_type['rk4'] = self.rk4_integrate_stochastic
            integrator_type['scipy'] = self.scipy_integrate_stochastic
            try:
                self.integrator = integrator_type[self.setup['it']]
            except KeyError:
                raise  ValueError("No such integrator : %s."%self.setup['it'])

    def rhs_pde(self,state,t=0):
        B0,BA,A,M=np.split(state,3)
        return np.ravel((self.dbdt(B0,BA,A,M,t,self.p['alpha']),
                         self.dwdt(B0,BA,A,M,t,self.p['alpha']))) + self.lapmat*state

    def rhs_ode(self,state,t=0):
        B0,BA,A,M=state
        return self.ode(B0,BA,A,M,t,self.p['alpha'],self.p['Tmax']).T[0]
    def scipy_ode_rhs(self,t,state):
        B0,BA,A,M=state
        return np.squeeze(self.ode(B0,BA,A,M,t,self.p['alpha'],self.p['Tmax']))
    def scipy_ode_jac(self,t,state):
        B0,BA,A,M=state
        return self.localJac(B0,BA,A,M,t,self.p['alpha'],self.p['Tmax'])
#    def calc_pde_analytic_jacobian(self,state):
#        B0,BA,A,M=np.split(state,3)
#        dbdb= sparse.diags(self.dbdb(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dbdw= sparse.diags(self.dbds1(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dbdh= sparse.diags(self.dbds2(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dwdb= sparse.diags(self.ds1db(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dwdw= sparse.diags(self.ds1ds1(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dwdh= sparse.diags(self.ds1ds2(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dhdb= sparse.diags(self.ds2db(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dhdw= sparse.diags(self.ds2ds1(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        dhdh= sparse.diags(self.ds2ds2(B0,BA,A,M,self.p['chi'],self.p['beta']))
#        local  = sparse.bmat([[dbdb,dbdw,dbdh],
#                              [dwdb,dwdw,dwdh],
#                              [dhdb,dhdw,dhdh]])
#        return sparse.csc_matrix(local)+sparse.csc_matrix(self.lapmat)

    def calc_ode_numerical_jacobian(self,B0,BA,A,M,delta=0.00000001):
        state = np.array([B0,BA,A,M])
        jacobian = []
        for j in range(len(state)):
            state_plus = np.copy(state)
            state_minus = np.copy(state)
            state_plus[j] = state_plus[j]+delta
            state_minus[j] = state_minus[j]-delta
            jacobian.append((np.array(self.dudt(state_plus))-np.array(self.dudt(state_minus)))/(2.0*delta))
        return np.array(jacobian).T
#    def check_pde_jacobians(self,n=100):
#        import time
#        timer_analytic=0
#        timer_numeric=0
#        error = np.zeros(n)
#        for i in range(n):
#            print(i)
#            x=np.random.random(self.setup['n'])
#            y=np.random.random(self.setup['n'])
#            state=np.ravel((B0,BA,A,M))
#            start_time=time.time()
#            numeric=self.calc_pde_numerical_jacobian(state)
#            mid_time=time.time()
#            analytic=self.calc_pde_analytic_jacobian(state)
#            end_time=time.time()
#            timer_numeric+=(mid_time-start_time)
#            timer_analytic+=(end_time-mid_time)
#            error[i]=np.max(np.abs(numeric-analytic))
#        print("Max difference is ",np.max(error), ", and mean difference is ",np.mean(error))
#        print("Average speed for numeric ", timer_numeric/float(n))
#        print("Average speed for analytic ", timer_analytic/float(n))
#        print("Analytic ", float(timer_numeric)/float(timer_analytic)," faster.")
#
#    def calc_pde_numerical_jacobian(self,state,delta=0.00000001):
#        n = len(state)
#        jacobian = []
#        for j in range(n):
#            state_plus = np.copy(state)
#            state_minus = np.copy(state)
#            state_plus[j] = state_plus[j]+delta
#            state_minus[j] = state_minus[j]-delta
#            jacobian.append((self.rhs_pde(state_plus)-self.rhs_pde(state_minus))/(2.0*delta))
#        return np.array(jacobian).T

#    def calc_numeric_pde_eigs(self,state):
#        return linalg.eigvals(self.calc_pde_numerical_jacobian(state))
#    def calc_analytic_pde_eigs(self,state):
#        return sparse.linalg.eigs(self.calc_pde_analytic_jacobian(state),k=3)[0]

#    def check_convergence(self,state,previous_state,tolerance=1.0e-5):
#        return np.max(np.abs(state-previous_state))<tolerance
    """Stochastic rainfall functions   """
    def update_parameters(self,parameters):
        intersection=[i for i in list(self.p.keys()) if i in parameters]
        if intersection:
            for key in intersection:
                if self.setup['verbose']:
                    print(str(key)+"="+str(parameters[key]))
                self.p[key]=parameters[key]

    """Generic integration function                """
#    def integrate(self,initial_state=None,step=10,
#                  max_time = 1000,tol=1.0e-5,plot=False,savefile=None,
#                  create_movie=False,check_convergence=True,
#                  sim_step=None,**kwargs):
#        if kwargs:
#            self.update_parameters(kwargs)
#        self.filename = savefile
#        if initial_state is None:
#            initial_state = self.initial_state
#        self.time_elapsed=0
#        if 'stochastic' in list(self.setup.keys()):
#            self.setup_stochastic_rainfall()
#        if sim_step is None:
#            self.sim_step=0
#        else:
#            self.sim_step=sim_step
#        if savefile is not None:
#            hn.setup_simulation(savefile,self.p,self.setup)
#            hn.save_sim_snapshot(savefile,self.sim_step,self.time_elapsed,
#                                 self.split_state(initial_state),self.setup)
##        old_result = initial_state
#        converged=False
#        result = []
#        result.append(initial_state)
##        t,result = self.integrator(initial_state,p=p,step=10,finish=10,savefile=self.filename)
#        if self.setup['verbose']:
#            start=time.time()
#            print("Step {:4d}, Time = {:5.1f}".format(self.sim_step,self.time_elapsed))
#        while not converged and self.time_elapsed<=max_time:
#            old_result = result[-1]
#            t,result = self.integrator(result[-1],step=step,finish=step)
##            self.time_elapsed+=t[-1]
#            self.sim_step=self.sim_step+1
#            if savefile is not None:
#                hn.save_sim_snapshot(savefile,self.sim_step,self.time_elapsed,
#                                     self.split_state(result[-1]),self.setup)            
#            if self.setup['verbose']:
#                print("Step {:4d}, Time = {:10.6f}, diff = {:7f}".format(self.sim_step,self.time_elapsed,np.max(np.abs(result[-1]-old_result))))
#            if check_convergence:
#                converged=self.check_convergence(result[-1],old_result,tol)
#                if converged:
#                    print("Convergence detected")
#        if self.setup['verbose']:
#            print("Integration time was {} s".format(time.time()-start))
#        if savefile is not None and create_movie:
#            print("Creating movie...")
#            hn.create_animation_b(savefile)
#        return result[-1]

    """ Integrators step functions """
    def ode_integrate(self,initial_state,step=1.0,start=0,finish=18250,
                          method='BDF',**kwargs):
        """ Using the new scipy interface to BDF method for stiff equations
        with option to switch to other methods
        """
        if kwargs:
            self.update_parameters(kwargs)
        t = np.arange(start,finish+step, step)
        if method=='BDF':
            sjac=self.scipy_ode_jac
        else:
            sjac=None
        sol=solve_ivp(fun=self.scipy_ode_rhs,t_span=(t[0],t[-1]),
                      y0=initial_state,method=method,max_step=step/10.0,
                      t_eval=t,jac=sjac)
        return sol.t,sol.y.T

    def euler_integrate(self,initial_state=None,step=0.1,finish=1000,**kwargs):
        """ Integration using Foreward Euler step
        """
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state
        time = np.arange(0,finish+step,step)
        result=np.zeros((len(time),len(initial_state)))
        t=0
        result[0]=initial_state
        for i,tout in enumerate(time[1:]):
            old=result[i]
            while t < tout:
                new=old+self.dt*self.rhs(old,self.time_elapsed)
                old=new
                t+=self.dt
                self.time_elapsed+=self.dt
            result[i+1]=old
        self.state=result[-1]
        return time,result
    def euler_integrate_stochastic(self,initial_state=None,step=0.1,finish=1000,**kwargs):
        """ Integration using Foreward Euler step
        """
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state
        time = np.arange(0,finish+step,step)
        self.p0=self.p['p']
        result=np.zeros((len(time),len(initial_state)))
        t=0
        result[0]=initial_state
        for i,tout in enumerate(time[1:]):
            old=result[i]
            while t < tout:
                self.add_stochasticity(t)
                new=old+self.dt*self.rhs(old,self.time_elapsed)
                old=new
                t+=self.dt
                self.time_elapsed+=self.dt
            result[i+1]=old
        self.state=result[-1]
        self.p_array=np.array(self.p_array)
        return time,result
    
    def rk4_integrate(self,initial_state=None,step=0.1,finish=1000,**kwargs):
        """ Integration using The Runge–Kutta method of 4th order as in:
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
        """
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state
        time = np.arange(0,finish+step,step)
        result=np.zeros((len(time),len(initial_state)))
        t=0
        result[0]=initial_state
        for i,tout in enumerate(time[1:]):
            old=result[i]
            while t < tout:
                k1=self.rhs(old,self.time_elapsed)
                k2=self.rhs(old+0.5*self.dt*k1,self.time_elapsed+(self.dt/2.0))
                k3=self.rhs(old+0.5*self.dt*k2,self.time_elapsed+(self.dt/2.0))
                k4=self.rhs(old+self.dt*k3,self.time_elapsed+(self.dt))
                new=old+(self.dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
                old=new
                t+=self.dt
                self.time_elapsed+=self.dt
            result[i+1]=old
        self.state=result[-1]
        return time,result

    def scipy_integrate_stochastic(self,initial_state=None,
                                   step=0.1,start=0,finish=1000,
                                   method='RK45',**kwargs):
        """ Using the new scipy interface to RK45 method
        with option to switch to other methods
        """
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state
        t = np.arange(start,finish+step,step)
        self.setup_stochastic_rainfall(start=start)
        sol=solve_ivp(fun=self.scipy_ode_rhs_stochastic,t_span=(t[0],t[-1]),
                      y0=initial_state,method=method,max_step=self.p['conv_T_to_t'],
                      t_eval=t)
        return sol.t,sol.y.T  
     
    def rk4_integrate_stochastic(self,initial_state=None,
                                 step=0.1,start=0,finish=1000,**kwargs):
        """ Integration using The Runge–Kutta method of 4th order as in:
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
        """
        print("Integration using stochastic rk4 step")
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state
        time = np.arange(start,finish+step,step)
        self.setup_stochastic_rainfall(start=start)
        result=np.zeros((len(time),len(initial_state)))
        t=start
        result[0]=initial_state
        self.prec=np.zeros(len(time))
        self.prec[0]=self.p0
        for i,tout in enumerate(time[1:]):
            old=result[i]
            while t < tout:
                self.add_stochasticity(t)
                k1=self.rhs(old,self.time_elapsed)
                k2=self.rhs(old+0.5*self.dt*k1,self.time_elapsed+(self.dt/2.0))
                k3=self.rhs(old+0.5*self.dt*k2,self.time_elapsed+(self.dt/2.0))
                k4=self.rhs(old+self.dt*k3,self.time_elapsed+(self.dt))
                new=old+(self.dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
                old=new
                t+=self.dt
                self.time_elapsed+=self.dt
            result[i+1]=old
            self.prec[i+1]=self.p_t(t,self.p['p'],self.p['a'],self.p['omegaf'])
        self.state=result[-1]
        self.p_array=np.array(self.p_array)
        return time,result
    def pseudo_spectral_integrate(self,initial_state=None,step=0.1,finish=1000,**kwargs):
#        print "Integration using pseudo-spectral step"
        if kwargs:
            self.update_parameters(kwargs)
        result=[]
        time = np.arange(0,finish+step,step)
        t=0
        result.append(initial_state)
        for tout in time[1:]:
            self.state=result[-1]
            B0,BA,A,M=self.state.reshape(self.setup['nvar'],*self.setup['n'])
            self.fftb=fftn(x)
            self.ffts1=fftn(y)
            self.ffts2=fftn(s2)
            while t < tout:
                self.fftb = self.multb*(self.fftb + self.dt*fftn(self.dbdt(B0,BA,A,M,t,self.p['p'],self.p['chi'],self.p['beta'],self.p['a'])))#.real
                self.ffts1 = self.mults1*(self.ffts1 + self.dt*fftn(self.dwdt(B0,BA,A,M,t,self.p['p'],self.p['chi'],self.p['beta'],self.p['a'])))#.real
                self.ffts2 = self.mults2*(self.ffts2 + self.dt*fftn(self.ds2dt(B0,BA,A,M,t,self.p['p'],self.p['chi'],self.p['beta'],self.p['a'])))#.real
                x= ifftn(self.fftb).real
                y= ifftn(self.ffts1).real
                s2= ifftn(self.ffts2).real
                t+=self.dt
                self.time_elapsed+=self.dt
            self.state=np.ravel((B0,BA,A,M))
            self.sim_step+=1
            result.append(self.state)
        return time,result

    def spectral_multiplier(self,dt):
        n=self.setup['n']
        nx=n[0]
        dx=self.dx
        # wave numbers
        k=2.0*np.pi*np.fft.fftfreq(nx,dx)
        if len(n)==1:
            k2=k**2
        if len(n)==2:
            k2=np.outer(k,np.ones(nx))**2
            k2+=np.outer(np.ones(nx),k)**2
        # multiplier
        self.multb = np.exp(-dt*k2)
        self.mults1= np.exp(-dt*self.p['delta_s']*k2)
        self.mults2= np.exp(-dt*self.p['delta_s']*k2)
    """ Auxilary root finding functions """
    def ode_root(self,initial_state,p=None,chi=None,beta=None,a=None,omegaf=None):
        """ """
        if p is not None:
            self.p['p']=p
        if chi is not None:
            self.p['chi']=chi
        if beta is not None:
            self.p['beta']=beta
        if a is not None:
            self.p['a']=a
        if a is not None:
            self.p['omegaf']=omegaf
        sol = root_ode(self.rhs_ode, initial_state)
        return sol.x
    def pde_root(self,initial_state,p=None,chi=None,beta=None,a=None,omegaf=None, fixiter=100,tol=6e-6,smaxiter=1000,integrateiffail=False):
        """ """
        if p is not None:
            self.p['p']=p
        if chi is not None:
            self.p['chi']=chi
        if beta is not None:
            self.p['beta']=beta
        if a is not None:
            self.p['a']=a
        if a is not None:
            self.p['omegaf']=omegaf
        try:
            sol = newton_krylov(self.rhs_pde, initial_state,iter=fixiter, method='lgmres', verbose=int(self.setup['verbose']),f_tol=tol,maxiter=max(fixiter+1,smaxiter))
            converged=True
        except NoConvergence:
            converged=False
            if self.setup['verbose']:
                print("No Convergence flag")
            sol=initial_state
            if integrateiffail:
                print("Integrating instead of root finding")
                sol=self.integrate_till_convergence(initial_state,p)
        return sol,converged

    def setup_initial_condition(self,Vs,**kwargs):
        n = self.setup['n']
        if type(Vs)==str:
            if Vs == "random":
                x = np.random.random(n)*0.5 + 0.1
                y= np.random.random(n)*(0.1) + 0.05
                s2= np.random.random(n)*(0.1) + 0.05
            if Vs == "bare":
                x = np.zeros(n)
                y= np.random.random(n)*(0.1) + 0.05
                s2= np.random.random(n)*(0.1) + 0.05
            elif Vs == "tile":
                fields = kwargs.get('fields', None)
                B0,BA,A,M = np.split(fields,self.setup['nvar'])
                x  = np.tile(x,(self.setup['n'][0],1))
                y = np.tile(y,(self.setup['n'][0],1))
                s2 = np.tile(s2,(self.setup['n'][0],1))                
            elif Vs == "half":
#                import matplotlib.pyplot as plt
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.integrate_ode_bdf([0.01,0.2,0.2],p)
#                plt.plot(t,result[0])
                b_0,s1_0,s2_0 = result.T[-1]
                t,result=self.integrate_ode_bdf([0.9,0.2,0.2],p)
#                plt.plot(t,result[0])
                b_s,s1_s,s2_s = result.T[-1]
                x = np.ones(n)*b_0
                y= np.ones(n)*s1_0
                s2= np.ones(n)*s2_0
                half = int(self.setup['n'][0]/2)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                x[:half]=b_s
                y[:half]=s1_s
                s2[:half]=s2_s
#                plt.show()
            elif Vs == "halfrandom":
#                import matplotlib.pyplot as plt
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.integrate_ode_bdf([0.01,0.2,0.2],p)
#                plt.plot(t,result[0])
                b_0,s1_0,s2_0 = result.T[-1]
                t,result=self.integrate_ode_bdf([0.9,0.2,0.2],p)
#                plt.plot(t,result[0])
                b_s,s1_s,s2_s = result.T[-1]
                x = np.ones(n)*b_0
                y= np.ones(n)*s1_0
                s2= np.ones(n)*s2_0
                half = int(self.setup['n'][0]/2)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                x[:half]=b_s*np.random.random(n)[:half]
                y[:half]=s1_s*np.random.random(n)[:half]
                s2[:half]=s2_s*np.random.random(n)[:half]
#                plt.show()
            self.initial_state = np.ravel((B0,BA,A,M))
        else:
            self.initial_state = Vs
        self.state = self.initial_state
        if self.setup['it'] == 'pseudo_spectral' and self.setup['setPDE']:
            self.spectral_multiplier(self.dt)
    """ Plot functions """
    def plotLastFrame(self,initial_state=None,p=None,chi=None,beta=None,a=None,savefile=None):
        sol=self.integrate_till_convergence(initial_state,p,chi,beta,a,savefile)
        self.plot(sol)
        return sol
    def split_state(self,state):
        return state.reshape(self.setup['nvar'],*self.setup['n'])
    
    def plot(self,state,fontsize=12,update=False):
        import matplotlib.pylab as plt
        if update:
            plt.ion()
        B0,BA,A,M=state.reshape(self.setup['nvar'],*self.setup['n'])
        if len(self.setup['n'])==1:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row')
    #        x = np.linspace(0,self.setup['l'][0],self.setup['n'][0])
            ax1.set_ylim([-0.1,1.0])
            ax2.set_ylim([0.0,self.p['s_fc']])
            ax3.set_ylim([0.0,self.p['s_fc']])
            ax1.plot(self.X,x)
            ax1.set_xlim([0,self.X[-1]])
            ax2.plot(self.X,y)
            ax3.plot(self.X,s2)
            ax1.set_title(r'$x$', fontsize=fontsize)
            ax2.set_title(r'$s_1$', fontsize=fontsize)
            ax3.set_title(r'$s_2$', fontsize=fontsize)
        elif len(self.setup['n'])==2:
            fig, (ax1, ax2,ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
            fig.subplots_adjust(right=0.8)
#            ax1.imshow(x,cmap=plt.cm.YlGn, vmin = bmin, vmax = bmax)
            ax1.imshow(x,cmap=plt.cm.YlGn)
            ax1.set_adjustable('box-forced')
            ax1.autoscale(False)
            ax1.set_title(r'$x$', fontsize=fontsize)
            ax2.imshow(y,cmap=plt.cm.Blues)
            ax2.set_adjustable('box-forced')
            ax2.autoscale(False)
            ax2.set_title(r'$s_1$', fontsize=fontsize)
#            ax3.imshow(y,cmap=plt.cm.Blues, vmin = smin, vmax = smax)
            ax3.imshow(s2,cmap=plt.cm.Blues)
            ax3.set_adjustable('box-forced')
            ax3.autoscale(False)
            ax3.set_title(r'$s_2$', fontsize=fontsize)
#            im4=ax4.imshow(s2,cmap=plt.cm.Blues, vmin = smin, vmax = smax)
#            cbar_ax2 = fig.add_axes([0.85, 0.54, 0.03, 0.35])
#            fig.colorbar(im1, cax=cbar_ax2)
#            plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()