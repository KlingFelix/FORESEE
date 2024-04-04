import numpy as np
import sys,math
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from cycler import cycler
#import HeavyNeutralLepton as hnl


src_path = "../.."
sys.path.append(src_path)
from src.foresee import Foresee, Utility, Model

foresee = Foresee()
energy = "14"
modelname="HNL"
FORESEE = Model(modelname, path="././")

Quark_level_cutoff = FORESEE.masses('331')

"""

Physical Constants 

"""
#from PDG (https://pdg.lbl.gov/2023/reviews/rpp2022-rev-phys-constants.pdf)
Gf = 1.1663788 * (10**-5) 

sin2w = 0.23121 

GeVtoS = (6.582119569 * 10**-25) 

c = 299792458 

#Weak coupling

glL = -0.5 + sin2w

glR = sin2w

guL = 0.5 - (2*sin2w/3)

guR = -(2*sin2w/3)

gdL = -0.5 + (sin2w/3)

gdR = (sin2w/3)


glL = -0.5 + sin2w
glR = sin2w
guL = 0.5 - (2*sin2w/3)
guR = -(2*sin2w/3)
gdL = -0.5 + (sin2w/3)
gdR = (sin2w/3)


# CKM Matrix Elements (PDG: https://pdg.lbl.gov/2022/web/viewer.html?file=../reviews/rpp2022-rev-ckm-matrix.pdf) 
Vckm = {'ud':0.97373 , 'us':0.2243, 'ub':3.82E-3 ,
             'cd': 0.221, 'cs':0.975, 'cb':40.8E-3 ,
             'td': 8.6E-3, 'ts':41.5E-3, 'tb':1.014}

# Meson Decay Constants (Helo, Kovalenko, Schmidt 2018: https://arxiv.org/abs/1005.1607) 
def f(x): 
    
    if x == 'rho+' or anti(x) == 'rho+': return 0.210 
    elif x == 'K+*' or anti(x) =='K+*': return 0.204
    elif x == 'rho0' or anti(x) =='rho0': return 0.220
    elif x == 'omega' or anti(x) =='omega': return 0.195
    elif x =='pi+' or anti(x) =='pi+': return  0.1303
    elif x =='K+' or anti(x) =='K+': return 0.1564
    elif x =='pi0' or anti(x) =='pi0': return 0.1303
    elif x =='eta' or anti(x) =='eta': return 0.0784
    elif x =='eta_p' or anti(x) =='eta_p': return -0.0957
           
    

    
    
    
"""

Kinematic Functions

"""

def Lambda(x,y,z):
    
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*x*z


def I_1_2body(x,y):
    
    return np.sqrt(Lambda(1,x,y)) * ((1 - x)**2 - y*(1 + x)) 

def I_2_2body(x,y): 
    
    return np.sqrt(Lambda(1,x,y)) * ((1+ x - y)*(1 + x + 2*y) - 4*x)

def I_1_3body(x,y,z): 
    
    #changed 1->s 
    
    integrand = lambda s: (1/s)*(s - x - y)*(1 + z - s)*np.sqrt(Lambda(s,x,y))*np.sqrt(Lambda(1,s,z))
    
    integral,error = integrate.quad(integrand, (np.sqrt(x) + np.sqrt(y))**2, (1 - np.sqrt(z))**2)
    
    
    return 12*integral

 
def I_2_3body(x,y,z): 
    
    integrand = lambda s: (1/s)*(1 + x - s)*np.sqrt(Lambda(s,y,z))*np.sqrt(Lambda(1,s,x))
    
    integral,error = integrate.quad(integrand, (np.sqrt(y) + np.sqrt(z))**2, (1 - np.sqrt(x))**2)
  
    
    return 24*np.sqrt(y*z)*integral


def delta(l1,l2):
    
    if l1 == l2 or l1 == -1*l2:
        return 1
    else: 
        return 0   


       
    
"""

Decay Widths (Helo, Kovalenko, Schmidt (2018): https://arxiv.org/abs/1005.1607)

"""

#N-> l_alpha P
def Gamma_lP(self,m,mode,cutoff=Quark_level_cutoff): 

    l,P = mode

    
    V_dict = {'pi+': Vckm['ud'], #pi+
              'anti_pi+':Vckm['ud'] ,#pi-
              'K+': Vckm['us'], #K+
              'anti_K+': Vckm['us'] #K-
    }
    
    

    prefactor = self.U[l]**2 * Gf**2 * m**3 * f(P)**2 * V_dict[P]**2 / (16*np.pi) 

    yl1 = FORESEE.masses(pid(l))/m

    yP = FORESEE.masses(pid(P))/m


    #evaluate mass thresholds
    if cutoff == None:

        if 1 >= yl1 + yP:
            return prefactor*I_1_2body(yl1**2,yP**2)
        else:
            return 0
    else:

        if 1 >= yl1 + yP and m <= cutoff:
            return prefactor*I_1_2body(yl1**2,yP**2)
        else:
            return 0 

#N -> nu P 
def Gamma_nuP(self,m,mode,cutoff=Quark_level_cutoff):

    nu,P  = mode


    prefactor =   (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2) * Gf**2 * m**3 * f(P)**2 / (16*np.pi)

    yP = FORESEE.masses(pid(P))/m
    
    gamma = prefactor*I_1_2body(0,yP**2)
    
    #evaluate mass thresholds
    if cutoff == None:
        if 1 >= yP:
            return gamma
        else:
            return 0 
    else: 
        if 1>= yP and m <= cutoff:
            return gamma
        else:
            return 0 

#N -> l_alpha V 
def Gamma_lV(self,m,mode,cutoff=Quark_level_cutoff):

    l,V = mode
    

    V_dict = {'rho+': Vckm['ud'],
              'anti_rho+': Vckm['ud'],
              'K+*': Vckm['us'],
              'anti_K+*': Vckm['us'],
             }

    prefactor =   self.U[l]**2*Gf**2 * m**3 * f(V)**2 * V_dict[V]**2 / (16*np.pi) 


    yl1 = FORESEE.masses(pid(l))/m

    yV = FORESEE.masses(pid(V))/m
    
    
    gamma = prefactor*I_2_2body(yl1**2,yV**2)
    
    #evaluate mass threshold
    if cutoff == None:
        if 1 >= yV+yl1:
            return gamma
        else:
            return 0   

    else:
        if 1 >= yV+yl1 and m<=cutoff:
            return gamma
        else:
            return 0  

#N -> nu_alpha V 
def Gamma_nuV(self,m,mode,cutoff=Quark_level_cutoff):

    nu,V = mode

    k_V = {#'rho0': 1-sin2w, 
           'rho0': 1-2*sin2w, 
           'anti_rho0': 1-2*sin2w,
           'omega': 4*sin2w/3,
           'anti_omega': 4*sin2w/3,
          }

    prefactor = (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2) * Gf**2 * m**3 * f(V)**2 * k_V[V]**2 / (16*np.pi)

    yV = FORESEE.masses(pid(V))/m

    
    gamma =  prefactor*I_2_2body(0,yV**2)
    
    
    #evaluate mass thresholds
    if cutoff == None:
        if 1 >= yV:
            return gamma
        else:
            return 0   

    else:
        if 1 >= yV and m<=cutoff:
            return gamma
        else:
            return 0  


#N -> l_alpha l_beta nu_beta
def Gamma_llnu(self,m,mode,cutoff=None):

    l1,l2,nu = mode

    
        
    yl1 = FORESEE.masses(pid(l1))/m
    yl2 = FORESEE.masses(pid(l2))/m
    
    
    ynu = 0
    
    
    
    gamma = Gf**2 * m**5 * ( self.U[l1]**2 * I_1_3body(0, yl1**2, yl2**2) + self.U[l2]**2 * I_1_3body(0,yl2**2,yl1**2) ) / (192 * np.pi**3)
    
    
    
    #evaluate mass thresholds 
    if 1 >= yl1 + yl2:
        return gamma
    else:
        return 0   

#N -> nu_alpha l_beta l_beta
def Gamma_null(self,m,mode,cutoff=None):
    nu,l1,l2 = mode

    yl = FORESEE.masses(pid(l1))/m
    

    ynu = 0

    del_e = delta(int(pid('e')),int(pid(l1)))
    del_mu = delta(int(pid('mu')),int(pid(l1)))
    del_tau = delta(int(pid('tau')),int(pid(l1)))
    

    prefactor =  Gf**2 * m**5 / (96*np.pi**3)

    term_e = self.U['e']**2 * (  ( glL*glR + glR*del_e )*I_2_3body(0,yl**2,yl**2) + (glL**2 + glR**2 + (1+2*glL)*del_e)*I_1_3body(0,yl**2,yl**2))
                                                                                                                   
    term_mu = self.U['mu']**2 * (  ( glL*glR + glR*del_mu )*I_2_3body(0,yl**2,yl**2) + (glL**2 + glR**2 + (1+2*glL)*del_mu)*I_1_3body(0,yl**2,yl**2))
    
    term_tau = self.U['tau']**2 * (  ( glL*glR + glR*del_tau )*I_2_3body(0,yl**2,yl**2) + (glL**2 + glR**2 + (1+2*glL)*del_tau)*I_1_3body(0,yl**2,yl**2))
    
                                                                                                                   
    gamma = prefactor*(term_e + term_mu + term_tau)

    #evaluate mass thresholds
    if 1 >= 2*yl:
        return gamma
    else:
        return 0   

#N -> nu_alpha nu nu 
def Gamma_nu3(self,m,mode,cutoff=None):
    nu,_,_ = mode


    gamma = (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2)*Gf**2 * m**5 / (96*np.pi**3)

    return gamma




#N -> l_alpha u d
def Gamma_lud(self,m,mode,cutoff=Quark_level_cutoff):

    l,u,d = mode

 


    yu = FORESEE.masses(pid(u))/m
    yd = FORESEE.masses(pid(d))/m
    yl = FORESEE.masses(pid(l))/m

    
    if 'anti' in d: V = Vckm[u+anti(d)]
    else: V = Vckm[anti(u)+d]

        
    gamma = self.U[l]**2*Gf**2 * V**2 * m**5 * I_1_3body(yl**2,yu**2,yd**2) / (64 * np.pi**3)
    
    #evaluate mass thresholds
    if cutoff == None:
        if 1 >= yu+yd+yl:
            return gamma
        else:
            return 0

    else:
        if 1 >= yu+yd+yl and m>=cutoff:
            return gamma
        else:
            return 0  


#N -> nu_alpha q q
def Gamma_nuqq(self,m,mode,cutoff=Quark_level_cutoff):

    nu,q,qbar = mode 

   
    yq = FORESEE.masses(pid(q))/m

    ynu = 0 


    prefactor = (self.U['e']**2 + self.U['mu']**2 + self.U['tau']**2) * Gf**2 * m**5  / (32*np.pi**3)
    
    #pick out coupling constants
    if q  in self.particle_content['quarks']['up'] or qbar in self.particle_content['quarks']['up']:
        gqL = guL
        gqR = guR
    elif q  in self.particle_content['quarks']['down'] or qbar in self.particle_content['quarks']['down']:
        gqL = gdL
        gqR = gdR


    term1 = gqL*gqR*I_2_3body(ynu**2,yq**2,yq**2)

    term2 = (gqL**2 + gqR**2)*I_1_3body(ynu,yq**2,yq**2)

    #evaluate mass threshold
    if cutoff == None:
        if 1 >= yq+yq:
            return prefactor*(term1 + term2)

        else:
            return 0
    else:
        if 1 >= yq+yq and m>=cutoff:
            return prefactor*(term1 + term2)
        else:
            return 0


"""

Additional Functions

"""
#returns the anti particle
def anti(x):
    
    if 'anti_' not in x: return 'anti_'+x
        
    elif 'anti_' in x:  return x.replace('anti_','')
    
plot_labels = {
        #Leptons
        'e': r'$e$',
        anti('e'): r'$e$',
    
        'mu': r'$\mu$',
        anti('mu'): r'$\mu$',
        
        'tau': r'$\tau$',
        anti('tau'): r'$\tau$',
        
        've': r'$\nu_e$',
        anti('ve'): r'$\overline{\nu}_e$',
    
        'vmu': r'$\nu_\mu$',
        anti('vmu'): r'$\overline{\nu}_\mu$',
        
        'vtau': r'$\nu_\tau$',
        anti('vtau'): r'$\overline{\nu}_\tau$',
        'nu':r'$\nu$',
    
        #Pseudos
        'pi+':r'$\pi^+$',
        anti('pi+'):r'$\pi^-$',
    
        'pi0':r'$\pi^0$',
    
        'K+': r'$K^+$',
        anti('K+'): r'$K^-$',
    
        'eta': r'$\eta$',
    
        #Vectors
        'rho+':r'$\rho^+$',
        anti('rho+'):r'$\rho^-$',
    
        'rho0':r'$\rho^0$',
    
        'K+*':r'$K^{*+}$',
        anti('K+*'):r'$K^{*-}$',
    
        'omega':r'$\omega$',
        
        #Quarks
        'd': r'$d$',
        anti('d'): r'$\overline{d}$',
        
        'u': r'$u$',
        anti('u'): r'$\overline{u}$',
        
        'c': r'$c$',
        anti('c'): r'$\overline{c}$',
    
        's': r'$s$',
        anti('s'): r'$\overline{s}$',
    
        't': r'$t$',
        anti('t'): r'$\overline{t}$',
    
        'b': r'$b$',
        anti('b'): r'$\overline{b}$',
        
            
        }

plot_labels_neut = {
        #Leptons
        'e': r'$e$',
        anti('e'): r'$e$',
    
        'mu': r'$\mu$',
        anti('mu'): r'$\mu$',
        
        'tau': r'$\tau$',
        anti('tau'): r'$\tau$',
        
        've': r'$\nu_e$',
        anti('ve'): r'$\overline{\nu}_e$',
    
        'vmu': r'$\nu_\mu$',
        anti('vmu'): r'$\overline{\nu}_\mu$',
        
        'vtau': r'$\nu_\tau$',
        anti('vtau'): r'$\overline{\nu}_\tau$',
        'nu':'$\nu$',
    
        #Pseudos
        'pi+':r'$\pi$',
        anti('pi+'):r'$\pi$',
    
        'pi0':r'$\pi^0$',
    
        'K+': r'$K$',
        anti('K+'): r'$K$',
    
        'eta': r'$\eta$',
    
        #Vectors
        'rho+':r'$\rho$',
        anti('rho+'):r'$\rho$',
    
        'rho0':r'$\rho^0$',
    
        'K+*':r'$K^{*}$',
        anti('K+*'):r'$K^{*}$',
    
        'omega':r'$\omega$',
        
        #Quarks
        'd': r'$d$',
        anti('d'): r'$\overline{d}$',
        
        'u': r'$u$',
        anti('u'): r'$\overline{u}$',
        
        'c': r'$c$',
        anti('c'): r'$\overline{c}$',
    
        's': r'$s$',
        anti('s'): r'$\overline{s}$',
    
        't': r'$t$',
        anti('t'): r'$\overline{t}$',
    
        'b': r'$b$',
        anti('b'): r'$\overline{b}$',
        
            
        }


pid_conversions = {    
        #Leptons
        'e': 11,
        'mu': 13, 
        'tau': 15,

        #Neutrinos
        've': 12,
        'vmu': 14,
        'vtau': 16,
        'nu':20,


        #Pseudos
        'pi+':211,
        'pi0':111,
        'K+':321,
        'eta':221,

        #Vectors
        'rho+':213,
        'rho0':113,
        'K+*':323,
        'omega':223,
    
        #Quarks
        'd': 1,
        'u':2,
        's':3,
        'c':4,
        'b':5,
        't':6,
        }


#returns the id given a particle
def pid(x):
      
    if 'anti_' not in x: return str(pid_conversions[x])
    
    elif 'anti_' in x:  return str(-1*pid_conversions[x.replace('anti_','')])
    

#returns the conjugate mode
def conjugate(x):
    
    conj_mode = []
    
    for p in x: conj_mode.append(anti(p))
        
    return tuple(conj_mode)
    
    
"""

HNL Decay Object

"""

class HNL_Decay:
    
    
    def __init__(self,couplings,cutoff=Quark_level_cutoff):
        
        
        
        
        self.cutoff = cutoff
        self.U = {'e':couplings[0], 'anti_e': couplings[0],
                  'mu':couplings[1], 'anti_mu': couplings[1],
                  'tau':couplings[2], 'anti_tau': couplings[2]} 
        
        
        
        

        #define particle content
        leptons = ['e','mu','tau']

        vectors = {'charged':['rho+','K+*'], 'neutral': ['rho0','omega'] }

        pseudos = {'charged':['pi+','K+'], 'neutral':['pi0','eta'] }
        
        neutrinos = ['nu']
        
        quarks = {'up':['u','c','t'], 'down': ['d','s','b']}

        self.particle_content = {'leptons':leptons,'neutrinos':neutrinos,'vectors':vectors,'pseudos':pseudos,'quarks':quarks}



        ##Compile all allowed decay modes for each decay channel##
        
        #N -> nu_alpha l_beta l_beta
        null = [('nu','e','anti_e'),('nu','mu','anti_mu'),('nu','tau','anti_tau')]

        
        
        #N -> l_alpha l_beta nu
        llnu = [
        ('e',anti('mu'),'nu'), ('mu',anti('e'),'nu'),
        ('e',anti('tau'),'nu'), ('tau',anti('e'),'nu'),
        ('mu',anti('tau'),'nu'), ('tau',anti('mu'),'nu')    
        ] 

        #N -> nu nu nu
        nu3 = [('nu','nu','nu')]
    
        #N -> nu_alpha P    
        nuP = [('nu','pi0'),('nu','eta')]

     

        #N -> l_alpha P
        lP = []

        for l in leptons:

            for P in pseudos['charged']:

                mode = (l,P)

                lP.append(mode)
                
                #conjugate mode 
                lP.append(conjugate(mode))
        
        #N -> nu_alpha V
        nuV = []

        

        for V in vectors['neutral']:

            mode = ('nu',V)

            nuV.append(mode)

        #N -> l_alpha V 
        lV = []

        for l in leptons:

            for V in vectors['charged']:

                mode = (l,V)

                lV.append(mode)
                
                #conjugate mode
                lV.append(conjugate(mode))
                    
        #N -> nu_alpha q q 
        nuqq = []
        
        
            
        for q in quarks['up'] + quarks['down']:

            mode = ('nu',q,anti(q))

            nuqq.append(mode)
                
        #N -> l_alpha u d      
        lud = []
        
        for l in leptons: 
            
            for u in quarks['up']:
                
                for d in quarks['down']:
                    
                    mode = (l,u,anti(d))
                    
                    lud.append(mode)
                    
                    lud.append(conjugate(mode))
                

        self.modes = {'null':null,'llnu':llnu,'nu3':nu3,'nuP':nuP,'lP':lP,'nuV':nuV, 'lV':lV,'lud':lud,'nuqq':nuqq}
        
        
        
        self.modes_inactive = {}
        self.modes_active = {}
        
        #Compile all modes that are allowed by couplings and filter out those that are not
        for channel in self.modes.keys():
            
            modes_inactive = []
            modes_active = []
            
            for mode in self.modes[channel]:
            
            
                if channel in ['null','nuqq','nuV','nuP','nu3']: modes_active.append(mode)
                
                elif channel in ['lV','lP','lud']: 
                    if self.U[mode[0]] != 0: modes_active.append(mode)
                    else: modes_inactive.append(mode)
                
                elif channel in ['llnu']:
                    if self.U[mode[0]] != 0 or self.U[mode[1]] != 0: modes_active.append(mode)
                    else: modes_inactive.append(mode)
                        
                     
            


            
            
            self.modes_inactive[channel] = modes_inactive
            
            self.modes_active[channel] = modes_active
        
        
        
    
        
    def gen_widths(self,mpts,full_range=False):
        
        """
        
        Generate decay widths for all active modes
        
        """
        
        channels = self.modes_active.keys()
        
        self.model_widths = {}
        
        self.mpts = mpts
       
        #iterate through each decay channel
        for channel in channels:
            
            channel_widths = {}
            
            for mode in self.modes_active[channel]:
                
                scope = globals().update(locals())
                
                #Evaluate the decay width with or without a cutoff
                if full_range == True:
                    
                    gamma_pts =  eval(f'[Gamma_{channel}(self,m=m,mode=mode,cutoff=None) for m in self.mpts]',scope)
       
                    channel_widths[mode] = gamma_pts
            
                else:
                    
                    gamma_pts = eval(f'[Gamma_{channel}(self,m=m,mode=mode) for m in self.mpts]',scope)
       
                    channel_widths[mode] = gamma_pts
            
            self.model_widths[channel] = channel_widths
            
            
       
                
    def gen_ctau(self,mpts):
        
        """
        
        Generate HNL Lifetime
        
        """
        
        self.mpts = mpts
        
        #generate decay widths
        self.gen_widths(mpts=self.mpts,full_range=False)
        
        total_width = []
        
        #iterate over mass points
        for i in range(len(mpts)):
            
            gamma_T = 0
            
            #sum over individual decay widths 
            for channel in self.modes_active.keys():
                
                for mode in self.modes_active[channel]:
                    
                    
                    gamma = self.model_widths[channel][mode][i]
                    

                    
                    gamma_T += gamma
            
            
            total_width.append(gamma_T)
        
        ctau = [(c/g)*GeVtoS for g in total_width]
        
        self.total_width = total_width
        
        self.ctau = ctau
        
        
        
    def gen_brs(self,save=False):
        
         
        """
        
        Generate Branching ratios
        
        """
        
        channels = self.model_widths.keys()
        
        
        self.model_brs = {}
        
        #iterate over decay channels
        for channel in channels:
            
            channel_brs = {}
            
            #sum over branching ratios
            for mode in self.model_widths[channel]:
                
                mode_br = []
                
               
                for i in range(len(mpts)):
                    
                    gamma_partial = self.model_widths[channel][mode][i]
                    
                    gamma_total = self.total_width[i]
                    
                 
                    
                    mode_br.append( gamma_partial/gamma_total)
                    
                
                channel_brs[mode] = mode_br
                
                
            self.model_brs[channel] = channel_brs
                
            
    def plot_br(self,curves,ylims,title,xlims = None,xscale = 'log',yscale = 'log',show_negligible=True,savefig=False,filename=None,gamma = False):
        
        if xlims == None: xlims =  (min(self.mpts),max(self.mpts))
        
       
        
        
        default_cycler = (cycler(linestyle=['-', '--'])*cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','gray','lightblue']) 
                  )

       
        
        
        plt.rc('axes', prop_cycle=default_cycler)
        
        
        
        
      
        
        
        fig,ax = plt.subplots()
        
        ax.set(xlim=xlims,ylim=ylims,xscale=xscale,yscale=yscale,title=title)
        
        
        ax.set_xlabel(r'$m_N$ (GeV)', fontsize = 20) 
        if gamma == False: ax.set_ylabel(r"$B(N\to X)$", fontsize = 20) 
        if gamma == True:  ax.set_ylabel(r"$\Gamma(N\to X)$", fontsize = 20) 
            
        ax.tick_params(axis='both', which='major',direction='in' , labelsize=15,length=8, width=1,top=True,right=True)
        ax.tick_params(axis='both', which='minor',direction='in' , labelsize=15,length=4, width=1,top=True,right=True)
        
        
        fig.set_size_inches(9,6, forward=True)
        negligible_curves = []
                    
        
        for curve in curves: 
            
            in_range = 0 
            
            
            
            curve_type = type(curve)
            
            
            
            if curve_type is list:
                
                #These are collective modes
                
                curve_label = curve[0]
                
                curve_br_pts = []
                
                curve = curve[1:]
                
                for i in range(len(self.mpts)):
            
                    br_m = 0.0
                    

                    for mode in curve:
                        
                        for channel in self.model_brs.keys():
                            
                            
                            
                            if mode in self.model_brs[channel].keys():
                                
                                if channel in ['nuqq','lud'] and self.mpts[i] > Quark_level_cutoff:
                                    
                                    if gamma == False: br_m += self.model_brs[channel][mode][i]
                                    elif gamma == True: br_m += self.model_widths[channel][mode][i]
                                        
                                elif channel in ['lV','nuV','nuP','lP'] and self.mpts[i] < Quark_level_cutoff:
                                    if gamma == False: br_m += self.model_brs[channel][mode][i]
                                    elif gamma == True: br_m += self.model_widths[channel][mode][i]
                                    
                                elif channel in ['null','llnu','nu3']:
                                    if gamma == False: br_m += self.model_brs[channel][mode][i]
                                    elif gamma == True: br_m += self.model_widths[channel][mode][i]
                                else: 
                                    br_m = np.nan
                                

                    if br_m != None:
                        if br_m > ylims[0] and in_range == 0: 
                            index_in_range = i
                            in_range +=1

                        curve_br_pts.append(br_m)
                
                if in_range > 0: 
                    line,= ax.plot(self.mpts,curve_br_pts,label=curve_label)
                    
                else: 
                    if not all(br== 0.0 for br in curve_br_pts):negligible_curves.append(curve_label)
                
                
                
                
            else:
                
                #These are all the individual modes
                

                mode = curve
                #print(curve)

                for channel in self.model_brs.keys():

                    if mode in self.model_brs[channel].keys():

                        in_range = 0 





                        if gamma == False: br_pts = self.model_brs[channel][mode]
                        if gamma == True: br_pts = self.model_widths[channel][mode]
                                

                        for i,br in enumerate(br_pts): 

                            if br > ylims[0]: 
                                in_range +=1

                                index_in_range = i
                                break

                        label = ""

                        for p in mode:

                            label += plot_labels[p]





                        if in_range > 0: 
                            line, = ax.plot(self.mpts,br_pts,label = label)


                        else: 
                            if not all(br== 0.0 for br in br_pts):negligible_curves.append(label)
        
        #relevant_legend = ax.legend(ncol=3,loc='upper left',framealpha=.4,bbox_to_anchor=(.49, .39),frameon=True, fontsize="12", columnspacing=0.5) #011
        relevant_legend = ax.legend(ncol=3,loc='upper left',framealpha=.4,bbox_to_anchor=(.49, .45),frameon=True, fontsize="12", columnspacing=0.5) #111
        #relevant_legend = ax.legend(ncol=2,loc='lower right',framealpha=.4,frameon=True, fontsize="14", columnspacing=0.5)
        ax.add_artist(relevant_legend)
        ax.axvline(x = Quark_level_cutoff, color = 'black', linestyle = 'dotted',alpha= .8,linewidth=1)
        #print('Out of Figure Bounds: ', negligible_curves)
        if len(negligible_curves)>0 and show_negligible == True: 
            lines = []
            for curve_label in negligible_curves:
                line,= ax.plot([], [], ' ', label=curve_label)
                lines.append(line)
                
            negligible_legend=ax.legend(handles=lines, loc='best',bbox_to_anchor=(1, 1),title=fr"$B(N\to X) < {ylims[0]:.1f}$")   
            
            ax.add_artist(negligible_legend)
                
         
       
        

        
        
        if savefig == True: fig.savefig(filename,quality =100,bbox_inches='tight')
            
        return fig,ax
                  
            
    def save_data(self,save_gamma,save_ctau,save_brs):
        
        """
        
        Save decay data
        
        """
        
        if save_gamma: 
            
            #clean decay width directory
            for channel in self.modes_active.keys():
            
            

                self.gamma_path = f"Decay Data/{(self.U['e'],self.U['mu'],self.U['tau'])}/gamma"

                channel_path_gamma = os.path.join(self.gamma_path,channel)

                os.makedirs(channel_path_gamma,exist_ok = True)

                try:
                    for f in os.listdir(channel_path_gamma):
                        os.remove(os.path.join(channel_path_gamma, f))

                except:

                    pass
  
                
            #save decay widths
            for channel in self.model_widths.keys():
                
                for mode in self.model_widths[channel]:
                    
                    gamma_pts = self.model_widths[channel][mode]
                    
                    df_data = {'m': mpts,'gamma':gamma_pts}


                    df=pd.DataFrame(df_data)
                    
                    channel_path_gamma = os.path.join(self.gamma_path,channel)
                    
                    
                    save_path = os.path.join(channel_path_gamma,f'{mode}.csv')

                    df.to_csv(save_path,sep=' ',header=False,index=False)
    
        if save_ctau: 
            
            #save ctau 
            ctau_pts = self.ctau
            
            df_data = {'m': mpts,'ctau':ctau_pts}


            df=pd.DataFrame(df_data)

            save_path = f"Decay Data/{(self.U['e'],self.U['mu'],self.U['tau'])}/ctau.txt"

            df.to_csv(save_path,sep=' ',header=False,index=False)


        if save_brs: 
        
            #clean br directories 
            for channel in self.modes_active.keys():

                

                self.br_path = f"Decay Data/{(self.U['e'],self.U['mu'],self.U['tau'])}/br"

                channel_path_br = os.path.join(self.br_path,channel)

                os.makedirs(channel_path_br,exist_ok = True)
                try:
                    for f in os.listdir(channel_path_br):
                        os.remove(os.path.join(channel_path_br, f))


                except:
                    pass
           
             
            #save branching ratios
            for channel in self.model_brs.keys():
                
                for mode in self.model_brs[channel]:
                    
                    br_pts = self.model_brs[channel][mode]
                    
                    df_data = {'m': mpts,'br':br_pts}


                    df=pd.DataFrame(df_data)
                    
                    channel_path_br = os.path.join(self.br_path,channel)
                    
                    
                    save_path = os.path.join(channel_path_br,f'{mode}.csv')

                    df.to_csv(save_path,sep=' ',header=False,index=False)


                
       
        
        
        
        