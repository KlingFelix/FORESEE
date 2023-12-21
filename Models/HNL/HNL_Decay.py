import numpy as np
import sys,math
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from cycler import cycler

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
    
    if x == 'rho+' or anti(x) == 'rho+': return 0.220 
    elif x == 'K+*' or anti(x) =='K+*': return 0.217
    elif x == 'rho0' or anti(x) =='rho0': return 0.220
    elif x == 'omega' or anti(x) =='omega': return 0.195
    elif x =='pi+' or anti(x) =='pi+': return  0.1307
    elif x =='K+' or anti(x) =='K+': return 0.1598
    elif x =='pi0' or anti(x) =='pi0': return 0.130
    elif x =='eta' or anti(x) =='eta': return 0.1647
    elif x =='eta_p' or anti(x) =='eta_p': return 0.1529
           
    

    
    
    
"""

Kinematic Functions

"""

def Lambda(x,y,z):
    
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*x*z

def I1(x,y,z):
    integrand = lambda s: (1/s) * (s - x**2 - y**2) * (1 + z**2 - s)* np.sqrt(Lambda(s,x**2,y**2)) * np.sqrt(Lambda(1,s,z**2))
    
    integral,error = integrate.quad(integrand,(x+y)**2,(1-z)**2)
    
    result = 12*integral
    return result

def I2(x,y,z):
    integrand = lambda s: (1/s) * (1 + x**2 - s)* np.sqrt(Lambda(s,y**2,z**2)) * np.sqrt(Lambda(1,s,x**2))
    
    integral,error = integrate.quad(integrand,(y+z)**2,(1-x)**2)
    
    result = 24*y*z*integral
    return result

def F_P(x,y):
    
    return np.sqrt(Lambda(1,x**2,y**2)) * ((1+x**2)*(1+x**2-y**2) - 4*x**2)

def F_V(x,y):
    
    return np.sqrt(Lambda(1,x**2,y**2)) * ((1-x**2)**2 + (1+x**2) * y**2 - 2*y**4) 

def delta(l1,l2):
    
    if l1 == l2 or l1 == -1*l2:
        return 1
    else: 
        return 0   

def F_P(x,y):
    return ((1+x**2)*(1+x**2-y**2) - 4*x**2)*np.sqrt(Lambda(1,x**2,y**2))

def F_V(x,y):
    return ((1-x**2)**2 + (y**2)*(1+x**2) - 2*y**4)*np.sqrt(Lambda(1,x**2,y**2))
 
       
    
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
    
    #check coupling
    if l == 'e' or l == anti('e'): U = self.couplings[0]
    elif l == 'mu' or l == anti('mu'): U = self.couplings[1]   
    elif l == 'tau' or l == anti('tau'): U = self.couplings[2]

    prefactor = U**2 * Gf**2 * m**3 * f(P)**2 * V_dict[P]**2 / (16*np.pi) 

    yl1 = FORESEE.masses(pid(l))/m

    yP = FORESEE.masses(pid(P))/m


    #evaluate mass thresholds
    if cutoff == None:

        if 1 >= yl1 + yP:
            return prefactor*F_P(yl1,yP)
        else:
            return 0
    else:

        if 1 >= yl1 + yP and m <= cutoff:
            return prefactor*F_P(yl1,yP)
        else:
            return 0 

#N -> nu_alpha P 
def Gamma_nuP(self,m,mode,cutoff=Quark_level_cutoff):

    nu,P  = mode

    #check couplings
    if nu == 've' or nu == anti('ve'): U = self.couplings[0]
    elif nu == 'vmu' or nu == anti('vmu'): U = self.couplings[1]   
    elif nu == 'vtau' or nu == anti('vtau'): U = self.couplings[2]


    prefactor =   U**2 * Gf**2 * m**3 * f(P)**2 / (64*np.pi)

    yP = FORESEE.masses(pid(P))/m

    #evaluate mass thresholds
    if cutoff == None:
        if 1 >= yP:
            return prefactor*(1-yP**2)**2
        else:
            return 0 
    else: 
        if 1>= yP and m <= cutoff:
            return prefactor*(1-yP**2)**2
        else:
            return 0 

#N -> l_alpha V 
def Gamma_lV(self,m,mode,cutoff=Quark_level_cutoff):

    l,V = mode
    
    #check coupling
    if l == 'e' or l == anti('e'): U = self.couplings[0]
    elif l == 'mu' or l == anti('mu'): U = self.couplings[1]   
    elif l == 'tau' or l == anti('tau'): U = self.couplings[2]


    V_dict = {'rho+': Vckm['ud'],
              'anti_rho+': Vckm['ud'],
              'K+*': Vckm['us'],
              'anti_K+*': Vckm['us'],
             }

    prefactor =   U**2*Gf**2 * m**3 * f(V)**2 * V_dict[V]**2 / (16*np.pi) 


    yl1 = FORESEE.masses(pid(l))/m

    yV = FORESEE.masses(pid(V))/m
    
    #evaluate mass threshold
    if cutoff == None:
        if 1 >= yV+yl1:
            return prefactor*F_V(yl1,yV)
        else:
            return 0   

    else:
        if 1 >= yV+yl1 and m<=cutoff:
            return prefactor*F_V(yl1,yV)
        else:
            return 0  

#N -> nu_alpha V 
def Gamma_nuV(self,m,mode,cutoff=Quark_level_cutoff):

    nu,V = mode

    #check couplings
    if nu == 've' or nu == anti('ve'): U = self.couplings[0]
    elif nu == 'vmu' or nu == anti('vmu'): U = self.couplings[1]   
    elif nu == 'vtau' or nu == anti('vtau'): U = self.couplings[2]

    k_V = {'rho0': sin2w/3, 
           'anti_rho0': sin2w/3,
           'omega': sin2w/3,
           'anti_omega': sin2w/3,
          }

    prefactor = U**2*Gf**2 * m**3 * f(V)**2 * k_V[V]**2 / (2*np.pi)

    yV = FORESEE.masses(pid(V))/m

    #evaluate mass thresholds
    if cutoff == None:
        if 1 >= yV:
            return prefactor*(1-yV**2)**2 * (1+2*yV**2)
        else:
            return 0   

    else:
        if 1 >= yV and m<=cutoff:
            return prefactor*(1-yV**2)**2 * (1+2*yV**2)
        else:
            return 0  


#N -> l_alpha l_beta nu_beta
def Gamma_llnu(self,m,mode,cutoff=None):

    l1,l2,nu = mode

    #check couplings
    if l1 == 'e' or l1 == anti('e'): U = self.couplings[0]
    elif l1 == 'mu' or l1 == anti('mu'): U = self.couplings[1]   
    elif l1 == 'tau' or l1 == anti('tau'): U = self.couplings[2]
        
    yl1 = FORESEE.masses(pid(l1))/m
    yl2 = FORESEE.masses(pid(l2))/m

    ynu = 0

    #evaluate mass thresholds 
    if 1 >= yl1 + yl2:
        return U**2*Gf**2 * m**5 * I1(yl1,ynu,yl2) *(1-delta(int(pid(l1)),int(pid(l2)))) / (192*np.pi**3)
    else:
        return 0   

#N -> nu_alpha l_beta l_beta
def Gamma_null(self,m,mode,cutoff=None):
    nu,l1,l2 = mode

    #check couplings
    if nu == 've' or nu == anti('ve'): U = self.couplings[0]
    elif nu == 'vmu' or nu == anti('vmu'): U = self.couplings[1]   
    elif nu == 'vtau' or nu == anti('vtau'): U = self.couplings[2]

    yl1 = FORESEE.masses(pid(l1))/m
    yl2 = FORESEE.masses(pid(l2))/m

    ynu = 0


    prefactor = U**2*Gf**2 * m**5 / (96*np.pi**3)

    term1 =  (glL*glR + delta(abs(int(pid(nu)))-1,abs(int(pid(l2))))*glR)*I2(ynu,yl2,yl2) 

    term2 = (glL**2 + glR**2 + delta(abs(int(pid(nu)))-1,abs(int(pid(l2))))*(1+2*glL))*I1(ynu,yl2,yl2)

    #evaluate mass thresholds
    if 1 >= yl1 + yl2:
        return prefactor*(term1 + term2)
    else:
        return 0   

#N -> nu_alpha nu nu 
def Gamma_nu3(self,m,mode,cutoff=None):
    nu,_,_ = mode

    #check couplings
    if nu == 've' or nu == anti('ve'): U = self.couplings[0]
    elif nu == 'vmu' or nu == anti('vmu'): U = self.couplings[1]   
    elif nu == 'vtau' or nu == anti('vtau'): U = self.couplings[2]

    gamma = U**2*Gf**2 * m**5 / (96*np.pi**3)

    return gamma




#N -> l_alpha u d
def Gamma_lud(self,m,mode,cutoff=Quark_level_cutoff):

    l,u,d = mode

    #check couplings
    if l == 'e' or l == anti('e'): U = self.couplings[0]
    elif l == 'mu' or l == anti('mu'): U = self.couplings[1]   
    elif l == 'tau' or l == anti('tau'): U = self.couplings[2]


    yu = FORESEE.masses(pid(u))/m
    yd = FORESEE.masses(pid(d))/m
    yl = FORESEE.masses(pid(l))/m

    
    if 'anti' in d: V = Vckm[u+anti(d)]
    else: V = Vckm[anti(u)+d]

    #evaluate mass thresholds
    if cutoff == None:
        if 1 >= yu+yd+yl:
            return U**2*Gf**2 * V**2 * m**5 * I1(yl,yu,yd) / (64 * np.pi**3)
        else:
            return 0

    else:
        if 1 >= yu+yd+yl and m>=cutoff:
            return U**2*Gf**2 * V**2 * m**5 * I1(yl,yu,yd) / (64 * np.pi**3)
        else:
            return 0  


#N -> nu_alpha q q
def Gamma_nuqq(self,m,mode,cutoff=Quark_level_cutoff):

    nu,q,qbar = mode 

    #check couplings
    if nu == 've' or nu == anti('ve'): U = self.couplings[0]
    elif nu == 'vmu' or nu == anti('vmu'): U = self.couplings[1]   
    elif nu == 'vtau' or nu == anti('vtau'): U = self.couplings[2]

    yq = FORESEE.masses(pid(q))/m

    ynu = 0 


    prefactor = U**2 * Gf**2 * m**5  / (32*np.pi**3)
    
    #pick out coupling constants
    if q  in self.particle_content['quarks']['up'] or qbar in self.particle_content['quarks']['up']:
        gqL = guL
        gqR = guR
    elif q  in self.particle_content['quarks']['down'] or qbar in self.particle_content['quarks']['down']:
        gqL = gdL
        gqR = gdR


    term1 = gqL*gqR*I2(ynu,yq,yq)

    term2 = (gqL**2 + gqR**2)*I1(ynu,yq,yq)

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
        
        self.couplings = couplings
        

        #define particle content
        leptons = ['e','mu','tau']

        neutrinos = ['ve','vmu','vtau']

        vectors = {'charged':['rho+','K+*'], 'neutral': ['rho0','omega'] }

        pseudos = {'charged':['pi+','K+'], 'neutral':['pi0','eta'] }
        
        quarks = {'up':['u','c','t'], 'down': ['d','s','b']}

        self.particle_content = {'leptons':leptons,'neutrinos':neutrinos,'vectors':vectors,'pseudos':pseudos,'quarks':quarks}



        ##Compile all allowed decay modes for each decay channel##
        
        #N -> nu_alpha l_beta l_beta
        null = []

        for nu in neutrinos: 

            for l in leptons: 

                mode = (nu,anti(l),l)

                null.append(mode)
        
        #N -> l_alpha l_beta nu
        llnu = [] 

        for l1 in leptons: 

            for flavor, l2 in enumerate(leptons): 
                
                if l1 != l2: 

                    mode = (l1, anti(l2), neutrinos[flavor])

                    llnu.append(mode)
                    
                    #conjugate mode
                    llnu.append((anti(l1),l2,neutrinos[flavor]))
        
        #N -> nu_alpha nu nu 
        nu3 = []

        for nu1 in neutrinos:

            mode = (nu1,'nu','nu')
            nu3.append(mode)
    
                
            
        #N -> nu_alpha P    
        nuP = []

        for nu in neutrinos:

            for P in pseudos['neutral']:

                mode = (nu,P)

                nuP.append(mode)
      
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

        for nu in neutrinos:

            for V in vectors['neutral']:

                mode = (nu,V)

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
        
        for nu in neutrinos: 
            
            for q in quarks['up'] + quarks['down']:
                
                mode = (nu,q,anti(q))
                
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
            
                p_coupled = mode[0]

                Ue,Umu,Utau = self.couplings


                if p_coupled in ['e',anti('e'),'ve', anti('ve')]:  U =  Ue

                elif p_coupled in ['mu',anti('mu'),'vmu', anti('vmu')]: U = Umu

                elif p_coupled in ['tau',anti('tau'),'vtau', anti('vtau')]:  U = Utau

                

                if U == 0:

                    modes_inactive.append(mode)
            
                else:

                    modes_active.append(mode)

                   


            
            
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
                
             
            
    def save_data(self,save_gamma,save_ctau,save_brs):
        
        """
        
        Save decay data
        
        """
        
        if save_gamma: 
            
            #clean decay width directory
            for channel in self.modes_active.keys():
            
            

                self.gamma_path = f"Decay Data/{self.couplings}/gamma"

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

            save_path = f"Decay Data/{self.couplings}/ctau.txt"

            df.to_csv(save_path,sep=' ',header=False,index=False)


        if save_brs: 
        
            #clean br directories 
            for channel in self.modes_active.keys():

                

                self.br_path = f"Decay Data/{self.couplings}/br"

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


                
       
        
        
        
        