import numpy as np
import sys,math,random
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scipy.integrate as integrate
from scipy.interpolate import interp1d

src_path = "../.."
sys.path.append(src_path)
from src.foresee import Foresee, Utility, Model

foresee = Foresee()
energy = "14"
modelname="HNL"
FORESEE = Model(modelname, path="././")

Quark_level_cutoff = 1.0


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
    
    if l1 == l2:
        return 1
    else: 
        return 0
    
    
def F_Z(mass): 
    
    #Need to Find actual function
    return 1.0

def F_P(x,y):
    return ((1+x**2)*(1+x**2-y**2) - 4*x**2)*np.sqrt(Lambda(1,x**2,y**2))

def F_V(x,y):
    return ((1-x**2)**2 + (y**2)*(1+x**2) - 2*y**4)*np.sqrt(Lambda(1,x**2,y**2))
        
    
def anti(pid):
    
    return tuple([-1*x for x in pid])




class DecayModes: 
    
    def __init__(self,couplings,pids='all',majorana=True):
        
        
        self.majorana = majorana
        
        self.pids = pids
        
        self.couplings = couplings
        
        self.pids_dict = {
            'lP': [(211,11),(211,13),(211,15),
                   (321,11),(321,13),(321,15)],
            'nuP': [(111,12),(111,14),(111,16),
                    (221,12),(221,14),(221,16),
                    (331,12),(331,14),(331,16),
                    (311,12),(311,14),(311,16)],
            
            'lV': [(213,11),(213,13),(213,15),
                   (323,11),(323,13),(323,15)],
            'nuV': [(113,12),(113,14),(113,16),
                    (223,12),(223,14),(223,16),
                    (313,12),(313,14),(313,16)],
            'llnu':[(11,11,12),(11,13,14),(11,15,16),
                    (13,11,12),(13,13,14),(13,15,16),
                    (15,11,12),(15,13,14),(15,15,16)],
            'null': [(12,11,11),(12,13,13),(12,15,15),
                     (14,11,11),(14,13,13),(14,15,15),
                     (16,11,11),(16,13,13),(16,15,15)],
            '3nu':[(12,'nu','nu'),(14,'nu','nu'),(16,'nu','nu')],
            'lud':[(2,-1,11),(2,-1,13),(2,-1,15),
                  (2,-3,11),(2,-3,13),(2,-3,15),
                  (2,-5,11),(2,-5,13),(2,-5,15),
                  (4,-1,11),(4,-1,13),(4,-1,15),
                  (4,-3,11),(4,-3,13),(4,-3,15),
                  (4,-5,11),(4,-5,13),(4,-5,15),
                  (6,-1,11),(6,-1,13),(6,-1,15),
                  (6,-3,11),(6,-3,13),(6,-3,15),
                  (6,-5,11),(6,-5,13),(6,-5,15)],
            'nuqq':[(12,1,-1),(14,1,-1),(16,1,-1),
                   (12,2,-2),(14,2,-2),(16,2,-2),
                   (12,3,-3),(14,3,-3),(16,3,-3),
                   (12,4,-4),(14,4,-4),(16,4,-4),
                   (12,5,-5),(14,5,-5),(16,5,-5),
                   (12,6,-6),(14,6,-6),(16,6,-6)],
        }
        self.pids_all_anti = []
        if majorana ==True:
            
            for channel in self.pids_dict.keys():
                for pid in self.pids_dict[channel]:
                    antipid = anti(pid)
                    self.pids_all_anti.append(antipid)

        
        
        
        self.pids_all = self.pids_dict['lP'] + self.pids_dict['nuP'] + self.pids_dict['lV'] + self.pids_dict['nuV'] + self.pids_dict['llnu'] + self.pids_dict['null'] + self.pids_dict['3nu'] + self.pids_dict['lud'] + self.pids_dict['nuqq']
        
        if pids =='all': self.pids = self.pids_all
        else: self.pids = pids
        
        self.Vckm = {'ud':0.97370 , 'us':0.2245, 'ub':3.82E-3 ,
             'cd': 0.221, 'cs':0.987, 'cb':41.0E-3 ,
             'td': 8.0E-3, 'ts':38.8E-3, 'tb':1.013}
        
        self.Gf = 1.166 * (10**-5)
        
        self.f_dict = {411: 0.2226, 421: 0.2226, 431: 0.2801,   #Charmed Mesons
                 311: 0.1598, 321: 0.1598,                      #Strange Mesons
                 111: 0.130, 211: 0.130,                        #Pions
                 521: 0.190, 541:.480,#Bottom Mesons
                 221: 0.1647, 331:0.1529,
                       
                 213:0.220, 323:0.217, 113:0.220,223:0.195,313:0.217,
                 }
        
        
    def Gamma_lP(self,m,mode,cutoff=Quark_level_cutoff): 
        
        pid_P,pid_l = mode
        
        V_dict = {211: self.Vckm['ud'],
                  321: self.Vckm['us']
        }
        
        if pid_l == 11 or pid_l == -11: U = self.couplings[0]
        elif pid_l == 13 or pid_l == -13: U = self.couplings[1]   
        elif pid_l == 15 or pid_l == -15: U = self.couplings[2]
        
        prefactor = U**2 * self.Gf**2 * m**3 * self.f_dict[abs(pid_P)]**2 * V_dict[abs(pid_P)]**2 / (16*np.pi) 
    
        yl1 = FORESEE.masses(str(pid_l))/m

        yP = FORESEE.masses(str(pid_P))/m
        
        
        
        
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

       
            
            
     
                    
        
    def Gamma_nuP(self,m,mode,cutoff=Quark_level_cutoff):
        
        pid_P,pid_l = mode
        
       
        
        if pid_l == 12 or pid_l == -12: U = self.couplings[0]
        elif pid_l == 14 or pid_l == -14: U = self.couplings[1]   
        elif pid_l == 16 or pid_l == -16: U = self.couplings[2]
        
        
        
        prefactor =   U**2 * self.Gf**2 * m**3 * self.f_dict[abs(pid_P)]**2 / (128*np.pi)
        
        yP = FORESEE.masses(str(pid_P))/m
        
        
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
        
        
        
        
        
    def Gamma_lV(self,m,mode,cutoff=Quark_level_cutoff):
        
        pid_V,pid_l = mode
         
        if pid_l == 11 or pid_l == -11: U = self.couplings[0]
        elif pid_l == 13 or pid_l == -13: U = self.couplings[1]   
        elif pid_l == 15 or pid_l == -15: U = self.couplings[2]
            
            
        V_dict = {213: self.Vckm['ud'],
                  323: self.Vckm['us']
        }
            
        prefactor =   U**2*self.Gf**2 * m**3 * self.f_dict[abs(pid_V)]**2 * V_dict[abs(pid_V)]**2 / (16*np.pi) 
            
        
        yl1 = FORESEE.masses(str(pid_l))/m

        yV = FORESEE.masses(str(pid_V))/m
        
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
        
        
    def Gamma_nuV(self,m,mode,cutoff=Quark_level_cutoff):
        
        pid_V,pid_l = mode
        
        
        
        if pid_l == 12 or pid_l == -12: U = self.couplings[0]
        elif pid_l == 14 or pid_l == -14: U = self.couplings[1]   
        elif pid_l == 16 or pid_l == -16: U = self.couplings[2]
        
        
        sin2w = 0.23121
        
        k_V = {113:sin2w/3,223:sin2w/3,313: (sin2w/3)-(1/4)}
        
        
        prefactor = U**2*self.Gf**2 * m**3 * self.f_dict[abs(pid_V)]**2 * k_V[abs(pid_V)]**2 / (4*np.pi)
        
        
        yV = FORESEE.masses(str(pid_V))/m
        
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
        
        
        
    def Gamma_llnu(self,m,mode,cutoff=None):
        
        pid_l1,pid_l2,pid_nu = mode
        
        if pid_l1 == 11 or pid_l1 == -11: U = self.couplings[0]
        elif pid_l1 == 13 or pid_l1 == -13: U = self.couplings[1]   
        elif pid_l1 == 15 or pid_l1 == -15: U = self.couplings[2]
        
        yl1 = FORESEE.masses(str(pid_l1))/m
        yl2 = FORESEE.masses(str(pid_l2))/m
        
        ynu = 0
        
        if cutoff == None:
            if 1 >= yl1 + yl2:
                return U**2*self.Gf**2 * m**5 * I1(yl1,ynu,yl2) *(1-delta(pid_l1,pid_l2)) / (192*np.pi**3)
            else:
                return 0   
            
        else:
            if 1 >= yl1+yl2 and m<=cutoff:
                return U**2*self.Gf**2 * m**5 * I1(yl1,ynu,yl2) *(1-delta(pid_l1,pid_l2)) / (192*np.pi**3)
            else:
                return 0  
        
        
        
        
        
    def Gamma_null(self,m,mode,cutoff=None):
        pid_nu,pid_l1,pid_l2 = mode
        
        if pid_nu == 12 or pid_nu == -12: U = self.couplings[0]
        elif pid_nu == 14 or pid_nu == -14: U = self.couplings[1]   
        elif pid_nu == 16 or pid_nu == -16: U = self.couplings[2]
        
        yl1 = FORESEE.masses(str(pid_l1))/m
        yl2 = FORESEE.masses(str(pid_l2))/m
        
        ynu = 0
        
        
        sin2w = 0.23121

        glL = -0.5 + sin2w

        glR = sin2w

        guL = 0.5 - (2*sin2w/3)

        guR = -(2*sin2w/3)

        gdL = -0.5 + (sin2w/3)

        gdR = (sin2w/3)

        
        
        prefactor = U**2*self.Gf**2 * m**5 / (192*np.pi**3)
        
        term1 =  (glL*glR + delta(abs(pid_nu)-1,abs(pid_l2))*glR)*I2(ynu,yl2,yl2) 
    
    
    
        term2 = (glL**2 + glR**2 + delta(abs(pid_nu)-1,abs(pid_l2))*(1+2*glL))*I1(ynu,yl2,yl2)
        
        if cutoff == None:
            if 1 >= yl1 + yl2:
                return prefactor*(term1 + term2)
            else:
                return 0   
            
        else:
            if 1 >= yl1+yl2 and m<=cutoff:
                return prefactor*(term1 + term2)
            else:
                return 0  
        
    def Gamma_3nu(self,m,mode,cutoff=None):
        pid_nu,_,_ = mode
        
        if pid_nu == 12 or pid_nu == -12: U = self.couplings[0]
        elif pid_nu == 14 or pid_nu == -14: U = self.couplings[1]   
        elif pid_nu == 16 or pid_nu == -16: U = self.couplings[2]
        
        gamma = U**2*self.Gf**2 * m**5 / (192*np.pi**3)
        
        if cutoff == None:
            if 1 >= 0:
                return gamma
            
            
        else:
            if 1 >= 0 and m<=cutoff:
                return gamma
            else:
                return 0  
        
        
        
        
        
    def Gamma_lud(self,m,mode,cutoff=Quark_level_cutoff):
        
        pid_u,pid_d,pid_l = mode
        
        if pid_l == 11 or pid_l == -11: U = self.couplings[0]
        elif pid_l == 13 or pid_l == -13: U = self.couplings[1]   
        elif pid_l == 15 or pid_l == -15: U = self.couplings[2]
            
            
        yu = FORESEE.masses(str(pid_u))/m
        yd = FORESEE.masses(str(pid_d))/m
        yl = FORESEE.masses(str(pid_l))/m
        
        quark_label = {1:'d',2:'u',3:'s',4:'c',5:'b',6:'t'}
        
        V = self.Vckm[quark_label[abs(pid_u)] + quark_label[abs(pid_d)]]
        
        if cutoff == None:
            if 1 >= yu+yd+yl:
                return U**2*self.Gf**2 * V**2 * m**5 * I1(yl,yu,yd) / (64 * np.pi**3)
            
            
        else:
            if 1 >= yu+yd+yl and m>=cutoff:
                return U**2*self.Gf**2 * V**2 * m**5 * I1(yl,yu,yd) / (64 * np.pi**3)
            else:
                return 0  
        
        
        
    def Gamma_nuqq(self,m,mode,cutoff=Quark_level_cutoff):
        
        pid_nu,pid_q,pid_qbar = mode 
        
        if pid_nu == 12 or pid_nu == -12: U = self.couplings[0]
        elif pid_nu == 14 or pid_nu == -14: U = self.couplings[1]   
        elif pid_nu == 16 or pid_nu == -16: U = self.couplings[2]
        
        yq = FORESEE.masses(str(pid_q))/m
        
        ynu = 0 
        
        sin2w = 0.23121

        glL = -0.5 + sin2w

        glR = sin2w

        guL = 0.5 - (2*sin2w/3)

        guR = -(2*sin2w/3)

        gdL = -0.5 + (sin2w/3)

        gdR = (sin2w/3)


        prefactor = U**2 * self.Gf**2 * m**5 * F_Z(m) / (32*np.pi**3)
        
        if abs(pid_q)  in [2,4,6]:
            gqL = guL
            gqR = guR
        elif abs(pid_q)  in [1,3,5]:
            gqL = gdL
            gqR = gdR
        
        
        term1 = gqL*gqR*I2(ynu,yq,yq)

        term2 = (gqL**2 + gqR**2)*I1(ynu,yq,yq)
        
        
        if cutoff == None:
            if 1 >= yq+yq:
                return prefactor*(term1 + term2)
            
            
        else:
            if 1 >= yq+yq and m>=cutoff:
                return prefactor*(term1 + term2)
            else:
                return 0  
        
        
        
    def gen_gamma_csv(self,mpts,N,pids = 'all'):
        
        if pids =='all': pids = self.pids_all 
            
        for mode in pids: 
            
            if len(mode) == 2: pid1,pid2 = mode
                
            elif len(mode) == 3: pid1,pid2,pid3 = mode
            
            
            for i in self.pids_dict.keys():
                
                if mode in self.pids_dict[i] or anti(mode) in self.pids_dict[i]:
                    
                    channel = i 
            
            
            
            
            gammapts = eval(f"[self.Gamma_{channel}(m,mode) for m in mpts]",{'self':self,'mode':mode,'couplings':self.couplings,'mpts':mpts},{})
            
            df_data = {'m': mpts,'Gamma':gammapts}


            df=pd.DataFrame(df_data)

            
            if len(mode) == 2: 
                
                pid1,pid2 = mode
                df.to_csv(fr"Decay Widths/{pid1}_{pid2}.csv",sep=' ',header=False,index=False)
                
                if self.majorana == True:
                    df.to_csv(fr"Decay Widths/{pid1*-1}_{pid2*-1}.csv",sep=' ',header=False,index=False)
                
            elif len(mode) == 3: 
                
                pid1,pid2,pid3 = mode
                df.to_csv(fr"Decay Widths/{pid1}_{pid2}_{pid3}.csv",sep=' ',header=False,index=False)
                if self.majorana == True:
                    df.to_csv(fr"Decay Widths/{pid1*-1}_{pid2*-1}_{pid3*-1}.csv",sep=' ',header=False,index=False)

    
    
   
        
    def plotter(self,pids='all'):
        
        if pids =='all': pids = self.pids_all + self.pids_all_anti
        
        fig,ax = plt.subplots()
        
        ax.set(xscale='log',yscale='log',xlabel=r'$m_N$ (GeV)', ylabel=r'$\Gamma \,/\,{U_{\alpha}}^2$ (GeV)', title = fr"$\Gamma\,(\,N \to \,\, l_\alpha^- \, P^+ )$")

        
        for pid in pids: 
            
            if len(pid) == 2: 
                pid1,pid2 = pid
                
                data = pd.read_csv(fr"Decay Widths/{pid1}_{pid2}.csv",sep=' ',header=None)
            
            elif len(pid) == 3: 
                pid1,pid2,pid3 = pid
                data = pd.read_csv(fr"Decay Widths/{pid1}_{pid2}_{pid3}.csv",sep=' ',header=None)
            
        
            mpts = data[0]
            gammapts = data[1]
            
            ax.plot(mpts,gammapts,label=fr"{pid}")
            
        ax.legend()
        
    def get_ctau(self,pids='all'):
        
        GeVtoS = (6.58 * 10**-25)

        c = 299792458 #m/s
        
        
        if pids =='all': pids = self.pids_all + self.pids_all_anti
        
        gammas = [] 
        for pid in pids: 
            
            if len(pid) == 2: 
                pid1,pid2 = pid
                
                data = pd.read_csv(fr"Decay Widths/{pid1}_{pid2}.csv",sep=' ',header=None)
                
                gammapts = list(data[1])
                
                mpts = list(data[0])
            elif len(pid) == 3: 
                pid1,pid2,pid3 = pid
                data = pd.read_csv(fr"Decay Widths/{pid1}_{pid2}_{pid3}.csv",sep=' ',header=None)
                gammapts = list(data[1])
                mpts = list(data[0])
                
            gammas.append(gammapts)
        gamma_t = [] 
        
        
        for i in range(len(gammas[0])):
            
            gamma_t_m = 0 
            
            for gamma_mode in gammas: 
                gamma_t_m += gamma_mode[i]
            
            
            gamma_t.append(gamma_t_m)
        
            
        ctau = [(c/g)*GeVtoS for g in gamma_t]
        
        df_data = {'m': mpts,'Gamma':ctau}
        df=pd.DataFrame(df_data)
        df.to_csv(fr"ctau.txt",sep=' ',header=False,index=False)
        
   
        
