import numpy as np
from src.foresee import Utility, Foresee

class HeavyNeutralLepton(Utility):
    
    ###############################
    #  Initiate
    ###############################
    
    def __init__(self, ve=1, vmu=0, vtau=0):
        self.vcoupling = {"11": ve, "13":vmu, "15": vtau}
        self.lepton = {"11": "e", "13":"\mu", "15": "\tau"}

    ###############################
    #  2-body decays
    ###############################
    
    #decay constants
    def fH(self,pid):
        if   pid in ["211","-211"]: return 0.130
        elif pid in ["321","-321"]: return 0.1598
        elif pid in ["411","-411"]: return 0.2226
        elif pid in ["431","-431"]: return 0.2801
        elif pid in ["521","-521"]: return 0.187
        elif pid in ["541","-541"]: return 0.434
        
    # Lifetimes
    def tau(self,pid):
        if   pid in ["2112","-2112"]: return 10**8
        elif pid in ["15","-15"    ]: return 290.1*1e-15
        elif pid in ["2212","-2212"]: return 10**8
        elif pid in ["211","-211"  ]: return 2.603*10**-8
        elif pid in ["323","-323"  ]: return 1.2380*10**-8
        elif pid in ["321","-321"  ]: return 1.2380*10**-8
        elif pid in ["411","-411"  ]: return 1040*10**-15
        elif pid in ["421","-421"  ]: return 410*10**-15
        elif pid in ["423", "-423" ]: return 3.1*10**-22
        elif pid in ["431", "-431" ]: return 504*10**-15
        elif pid in ["511", "-511" ]: return 1.519*10**-12
        elif pid in ["521", "-521" ]: return 1.638*10**-12
        elif pid in ["531", "-531" ]: return 1.515*10**-12
        elif pid in ["541", "-541" ]: return 0.507*10**-12
        elif pid in ["310"         ]: return 8.954*10**-11
        elif pid in ["130"         ]: return 5.116*10**-8
        elif pid in ["3122","-3122"]: return 2.60*10**-10
        elif pid in ["3222","-3222"]: return 8.018*10**-11
        elif pid in ["3112","-3112"]: return 1.479*10**-10
        elif pid in ["3322","-3322"]: return 2.90*10**-10
        elif pid in ["3312","-3312"]: return 1.639*10**-10
        elif pid in ["3334","-3334"]: return 8.21*10**-11
        
    # CKM matrix elements
    def VH(self,pid):
        if   pid in ["211","-211"]: return 0.97370 #Vud
        elif pid in ["321","-321"]: return 0.2245 #Vus
        elif pid in ["213","-213"]: return 0.97370
        elif pid in ["411","-411"]: return 0.221
        elif pid in ["431","-431"]: return 0.987
        elif pid in ["521","-521"]: return 3.82*10**-3
        elif pid in ["541","-541"]: return 41*10**-3
    
    # Branching fraction
    def get_2body_decay(self,pid0,pid1):
        
        #read constant
        mH, mLep, tauH = self.masses(pid0), self.masses(pid1), self.tau(pid0)
        vH, fH = self.VH(pid0), self.fH(pid0)
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)
        
        #calculate rate
        prefactor=(tauH*GF**2*fH**2*vH**2)/(8*np.pi)
        prefactor*=self.vcoupling[str(abs(int(pid1)))]**2
        br=str(prefactor)+"*coupling**2*mass**2*"+str(mH)+"*(1.-(mass/"+str(mH)+")**2 + 2.*("+str(mLep)+"/"+str(mH)+")**2 + ("+str(mLep)+"/mass)**2*(1.-("+str(mLep)+"/"+str(mH)+")**2)) * np.sqrt((1.+(mass/"+str(mH)+")**2 - ("+str(mLep)+"/"+str(mH)+")**2)**2-4.*(mass/"+str(mH)+")**2)"
        return br
        
    ###############################
    #  3-body decays
    ###############################
        
    #VHH in 3-body decays - CKM matrix elements
    def VHHp(self,pid0,pid1):
        if   pid0 in ['411','-411'] and pid1 in ['311','-311']: return 0.987
        elif pid0 in ['421','-421'] and pid1 in ['321','-321']: return 0.997
        elif pid0 in ['521','-521'] and pid1 in ['421','-421']: return 41*10**-3
        elif pid0 in ['511','-511'] and pid1 in ['411','-411']: return 42.2*10**-3
        elif pid0 in ['531','-531'] and pid1 in ['431','-431']: return 42.2*10**-3
        elif pid0 in ['541','-541'] and pid1 in ['511','-511']: return 0.221
        elif pid0 in ['541','-541'] and pid1 in ['531','-531']: return 0.987
        elif pid0 in ['421','-421'] and pid1 in ['323','-323']: return 0.967
        elif pid0 in ['521','-521'] and pid1 in ['423','-423']: return 41*10**-3
        elif pid0 in ['511','-511'] and pid1 in ['413','-413']: return 41*10**-3
        elif pid0 in ['531','-531'] and pid1 in ['433','-433']: return 41*10**-3
        elif pid0 in ['541','-541'] and pid1 in ['513','-513']: return 0.221
        elif pid0 in ['541','-541'] and pid1 in ['533','-533']: return 41*10**-3
    

    def get_3body_decay_pseudoscalar(self,pid0,pid1,pid2):
            
        # read constant
        mH, mHp, mLep = self.masses(pid0), self.masses(pid1), self.masses(pid2)
        VHHp, tauH = self.VHHp(pid0,pid1), self.tau(pid0)
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)
        
        # prefactor
        prefactor=tauH*VHHp**2*GF**2/(64*np.pi**3*mH**2)
        prefactor*=self.vcoupling[str(abs(int(pid2)))]**2
        
        # form factors
        if pid0 in ["411","421","431","-411","-421","-431"]:
            f00, MV, MS = .747, 2.01027, 2.318
        if pid0 in ["511","521","-511","-521"]:
            f00, MV, MS = 0.66, 6.400, 6.330
        if pid0 in ["531","-531"]:
            f00, MV, MS = 0.57, 6.400, 6.330
        if pid0 in ["541","-541"] and pid1 in ["511","-511"]:
            f00, MV, MS = -0.58, 6.400, 6.330
        if pid0 in ["541","-541"] and pid1 in ["531","-531"]:
            f00, MV, MS = -0.61, 6.400, 6.330
        fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
        f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        fm="("+f0+"-"+fp+")*("+str(mH)+"**2-"+str(mHp)+"**2)/q**2"
        term1="("+fm+")**2*(q**2*(mass**2+"+str(mLep)+"**2)-(mass**2-"+str(mLep)+"**2)**2)"
        term2=f"2*("+fp+")*("+fm+")*mass**2*(2*"+str(mH)+"**2-2*"+str(mHp)+"**2-4*energy*"+str(mH)+"-"+str(mLep)+"**2+mass**2+q**2)"
        term3=f"(2*("+fp+")*("+fm+")*"+str(mLep)+"**2*(4*energy*"+str(mH)+"+ "+str(mLep)+"**2-mass**2-q**2))"
        term4=f"("+fp+")**2*(4*energy*"+str(mH)+"+"+str(mLep)+"**2-mass**2-q**2)*(2*"+str(mH)+"**2-2*"+str(mHp)+"**2-4*energy*"+str(mH)+"-"+str(mLep)+"**2+mass**2+q**2)"
        term5=f"-("+fp+")**2*(2*"+str(mH)+"**2+2*"+str(mHp)+"**2-q**2)*(q**2-mass**2-"+str(mLep)+"**2)"
        bra=str(prefactor)  + "* coupling**2 *(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
        return(bra)


    
