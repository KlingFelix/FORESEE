import numpy as np
from src.foresee import Utility, Foresee

class HeavyNeutralLepton(Utility):

    ###############################
    #  Initiate
    ###############################

    def __init__(self, ve=1, vmu=0, vtau=0):
        self.vcoupling = {"11": ve, "13":vmu, "15": vtau}
        self.lepton = {"11": "e", "13":"mu", "15": "tau"}
        self.hadron = {"211": "pi", "321": "K", "213": "rho"}

    ###############################
    #  2-body decays
    ###############################
    
    #decay constants
    def fH(self,pid):
        if   pid in ["211","-211","111"]: return 0.130
        elif pid in ["221","-221"]: return 1.2*0.130
        elif pid in ["313","-313","323","-323"]: return 0.204      #for K*, need to see which values we should use, these seemed pretty good though
        elif pid in ["321","-321"]: return 0.1598
        elif pid in ["331","-331"]: return -0.45*0.130
        elif pid in ["421","411","-411"]: return 0.2226 
        elif pid in ["423","-423"]: return 0.2235        #D*0, value is pretty close to D, so this is probably right
        elif pid in ["431","-431"]: return 0.2801
        elif pid in ["511", "511","521","-521"]: return 0.190
        elif pid in ["531","-531"]: return 0.230   #not sure if neutral has same as charged
        elif pid in ["541","-541"]: return 0.480

    # Lifetimes
    def tau(self,pid):
        if   pid in ["2112","-2112"]: return 10**8
        elif pid in ["15","-15"    ]: return 290.1*1e-15
        elif pid in ["2212","-2212"]: return 10**8
        elif pid in ["211","-211"  ]: return 2.603*10**-8
        elif pid in ["323","-323"  ]: return 1.2380*10**-8   #### WRONG
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
        elif pid in ["321","-321","323","-323"]: return 0.2245 #Vus
        elif pid in ["213","-213"]: return 0.97370
        elif pid in ["411","-411"]: return 0.221
        elif pid in ["431","-431"]: return 0.987
        elif pid in ["521","-521"]: return 3.82*10**-3
        elif pid in ["541","-541"]: return 41*10**-3
        elif pid in []: return 0.97370 #Vud
        elif pid in []: return 0.2245 #Vus
        elif pid in ["411","-411"]: return 0.221 #Vcd
        elif pid in []: return 0.987 #Vcs
        elif pid in ["541","-541"]: return 41*10**-3 #Vcb
        elif pid in ["521","-521"]: return 3.82*10**-3 #Vub
        elif pid in []: return 8*10**-3 #Vtd
        elif pid in []: return 38.8*10**-3 #Vts
        elif pid in []: return 1.013 #Vtb

    #for HNL decays to neutral vector mesons
    def kV(self,pid):
        xw=0.231
        if pid in ["313","-313"]: return (-1/4+(1/3)*xw)
        elif pid in ["423","-423","443"]: return (1/4-(2/3)*xw)

    # Branching fraction
    def get_2body_br(self,pid0,pid1):

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

    def get_2body_br_tau(self,pid0,pid1):
        if pid1 in ['213','-213']:
            grho, VH, tautau = 0.102, self.VH(pid1), self.tau(pid0)
            Mtau, Mrho=self.masses(pid0), self.masses(pid1)
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=tautau*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=tautau*grho**2*GF**2*VH**2*Mtau**3/(8*np.pi*Mrho**2)
            prefactor*=self.vcoupling[str(abs(int(pid0)))]**2
            br=f"{prefactor}*coupling**2*((1-(mass**2/self.masses('{pid0}')**2))**2+(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+((mass**2-2*self.masses('{pid1}')**2)/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2))*(1-((self.masses('{pid1}')+mass)**2/self.masses('{pid0}')**2)))"
        else:
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)
            tautau=tautau*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            VH=self.VH(pid1)
            fH=self.fH(pid1)
            Mtau=self.masses(pid0)
            prefactor=tautau*GF**2*VH**2*fH**2*Mtau**3/(16*np.pi)
            prefactor*=self.vcoupling[str(abs(int(pid0)))]**2
            br=f"{prefactor}*coupling**2*((1-(mass**2/self.masses('{pid0}')**2))**2-(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+(mass**2/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid1}')+mass)**2/self.masses('{pid0}')**2))))"
        return (br)

    ###############################
    #  3-body decays
    ###############################

    #VHH in 3-body decays - CKM matrix elements
    def VHHp(self,pid0,pid1):
        if   pid0 in ["2","-2"    ] and pid1 in ["1","-1"    ]: return 0.97370
        elif pid0 in ['411','-411'] and pid1 in ['311','-311']: return 0.987
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


    def get_3body_dbr_pseudoscalar(self,pid0,pid1,pid2):

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
        '''fp=f"{f00}/(1-q**2/{MV}**2)"
        f0=f"{f00}/(1-q**2/{MS}**2)"
        fm=f"({f0}-{fp})*(self.masses('{pid0}')**2-self.masses('{pid1}')**2)/q**2"
        term1=f"({fm})**2*(q**2*(m3**2+self.masses('{pid2}')**2)-(m3**2-self.masses('{pid2}')**2)**2)"
        term2=f"2*({fp})*({fm})*m3**2*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
        term3=f"(2*({fp})*({fm})*self.masses('{pid2}')**2*(4*EN*self.masses('{pid0}')+ self.masses('{pid2}')**2-m3**2-q**2))"
        term4=f"({fp})**2*(4*EN*self.masses('{pid0}')+self.masses('{pid2}')**2-m3**2-q**2)*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
        term5=f"-({fp})**2*(2*self.masses('{pid0}')**2+2*self.masses('{pid1}')**2-q**2)*(q**2-m3**2-self.masses('{pid2}')**2)"
        bra=str(prefactor)  + "*(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
        return(bra)'''
        return(bra)

    def get_3body_dbr_vector(self,pid0,pid1,pid2):
        #SecToGev=1./(6.58*pow(10.,-25.))
        #tauH=1.638*10**-12
        #tauH=tauH*SecToGev
        #GF=1.166378*10**(-5) #GeV^(-2)
        #VHV=41*10**-3 #Vcs matrix element
        #tauH=df.loc[df['pid0']==pid0]['tauH (sec)'].values[0]
        #tauH=tauH*SecToGev
        #VHV=df.loc[df['pid0']==pid0]['VHHp'].values[0]


        tauH=self.tau(pid0)
        SecToGev=1./(6.58*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.1663787*10**(-5)
        VHV=self.VHHp(pid0,pid1)
        #'D^0 -> K*^- + e^+ + N'
        if pid0 in ['421'] and pid1 in ['323','-323']:
            A00=.76; Mp=1.97; s1A0=.17; s2A0=0; V0=1.03; MV=2.11; s1V=.27; s2V=0; A10=.66; s1A1=.3
            s2A1=.2*0; A20=.49; s1A2=.67; s2A2=.16*0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^+ -> \bar{D}*^0 + e^+ + N' or 'B^0 -> D*^- + e^+ + N'
        if (pid0 in ['521','-521'] and pid1 in ['423','-423']) or (pid0 in ['511'] and pid1 in ['413','-413']):
            A00=0.69; Mp=6.277; s1A0=0.58; s2A0=0; V0=0.76; MV=6.842; s1V=0.57; s2V=0; A10=0.66; s1A1=0.78
            s2A1=0; A20=0.62; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^0_s -> D^*_s^- + e^+ + N'
        if pid0 in ['531'] and pid1 in ['433','-433']:
            A00=0.67; Mp=6.842; s1A0=0.35; s2A0=0; V0=0.95; MV=6.842; s1V=0.372
            s2V=0; A10=0.70; s1A1=0.463; s2A1=0; A20=0.75; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #'B^+_c -> B*^0 + e^+ + N'
        if pid0 in ['541','-541'] and pid1 in ['513','-513']:
            A00=-.27; mfitA0=1.86; deltaA0=.13; V0=3.27; mfitV=1.76; deltaV=-.052
            A10=.6; mfitA1=3.44; deltaA1=-1.07; A20=10.8; mfitA2=1.73; deltaA2=0.09
            A0=f"{A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2)"
            V=f"{V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2)"
            #form factors for A1 and A2
            A1=f"{A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2)"
            A2=f"{A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2)"
        #'B^+_c -> B^*_s^0+ e^+ + N'
        if pid0 in ['541','-541'] and pid1 in ['533','-533']:
            A00=-.33; mfitA0=1.86; deltaA0=.13; V0=3.25; mfitV=1.76; deltaV=-.052
            A10=.4; mfitA1=3.44; deltaA1=-1.07; A20=10.4; mfitA2=1.73; deltaA2=0.09
            A0=f"{A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2)"
            V=f"{V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2)"
            #form factors for A1 and A2
            A1=f"{A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2)"
            A2=f"{A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2)"
        f1=f"({V}/(self.masses('{pid0}')+self.masses('{pid1}')))"
        f2=f"((self.masses('{pid0}')+self.masses('{pid1}'))*{A1})"
        f3=f"(-{A2}/(self.masses('{pid0}')+self.masses('{pid1}')))"
        f4=f"((self.masses('{pid1}')*(2*{A0}-{A1}-{A2})+self.masses('{pid0}')*({A2}-{A1}))/q**2)"
        f5=f"({f3}+{f4})"
        #form factors for A0 and V, the form at least
        #s1A0 is sigma_1(A0) etc.
        omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2+m3**2-self.masses('{pid2}')**2-2*self.masses('{pid0}')*energy)"
        Omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2-q**2)"
        prefactor=f"(({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*self.masses('{pid0}')**2))*{self.vcoupling[str(abs(int(pid2)))]}**2"
        term1=f"({f2}**2/2)*(q**2-m3**2-self.masses('{pid2}')**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
        term2=f"({f5}**2/2)*(m3**2+self.masses('{pid2}')**2)*(q**2-m3**2+self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
        term3=f"2*{f3}**2*self.masses('{pid1}')**2*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)*(m3**2+self.masses('{pid2}')**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
        term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
        term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-self.masses('{pid2}')**2))"
        term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/self.masses('{pid1}')**2)*(m3**2-self.masses('{pid2}')**2)+({Omegasqr}**2/self.masses('{pid1}')**2)*self.masses('{pid2}')**2+2*(m3**2-self.masses('{pid2}')**2)**2-2*q**2*(m3**2+self.masses('{pid2}')**2))"
        term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2)+2*{omegasqr}*(self.masses('{pid2}')**2-m3**2)+{Omegasqr}*(m3**2-self.masses('{pid2}')**2-q**2))"
        term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+self.masses('{pid2}')**2)-2*self.masses('{pid1}')**2*(q**4-(m3**2-self.masses('{pid2}')**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-self.masses('{pid2}')**2)+2*{omegasqr}**2*q**2)"
        bra=str(prefactor) + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
        return(bra)

    #pid0 is tau, pid1 is produced lepton and pid2 is the neutrino
    def get_3body_dbr_tau(self,pid0,pid1,pid2):
        if pid2=='18':
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=f"({tautau}*coupling**2*{GF}**2*self.masses('{pid0}')**2*energy/(2*np.pi**3))*{self.vcoupling[str(abs(int(pid1)))]}**2"
            dbr=f"{prefactor}*(1+((mass**2-self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-2*(energy/self.masses('{pid0}')))*(1-(self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*energy*self.masses('{pid0}'))))*np.sqrt(energy**2-mass**2)"
        else:
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=f"({tautau}*coupling**2*{GF}**2*self.masses('{pid0}')**2/(4*np.pi**3))*{self.vcoupling[str(abs(int(pid1)))]}**2"
            dbr=f"{prefactor}*(1-self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*energy*self.masses('{pid0}')))**2*np.sqrt(energy**2-mass**2)*((self.masses('{pid0}')-energy)*(1-(mass**2+self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-(1-self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*energy*self.masses('{pid0}')))*((self.masses('{pid0}')-energy)**2/self.masses('{pid0}')+((energy**2-mass**2)/(3*self.masses('{pid0}')))))"
        return(dbr)
