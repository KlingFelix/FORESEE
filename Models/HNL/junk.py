#prefactor for HNL 2 body branching ratio
def calc_prefactor(MH,VH,tauH,fH):
    SecToGev=1./(6.582122*pow(10.,-25.))
    tauH=tauH*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    prefactor=((tauH*GF**2*fH**2*VH**2)/(8*np.pi))
    return(prefactor)


#meson goes to N + e, or H -> N + e
#done
#df is a dataframe consisting of all information available for meson two body decays
'''def add_2bodydecay_h_e(df,pidH,pid1='11',energy='14',nsample=10,generator="Pythia8"):
    pid=pidH
    if int(pidH)>0:
        pid1=str(-abs(int(pid1)))
    if int(pidH)<0:
        pid1=str(abs(int(pid1)))
    #pid=str(abs(int(pid)))
    H_attr=df.loc[df['pid']==pid]
    a=df.loc[df['pid']=='411']
    tauH=H_attr['tauH (sec)'].values[0]
    MH=H_attr['MH (GeV)'].values[0]
    VH=H_attr['VH (unitless)'].values[0]
    fH=H_attr['fH (GeV)'].values[0]
    prefactor=calc_prefactor(MH,VH,tauH,fH)
    model.add_production_2bodydecay(
    pid0 = f"{pid}",
    pid1 = f"{pid1}",
    br=str(prefactor)+f"*coupling**2*mass**2*self.masses('{pid}')*(1.-(mass/self.masses('{pid}'))**2 + 2.* (self.masses('{pid1}')/self.masses('{pid}'))**2 + (self.masses('{pid1}')/mass)**2*(1.-(self.masses('{pid1}')/self.masses('{pid}'))**2)) * np.sqrt((1.+(mass/self.masses('{pid}'))**2 - (self.masses('{pid1}')/self.masses('{pid}'))**2)**2-4.*(mass/self.masses('{pid}'))**2)",
    generator = generator,
    energy = energy,
    nsample = nsample
    )'''

#add_2bodydecay_h_e(df,'431',pid1='15',energy='14',nsample=100,generator="pythia8")



#tau 2 body decay
#done
'''pid='15'
prefactor=tautau*coupling**2*GF**2*VH**2*fH**2*Mtau**3
#tau -> H + N
model.add_production_2bodydecay(
    pid0 = f"{pid}",
    pid1 = f"{pid1}",
    br=f"prefactor*((1-(mass**2/self.masses('{pid0}')))**2-(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+(mass**2/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid0}')+mass)**2/self.masses('{pid0}')**2))))",
    generator = generator,
    energy = energy,
    nsample = nsample
)
prefactor=tautau* coupling**2*grho**2*GF**2*Vud**2*fH**2*Mtau**3/(8*np.pi*Mrho**2)
#tau -> rho + N
model.add_production_2bodydecay(
    pid0 = f"{pid}",
    pid1 = f"{pid1}",
    br=f"prefactor*((1-(mass**2/self.masses('{pid0}')))**2+(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+((mass**2-2*self.masses('{pid1}')**2)/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid0}')+mass)**2/self.masses('{pid0}')**2))))",
    generator = generator,
    energy = energy,
    nsample = nsample
)'''



#tau 3 body decays
#done
#tau -> \nu_{\tau} + l_{\alpha} + N
'''pid0="15"
pid2="18"
pid1="11"
SecToGev=1./(6.582122*pow(10.,-25.))
tautau=290.1*1e-15*SecToGev
GF=1.166378*10**(-5) #GeV^(-2)
prefactor=f"({tautau}*coupling**2*{GF}**2*self.masses('{pid0}')**2*EN/(2*np.pi**3))"
dbr=f"{prefactor}*(1+((mass**2-self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-2*(EN/self.masses('{pid0}')))*(1-(self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*EN*self.masses('{pid0}'))))*np.sqrt(EN**2-mass**2)"
#for another type of decay
prefactor=f"({tautau}*coupling**2*{GF}**2*self.masses('{pid0}')**2/(4*np.pi**3))"
dbr=f"{prefactor}*(1-self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*EN*self.masses('{pid0}')))**2*np.sqrt(EN**2-mass**2)*((self.masses('{pid0}')-EN)*(1-(mass**2+self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-(1-self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*EN*self.masses('{pid0}')))*((self.masses('{pid0}')-EN)**2/self.masses('{pid0}')+((EN**2-mass**2)/(3*self.masses('{pid0}')))))"
model.add_production_3bodydecay(
    label= "5_di",
    pid0 = pid0,
    pid1 = pid1,
    pid2 = pid2,
    br = dbr,
    generator ="pythia8",
    energy = energy,
    nsample = 100,
    scaling = 0, 
)'''



#3 body decays; pseudoscalar into semileptonic decay
#prefactor for HNL 3 body branching ratio
#done

'''#pid0 is parent meson pid1 is produced pseudoscalar pid2 is lepton pid ['particle','pid0','pid1','tauH (sec)','VHHp (unitless)']
df_D=pd.DataFrame([],columns=['particle','pid0','pid1','tauH (sec)','VHHp'])
particles=[['D^+ -> \bar{K}^0 + l + N','411','-311',1040*10**(-15),0.987],['D^- -> \bar{K}^0 + l + N','-411','-311',1040*10**(-15),0.987],['D^0 -> K^- + l + N','421','-321',410.1*10**(-15),0.997],['B^+ -> \bar{D}^0 + l + N','521','421',1.638*10**(-12),41*10**(-3)],['B^0 -> D^- + l + N','511','-411',1.519*10**-12,42.2*10**-3],['B^0_s -> D^-_s + l + N','531','-431',1.515*10**-12,42.2*10**-3],['B^+_c -> B^0 + l + N','541','511',0.510*10**-12,0.221],['B^+_c -> B^0_s + l + N','541','511',0.510*10**-12,0.987]]
for n in range(len(particles)):
    df_D.loc[len(df_D)]=particles[n]
def calc_prefactor(pid0,VHHp,tauH):
    SecToGev=1./(6.582122*pow(10.,-25.))
    tauH=tauH*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    prefactor=f"{tauH}*coupling**2*{VHHp}**2*{GF}**2/(64*np.pi**3*self.masses('{pid0}')**2)"
    #prefactor=((tauH*GF**2*fH**2*VH**2)/(8*np.pi))
    return(prefactor)
#pid2 is produced lepton
tauH=410.1*10**(-15) #GeV; lifetime of meson
VHHp=0.997 #unitless Vcd matrix element
pid0="421" #parent H
pid1="-321" #produced meson H'
pid2="-11" #produced lepton l

def decay_semi_pseud_3body_D(df,pid0,pid2,channel="D"):

    tauH=df.loc[df['pid0']==pid0]['tauH (sec)'].values[0]
    VHHp=df.loc[df['pid0']==pid0]['VHHp'].values[0]
    pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    prefactor=calc_prefactor(pid0,VHHp,tauH)
    if channel=="D":
        f00=.747 #for D mesons
        fp0=f00 #for D mesons
        MV=2.01027 #mass in GeV for D mesons
        MS=2.318    #mass in GeV for D mesons
    if channel=="B":
        f00=0.66
        fp0=f00 #for B mesons
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        #MS=5.4154 for B meson
        MS=1.969
    if channel=="Bs":
        f00=0.57
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->B":
        f00=-0.58
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->Bs":
        f00=-0.61
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    pidk="-321"
    pidpi="211"
    fp=f"{f00}/(1-q**2/{MV}**2)"
    f0=f"{f00}/(1-q**2/{MS}**2)"
    fm=f"({f0}-{fp})*(self.masses('{pid0}')**2-self.masses('{pid1}')**2)/q**2"
    term1=f"({fm})**2*(q**2*(m3**2+self.masses('{pid2}')**2)-(m3**2-self.masses('{pid2}')**2)**2)"
    term2=f"2*({fp})*({fm})*m3**2*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term3=f"(2*({fp})*({fm})*self.masses('{pid2}')**2*(4*EN*self.masses('{pid0}')+ self.masses('{pid2}')**2-m3**2-q**2))"
    term4=f"({fp})**2*(4*EN*self.masses('{pid0}')+self.masses('{pid2}')**2-m3**2-q**2)*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term5=f"-({fp})**2*(2*self.masses('{pid0}')**2+2*self.masses('{pid1}')**2-q**2)*(q**2-m3**2-self.masses('{pid2}')**2)"
    bra=str(prefactor)  + "*(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"

    model.add_production_3bodydecay(
        label= "5_di",
        pid0 = pid0,
        pid1 = pid1,
        pid2 = pid2,
        br = bra,
        generator = "Pythia8",
        energy = energy,
        nsample = 100,
        scaling = 0, 
    )

def integrate_pseud_3body_D(df,pid0,pid2,m3,nsample=100,channel="D"):
    tauH=df.loc[df['pid0']==pid0]['tauH (sec)'].values[0]
    VHHp=df.loc[df['pid0']==pid0]['VHHp'].values[0]
    pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    prefactor=calc_prefactor(pid0,VHHp,tauH)
    if channel=="D":
        f00=.747 #for D mesons
        fp0=f00 #for D mesons
        MV=2.01027 #mass in GeV for D mesons
        MS=2.318    #mass in GeV for D mesons
    if channel=="B":
        f00=0.66
        fp0=f00 #for B mesons
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
        #MS=1.969
    if channel=="Bs":
        f00=0.57
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->B":
        f00=-0.58
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->Bs":
        f00=-0.61
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    m0=self.masses(pid0)
    m1=self.masses(pid1)
    m2=self.masses(pid2)
    pidk="-321"
    pidpi="211"
    fp=f"{f00}/(1-q**2/{MV}**2)"
    f0=f"{f00}/(1-q**2/{MS}**2)"
    fm=f"({f0}-{fp})*(self.masses('{pid0}')**2-self.masses('{pid1}')**2)/q**2"
    term1=f"({fm})**2*(q**2*(m3**2+self.masses('{pid2}')**2)-(m3**2-self.masses('{pid2}')**2)**2)"
    term2=f"2*({fp})*({fm})*m3**2*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term3=f"(2*({fp})*({fm})*self.masses('{pid2}')**2*(4*EN*self.masses('{pid0}')+ self.masses('{pid2}')**2-m3**2-q**2))"
    term4=f"({fp})**2*(4*EN*self.masses('{pid0}')+self.masses('{pid2}')**2-m3**2-q**2)*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term5=f"-({fp})**2*(2*self.masses('{pid0}')**2+2*self.masses('{pid1}')**2-q**2)*(q**2-m3**2-self.masses('{pid2}')**2)"
    bra=str(prefactor)  + "*(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
   
    integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
    return(integ)

pid0="521"
pid2="-11"
decay_semi_pseud_3body_D(df_D,pid0,pid2)
'tauH VHHp, pid1'
print(df_D.loc[df_D['pid0']==pid0]['tauH (sec)'].values[0],
df_D.loc[df_D['pid0']==pid0]['VHHp'].values[0],
df_D.loc[df_D['pid0']==pid0]['pid1'].values[0])'''


#for 3 body decay into vector meson (original)
#D->V+l+\nu V here is a vector meson like K^*
#K+ u \bar{s}; D0 c \bar{u}
#in particular we are going to do D0->K*- + e+ + N
#I am assuming Mp is the mass of Ds (scalar) looking at the reference this appears to be correct
#MV is the mass of D*s
#0~parent meson; 1~daughter meson; 2~lepton; 3~sterile neutrino
#self=model
'''SecToGev=1./(6.582122*pow(10.,-25.))
tauH=410.1*10**(-15)
tauH=tauH*SecToGev
GF=1.166378*10**(-5) #GeV^(-2)
VHV=.967 #Vcs matrix element
pid0="421"
pid1="-323"
pid2="-11"
#pid3=
A00=.76
Mp=1.97
s1A0=.17
s2A0=0
V0=1.03
MV=2.11
#Ml=self.masses(pid2)
s1V=.27
s2V=0
A10=.66
s1A1=.3
s2A1=.2
A20=.49
s1A2=.67
s2A2=.16

A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
#form factors for A1 and A2
A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

f1=f"({V}/(self.masses('{pid0}')+self.masses('{pid1}')))"
f2=f"((self.masses('{pid0}')+self.masses('{pid1}'))*{A1})"
f3=f"(-{A2}/(self.masses('{pid0}')+self.masses('{pid1}')))"
f4=f"((self.masses('{pid1}')*(2*{A0}-{A1}-{A2})+self.masses('{pid0}')*({A2}-{A1}))/q**2)"
f5=f"({f3}+{f4})"
#form factors for A0 and V, the form at least
#s1A0 is sigma_1(A0) etc.
omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2+m3**2-self.masses('{pid2}')**2-2*self.masses('{pid0}')*EN)"
#omega=f"self.masses('{pid0}')*m3*EN/q**2"
Omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2-q**2)"
prefactor=f"(({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*self.masses('{pid0}')**2))"
term1=f"({f2}**2/2)*(q**2-m3**2-self.masses('{pid2}')**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
term2=f"({f5}**2/2)*(m3**2+self.masses('{pid2}')**2)*(q**2-m3**2+self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
term3=f"2*{f3}**2*self.masses('{pid1}')**2*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)*(m3**2+self.masses('{pid2}')**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-self.masses('{pid2}')**2))"
term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/self.masses('{pid1}')**2)*(m3**2-self.masses('{pid2}')**2)+({Omegasqr}**2/self.masses('{pid1}')**2)*self.masses('{pid2}')**2+2*(m3**2-self.masses('{pid2}')**2)**2-2*q**2*(m3**2+self.masses('{pid2}')**2))"
term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2)+2*{omegasqr}*(self.masses('{pid2}')**2-m3**2)+{Omegasqr}*(m3**2-self.masses('{pid2}')**2-q**2))"
term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+self.masses('{pid2}')**2)-2*self.masses('{pid1}')**2*(q**4-(m3**2-self.masses('{pid2}')**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-self.masses('{pid2}')**2)+2*{omegasqr}**2*q**2)"
bra=prefactor + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
#MH=self.masses(pid0)
model.add_production_3bodydecay(
    label= "5_di",
    pid0 = pid0,
    pid1 = "-323",
    pid2 = pid2,
    br = bra,
    generator ="pythia8",
    energy = energy,
    nsample = 100,
    scaling = 0, 
)
self=model
#the bounds of integration are not correct rn.
#term 8 seems huge
#print(self.masses(pid0))
#print(self.masses(pid1))
#print(self.masses(pid2))

print(eval(term8))
print(eval(bra))
m0=self.masses(f'{pid0}')
m1=self.masses(f'{pid1}')
m2=self.masses(f'{pid2}')
m3=0
nsample=100
integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
print(integ)
q=1
EN=1
m3=1
print(eval(bra))'''



#for 3 body decay into vector meson
#recently silenced this function
#D->V+l+\nu V here is a vector meson like K^*
#K+ u \bar{s}; D0 c \bar{u}
#in particular we are going to do D0->K*- + e+ + N
#I am assuming Mp is the mass of Ds (scalar) looking at the reference this appears to be correct
#MV is the mass of D*s
#0~parent meson; 1~daughter meson; 2~lepton; 3~sterile neutrino
#self=model
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
import math
import random
from skhep.math.vectors import LorentzVector, Vector3D
from scipy import interpolate
from matplotlib import gridspec
from skhep.math.vectors import LorentzVector, Vector3D
import random

df=pd.DataFrame([],columns=['particle','pid0','pid1','tauH (sec)','VHHp'])
'''['D^0 -> K*^- + e^+ + N','421','-323',410.1*10**(-15),0.987]'''
particles=[['D^0 -> K*^- + e^+ + N','421','-323',410.1*10**(-15),0.967],['B^+ -> \bar{D}*^0 + e^+ + N','521','-423',1.638*10**-12,41*10**-3],['B^0 -> D*^- + e^+ + N','511','-413',1.519*10**-12,41*10**-3],['B^0_s -> D^*_s^- + e^+ + N','531','-433',1.515*10**-12,41*10**-3],['B^+_c -> B*^0 + e^+ + N','541','513',.510*10**-12,.221],['B^+_c -> B^*_s^0+ e^+ + N','541','533',.510*10**-12,41*10**-3]]
for n in range(len(particles)):
    df.loc[len(df)]=particles[n]
print(df)
SecToGev=1./(6.58*pow(10.,-25.))
tauH=1.638*10**-12
tauH=tauH*SecToGev
GF=1.166378*10**(-5) #GeV^(-2)
VHV=41*10**-3 #Vcs matrix element
pid0="421"
pid1="-323"
pid2="-11"
#pid3=
def calc_br(df,pid0,pid2,m3,nsample=100,pid1=None):
    tauH=df.loc[df['pid0']==pid0]['tauH (sec)'].values[0]
    tauH=tauH*SecToGev
    VHV=df.loc[df['pid0']==pid0]['VHHp'].values[0]
    pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    GF=1.1663787*10**(-5)
    if df.loc[df['pid0']==pid0]['particle'].values[0]=='D^0 -> K*^- + e^+ + N':
        A00=.76
        Mp=1.97
        s1A0=.17
        s2A0=0
        V0=1.03
        MV=2.11
        #Ml=self.masses(pid2)
        s1V=.27
        s2V=0
        A10=.66
        s1A1=.3
        s2A1=.2*0
        A20=.49
        s1A2=.67
        s2A2=.16*0
        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
    if df.loc[df['pid0']==pid0]['particle'].values[0]=='B^+ -> \bar{D}*^0 + e^+ + N' or df.loc[df['pid0']==pid0]['particle'].values[0]=='B^0 -> D*^- + e^+ + N':    
        A00=0.69
        Mp=6.277
        s1A0=0.58
        s2A0=0
        V0=0.76
        MV=6.842
        #Ml=self.masses(pid2)
        s1V=0.57
        s2V=0
        A10=0.66
        s1A1=0.78
        s2A1=0
        A20=0.62
        s1A2=1.04
        s2A2=0
        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"   

    if df.loc[df['pid0']==pid0]['particle'].values[0]=='B^0_s -> D^*_s^- + e^+ + N':
        A00=0.67
        Mp=6.842
        s1A0=0.35
        s2A0=0
        V0=0.95
        MV=6.842
        #Ml=self.masses(pid2)
        s1V=0.372
        s2V=0
        A10=0.70
        s1A1=0.463
        s2A1=0
        A20=0.75
        s1A2=1.04
        s2A2=0
        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        
    if df.loc[df['pid0']==pid0]['particle'].values[0]=='B^+_c -> B*^0 + e^+ + N' and pid1=='513':
        A00=-.27
        mfitA0=1.86
        deltaA0=.13
        V0=3.27
        mfitV=1.76
        deltaV=-.052
        A10=.6
        mfitA1=3.44
        deltaA1=-1.07
        A20=10.8
        mfitA2=1.73
        deltaA2=0.09
        A0=f"{A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2)"
        V=f"{V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2)"
        #form factors for A1 and A2
        A1=f"{A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2)"
        A2=f"{A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2)"

    if df.loc[df['pid0']==pid0]['particle'].values[0]=='B^+_c -> B^*_s^0+ e^+ + N' and pid1=='533':
        A00=-.33
        mfitA0=1.86
        deltaA0=.13
        V0=3.25
        mfitV=1.76
        deltaV=-.052
        A10=.4
        mfitA1=3.44
        deltaA1=-1.07
        A20=10.4
        mfitA2=1.73
        deltaA2=0.09
        A0=f"{A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2)"
        V=f"{V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2)"
        #form factors for A1 and A2
        A1=f"{A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2)"
        A2=f"{A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2)"

    '''   
    A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
    V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
    #form factors for A1 and A2
    A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
    A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"'''


    f1=f"({V}/(self.masses('{pid0}')+self.masses('{pid1}')))"
    f2=f"((self.masses('{pid0}')+self.masses('{pid1}'))*{A1})"
    f3=f"(-{A2}/(self.masses('{pid0}')+self.masses('{pid1}')))"
    f4=f"((self.masses('{pid1}')*(2*{A0}-{A1}-{A2})+self.masses('{pid0}')*({A2}-{A1}))/q**2)"
    f5=f"({f3}+{f4})"
    #form factors for A0 and V, the form at least
    #s1A0 is sigma_1(A0) etc.
    omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2+m3**2-self.masses('{pid2}')**2-2*self.masses('{pid0}')*EN)"
    #omega=f"self.masses('{pid0}')*m3*EN/q**2"
    Omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2-q**2)"
    prefactor=f"(({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*self.masses('{pid0}')**2))"
    term1=f"({f2}**2/2)*(q**2-m3**2-self.masses('{pid2}')**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
    term2=f"({f5}**2/2)*(m3**2+self.masses('{pid2}')**2)*(q**2-m3**2+self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
    term3=f"2*{f3}**2*self.masses('{pid1}')**2*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)*(m3**2+self.masses('{pid2}')**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
    term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
    term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-self.masses('{pid2}')**2))"
    term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/self.masses('{pid1}')**2)*(m3**2-self.masses('{pid2}')**2)+({Omegasqr}**2/self.masses('{pid1}')**2)*self.masses('{pid2}')**2+2*(m3**2-self.masses('{pid2}')**2)**2-2*q**2*(m3**2+self.masses('{pid2}')**2))"
    term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2)+2*{omegasqr}*(self.masses('{pid2}')**2-m3**2)+{Omegasqr}*(m3**2-self.masses('{pid2}')**2-q**2))"
    term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+self.masses('{pid2}')**2)-2*self.masses('{pid1}')**2*(q**4-(m3**2-self.masses('{pid2}')**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-self.masses('{pid2}')**2)+2*{omegasqr}**2*q**2)"
    bra=prefactor + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
    #,A0,A1,A2,V,f1,f2,f3,f4,f5
    return(bra,omegasqr,Omegasqr,prefactor,term1,term2,term3,term4,term5,term6,term7,term8)


def integrate_vec_3body(df,pid0,pid2,m3,nsample=100,pid1=None):
    pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    bra=calc_br(df,pid0,pid2,m3,nsample=nsample,pid1=pid1)[0]

    m0=self.masses(f'{pid0}')
    m1=self.masses(f'{pid1}')
    m2=self.masses(f'{pid2}')
    integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
    return(integ)

def calc_br_vector_felix(pid0,mass,EN,q2):
    #B0s -> Ds-*lN
    _mlep2 = pow(0.0005109989461,2)
    _mN= mass;
    _mN2= pow(_mN,2);
    if pid0=='531':
        _tau = 1.505e-12;
        _mH = 5.36689;
        _mVec = 2.11210;
        _Vckm2 = pow(0.0405,2);
        _Vat0=0.95;
        _A0at0=0.67;
        _A1at0=0.7;
        _A2at0=0.75;
        _sig1V=0.372;
        _sig1A0=0.35;
        _sig1A1=0.463;
        _sig1A2=1.04;
        _mVvec2 = pow(6.842,2);
        _mSvec2 = pow(6.842,2);
        _mHprim=_mVec;
        _tau = _tau*(1.e25/6.58);
        _mH2 = pow(_mH,2);
        _mHp2 = pow(_mHprim,2);
        _mVec2 = pow(_mVec,2);
        GF2=pow(1.1663787e-5,2);
        pi3=pow(np.pi,3);
    if pid0=='511':
        _tau = 1.520e-12;
        _mH = 5.27963;
        _mVec = 2.01026;
        _Vckm2 = pow(0.0405,2);
        _Vat0=0.76;
        _A0at0=0.69;
        _A1at0=0.66;
        _A2at0=0.62;
        _sig1V=0.57;
        _sig1A0=0.58;
        _sig1A1=0.78;
        _sig1A2=1.4;
        _mVvec2 = pow(6.842,2);
        _mSvec2 = pow(6.277,2);
        _mHprim=_mVec;
        _tau = _tau*(1.e25/6.58);
        _mH2 = pow(_mH,2);
        _mHp2 = pow(_mHprim,2);
        _mVec2 = pow(_mVec,2);
        GF2=pow(1.1663787e-5,2);
        pi3=pow(np.pi,3);

    if pid0=='421':
        _tau = 4.101e-13
        _mH = 1.86959
        _mH = 1.86484 #my value
        _mVec = 0.89176
        _mVec = 0.89166 #my value
        _Vckm2 = pow(0.995,2)
        _Vckm2 = pow(0.967,2) #my value
        _Vat0=1.03
        _A0at0=0.76
        _A1at0=0.66
        _A2at0=0.49
        _sig1V=0.27
        _sig1A0=0.17
        _sig1A1=0.3
        _sig1A2=0.67
        _mVvec2 = pow(2.007,2)
        _mVvec2 = pow(2.11,2)
        _mSvec2 = pow(1.97,2)
        _mHprim=_mVec
        _tau = _tau*(1.e25/6.58)
        _mH2 = pow(_mH,2);
        _mHp2 = pow(_mHprim,2)
        _mVec2 = pow(_mVec,2)
        GF2=pow(1.1663787e-5,2)
        pi3=pow(np.pi,3)




    om2 = _mH2-_mVec2+_mN2-_mlep2-2.*_mH*EN
    com2 = _mH2-_mVec2-q2
    aux = (_tau*GF2*_Vckm2)/(32.*pi3*_mH2)
    auxf2sq = 0.5*(q2-_mN2-_mlep2+om2*(com2-om2)/_mVec2)
    auxf5sq = 0.5*(_mN2+_mlep2)*(q2-_mN2+_mlep2)*(pow(com2,2)/(4.*_mVec2) - q2)
    auxf3sq = 2.*_mVec2*(pow(com2,2)/(4.*_mVec2)-q2)*(_mN2+_mlep2-q2+om2*(com2-om2)/_mVec2)
    auxf3f5 = 2.*(_mN2*om2+(com2-om2)*_mlep2)*(pow(com2,2)/(4.*_mVec2)-q2)
    auxf1f2 = 2.*(q2*(2.*om2-com2)+com2*(_mN2-_mlep2))
    auxf2f5 = 0.5*(om2*com2*(_mN2-_mlep2)/_mVec2+pow(com2,2)*_mlep2/_mVec2+2.*pow(_mN2-_mlep2,2)-2.*q2*(_mN2+_mlep2))
    auxf2f3 = com2*om2*(com2-om2)/_mVec2 + 2.*om2*(_mlep2-_mN2) + com2*(_mN2-_mlep2-q2)
    auxf1sq = pow(com2,2)*(q2-_mN2+_mlep2)-2.*_mVec2*(pow(q2,2)-pow(_mN2-_mlep2,2)) + 2.*om2*com2*(_mN2-q2-_mlep2) + 2.*pow(om2,2)*q2
    
    V  = _Vat0/((1.-(q2/_mVvec2))*(1.-_sig1V*(q2/_mVvec2)))
    A0 = _A0at0/((1.-(q2/_mSvec2))*(1.-_sig1A0*(q2/_mSvec2)))
    A1 = _A1at0/(1.-_sig1A1*(q2/_mVvec2))
    A2 = _A2at0/(1.-_sig1A2*(q2/_mVvec2))
    
    f1 = V/(_mH+_mVec)
    f2 = (_mH+_mVec)*A1
    f3 = -A2/(_mH+_mVec)
    f4 = (_mVec*(2.*A0-A1-A2)+_mH*(A2-A1))/q2
    f5 = f3+f4
    
    result = aux*(pow((f2),2)*(auxf2sq) + pow((f5),2)*(auxf5sq) + pow((f3),2)*(auxf3sq) + (f3)*(f5)*(auxf3f5) + (f1)*(f2)*(auxf1f2) + (f2)*(f5)*(auxf2f5) + (f2)*(f3)*(auxf2f3) + pow((f1),2)*(auxf1sq));
    #,V,A0,A1,A2,f1,f2,f3,f4,f5
    return (result,om2,com2,aux,auxf2sq,auxf5sq,auxf3sq,auxf3f5,auxf1f2,auxf2f5,auxf2f3,auxf1sq);

#this is a test
def integrate_vec_3body_felix(pid0,df,mass,nsample):
    #pid0='531'
    pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    m0=self.masses(f'{pid0}')
    m1=self.masses(f'{pid1}')
    m2=self.masses(f'{pid2}')



    # prepare output
    particles, weights = [], []
    #create parent 4-vector
    p_mother=LorentzVector(0,0,0,m0)

    #integration boundary
    q2min,q2max = (m2+m3)**2,(m0-m1)**2
    '''E2st = (q**2 - m2**2 + m3**2)/(2*q)
    E3st = (m0**2 - q**2 - m1**2)/(2*q)
    m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
    m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
    cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
    cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
    #felix bounds
    ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
    ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)'''
    #cthmin,cthmax = m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0)
    #print("integration boundaries ",[q2min,q2max],[cthmin,cthmax])
    mass = m2
    #mass=1
    #numerical integration
    integral=0
    ####added
    #cth=EN
    ####
    #felix   | mine
    # m_{chi}| m2
    # m2     | m3
    # m0     | m0
    # m1     | m1
    for i in range(nsample):

        #Get kinematic Variables
        q2 = random.uniform(q2min,q2max)
        q  = math.sqrt(q2)
        E2st = (q**2 - m2**2 + m3**2)/(2*q)
        E3st = (m0**2 - q**2 - m1**2)/(2*q)
        m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
        m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
        #cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        #cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
        #ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0) #Alec
        #ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0) #Alec
        ENmin = -(m232max  - m2**2 - m0**2)/(2*m0)    #Felix
        ENmax = -(m232min  - m2**2 - m0**2)/(2*m0)      #Felix
        EN = random.uniform(ENmin,ENmax)
        #th = np.arccos(EN)
        br=calc_br_vector_felix(pid0,mass,EN,q2)[0]

        #decay meson and V
        cosQ =random.uniform(-1,1)
        phiQ =random.uniform(-math.pi,math.pi)
        cosM =random.uniform(-1.,1.)
        phiM =random.uniform(-math.pi,math.pi)
        #p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
        #p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

        #branching fraction
        brval  = br
        #print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
        brval *= (q2max-q2min)*(ENmax-ENmin)/float(nsample)

        #save
        #particles.append(p_3)
        weights.append(brval)
    #print('branching frac', brval)
    print("sum of weights ",sum(weights))
    #return particles,weights
    return(sum(weights))
    #integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
    #return(integ)
#MH=self.masses(pid0)
'''model.add_production_3bodydecay(
    label= "5_di",
    pid0 = pid0,
    pid1 = "-323",
    pid2 = pid2,
    br = bra,
    generator ="pythia8",
    energy = energy,
    nsample = 100,
    scaling = 0, 
)'''
self=model
#the bounds of integration are not correct rn.
#term 8 seems huge
#print(self.masses(pid0))
#print(self.masses(pid1))
#print(self.masses(pid2))


#calc_br_vector_felix(pid0,mass,EN,q2)
'''
m3=0
mass=0
delm=.1
xf=[]; yf=[]
x=[]; y=[]
for m in range(100):
    result,V,A0,A1,A2,f1,f2,f3,f4,f5=calc_br_vector_felix('421',mass,EN,q2)
    bra,f1f,f2f,f3f,f4f,f5f,A0f,A1f,A2f=calc_br(df,'421','-11',mass,nsample=nsample)
    x.append(mass)
    y.append(eval(bra))
    xf.append(mass)
    yf.append(result)
    m3+=delm
    mass+=delm
plt.plot(x,y)
plt.plot(xf,yf)
plt.show()
print(x,y)'''
pid0='421'
'''coupling=1
mass=.3
m3=mass
EN=2
q=3
q2=9
result,om2,com2,aux,auxf2sq,auxf5sq,auxf3sq,auxf3f5,auxf1f2,auxf2f5,auxf2f3,auxf1sq=calc_br_vector_felix(pid0,mass,EN,q2)
bra,omegasqr,Omegasqr,prefactor,term1,term2,term3,term4,term5,term6,term7,term8=calc_br(df,pid0,pid2,m3,nsample=100,pid1=None)
print('mine ',eval(bra), ' felix ',result)
x=[]; y=[]
xf=[]; yf=[]
q=0.01
q2=0.01**2
delq=.01
for n in range(100):
    result,om2,com2,aux,auxf2sq,auxf5sq,auxf3sq,auxf3f5,auxf1f2,auxf2f5,auxf2f3,auxf1sq=calc_br_vector_felix(pid0,mass,EN,q2)
    bra,omegasqr,Omegasqr,prefactor,term1,term2,term3,term4,term5,term6,term7,term8=calc_br(df,pid0,pid2,m3,nsample=100,pid1=None)
    x.append(q**2)
    y.append(eval(bra))
    xf.append(q2)
    yf.append(result)
    q+=delq
    q2=q**2
plt.plot(x,y)
plt.plot(xf,yf)
plt.show()'''



#for 3 body pseudoscalar meson decay D mesons
#comparing bell curves with the paper
'''print(df_D)
self=model
m0=self.masses("521")
m1=self.masses("-321")
m2=self.masses("-11")
pid0='511'
pid2='-11'
nsample=1000
delm=.1
x=[]
y=[]
m3=0
for n in range(1,30):
    #y.append(foresee.integrate(bra, 1, m0, m1, m2,m3, nsample))
    y.append(integrate_pseud_3body_D(df_D,pid0,pid2,m3,nsample=10000,channel="B"))
    x.append(m3)
    m3+=delm


felixx=[0.0043741884,
0.15978788,
0.38926172,
0.54105836,
0.6966135,
0.9595848,
1.1189028,
1.3486366,
1.6709628,
2.0561743,
2.5004394,
2.988889,
]
felixy=[0.020463001,
0.020125093,
0.019263254,
0.018369755,
0.017075056,
0.014824672,
0.013097913,
0.0104768155,
0.007085065,
0.003693961,
0.001198528,
9.2440365E-5,

]
print(x)
#plt.plot(x,np.array(y))
#plt.ylim([-.04,.04])
#plt.xlim([0,1.5])
#plt.plot(x1,y1)
#plt.show()
#print('results ',integrate_pseud_3body_D(df_D,pid0,pid2,0,nsample=100,channel="B"))
print(x,y)
plt.ylim([0,.03])
plt.xlim([0,3])
plt.plot(x,np.array(y))
plt.plot(felixx,felixy)
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.show()
print('should match experiment')
integrate_pseud_3body_D(df_D,pid0,pid2,0,nsample=10000,channel="Bs")
#we ended on B+ → D0 +e+ +N process and it agreed with experiment
#there is significant discrepancy between experiment and predictions for B^0_s -> D^-_s + l + N, roughly 1 percent (prediction) and 8 percent (experiment)
#I cant find anything for B+_c decays'''



#compare with Felix
'''expx=[0.0]
expy=[.089]
felixx = [.007031207,
   0.22711568,
   0.428421,
   0.69046235,
   0.91490155,
   1.1297208,
   1.213503,
   1.264641
]
felixy = [0.05467126,
   0.046997737,
   0.03590057,
   0.018104471,
   0.0064042676,
   0.0011470531,
   2.76742 *10** - 4,
   9.814598 *10** - 5
]
plt.ylim([0,.1])
plt.xlim([0,3])
plt.plot(x,np.array(y),label="Alec/Daniel")
plt.plot(np.array(expx),np.array(expy), marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red",label=r"Experiment ($m_n=0$)")
plt.plot(felixx,felixy,label="Felix")
plt.legend(framealpha=1, frameon=True)
plt.xlabel(r"$m_N (GeV)$")
plt.ylabel(r"Br($D^+ \rightarrow \bar{K}^0 + e^+ + N$)")
#plt.show()
plt.savefig('/Users/alechewitt/Desktop/br_comparison.pdf')'''



#for 3 body vector meson decay
#comparing bell curves with the paper
'''
df=pd.DataFrame([],columns=['particle','pid0','pid1','tauH (sec)','VHHp'])
particles=[['D^0 -> K*^- + e^+ + N','421','-323',410.1*10**(-15),0.967],['B^+ -> \bar{D}*^0 + e^+ + N','521','-423',1.638*10**-12,41*10**-3],['B^0 -> D*^- + e^+ + N','511','-413',1.519*10**-12,41*10**-3],['B^0_s -> D^*_s^- + e^+ + N','531','-433',1.515*10**-12,41*10**-3],['B^+_c -> B*^0 + e^+ + N','541','513',.510*10**-12,.221],['B^+_c -> B^*_s^0+ e^+ + N','541','533',.510*10**-12,41*10**-3]]
for n in range(len(particles)):
    df.loc[len(df)]=particles[n]

self=model
pid0='521'
pid1='-421'
pid2='-11'
m0=self.masses(pid0)
m1=self.masses(pid1)
m2=self.masses(pid2)
print('D mass ',m0)
print('k_mass ',m1)
print('e mass ',m2)
#bra=calc_br(df,pid0,pid2,m3,nsample=100)
nsample=1000
delm=.1

x=[]
y=[]
pid0='421'
#pid1='533'
pid2='-11'
m3=.01*0
print(df)
for n in range(1,15):
    y.append(integrate_vec_3body(df,pid0,pid2,m3,nsample=nsample))
    #y1.append(foresee.integrate(bra, 1, m0, m1, m2,m3, nsample))
    x.append(m3)
    m3+=delm

felixx=[6.70496E-4,
0.21900134,
0.60402113,
0.9299689,
1.2485777,
1.5561073,
1.7709717,
2.1043131,
2.3819897,
2.7668908,
2.977787
]
felixy=[0.020468595,
0.019915968,
0.017823743,
0.014960256,
0.011695955,
0.008308234,
0.0061834347,
0.003350851,
0.0017199162,
4.2917134E-4,
1.230765E-4
]
x1=[]
y1=[]
m3=.01*0
print(df)
nsample=1000
for n in range(1,15):
    y1.append(integrate_vec_3body_felix(pid0,df,m3,nsample))
    #y1.append(foresee.integrate(bra, 1, m0, m1, m2,m3, nsample))
    x1.append(m3)
    m3+=delm
xdan=[-0.0052724076,
0.13181019,
0.24780317,
0.42179263,
0.56414765,
0.685413,
0.8594025,
1.3233744
]
ydan=[0.025273224,
0.022336066,
0.01796448,
0.01010929,
0.0038934427,
0.0015710383,
6.830601E-5,
6.830601E-5
]
plt.plot(x,np.array(y),label="Alec")
#plt.plot(xdan,ydan,label="Daniel")
plt.plot(x1,np.array(y1),label="Felix")
plt.ylim([-.04,.04])
plt.xlim([0,3])
plt.legend(framealpha=1, frameon=True)
plt.xlabel(r"$m_N (GeV)$")
plt.ylabel(f"Br({df.loc[df['pid0']==pid0]['particle'].values[0]})")
plt.show()
#print(foresee.integrate(bra, 1, m0, m1, m2,0, 10000))
#print('this should agree with experiment')'''
#print('results ',foresee.integrate(bra, 1, m0, m1, m2,0, 10))'''
'''mine=integrate_vec_3body(df,pid0,pid2,.3,nsample=10000,pid1=pid1)
felix=integrate_vec_3body_felix(pid0,df,.05,1000)
print('mine ',mine,' felix ', felix)
print(((mine-felix)/mine)*100)
print(y)
print(y1)'''
#print(x1,y1)

((0.006503736976422159-0.011584623694920032)/0.006503736976422159)*100





#for tau 3 body decay tau -> nutau + e + N
'''self=model
pid0="15"
pid2="18"
pid1="11"
m0=self.masses(pid0)
m3=self.masses(pid2)
m1=self.masses(pid1)

nsample=1000
delm=.1
x=[]
y=[]
m2=.01*0
for n in range(1,17):
    y.append(foresee.integrate_tau(dbr, 1, m0, m1, m2,m3, nsample))
    x.append(m2)
    m2+=delm

plt.plot(x,np.array(y))

plt.show()
print(y)
#print('results ',foresee.integrate(bra, 1, m0, m1, m2,0, 10))'''




'''#check for 2 body after running this it gives us what appears to be identical results
pid = f"431"
pid1 = f"-11"
H_attr=df.loc[df['pid']==pid]
#a=df.loc[df['pid']=='411']
tauH=H_attr['tauH (sec)'].values[0]
MH=H_attr['MH (GeV)'].values[0]
VH=H_attr['VH (unitless)'].values[0]
fH=H_attr['fH (GeV)'].values[0]
prefactor=calc_prefactor(MH,VH,tauH,fH)
br=str(prefactor)+f"*coupling**2*mass**2*self.masses('{pid}')*(1.-(mass/self.masses('{pid}'))**2 + 2.* (self.masses('{pid1}')/self.masses('{pid}'))**2 + (self.masses('{pid1}')/mass)**2*(1.-(self.masses('{pid1}')/self.masses('{pid}'))**2)) * np.sqrt((1.+(mass/self.masses('{pid}'))**2 - (self.masses('{pid1}')/self.masses('{pid}'))**2)**2-4.*(mass/self.masses('{pid}'))**2)"
coupling=1
nsample=1000
delm=.1
x=[]
y=[]
mass=.01
for n in range(1,25):
    y.append(eval(br))
    x.append(mass)
    mass+=delm
print(x)
plt.plot(x,np.array(y))
plt.ylim([0,.4])
plt.xlim([0,3])
#plt.plot(x1,y1)
plt.show()
print(br)'''




from skhep.math.vectors import LorentzVector, Vector3D
import random
import math
# prepare output
particles, weights = [], []
m0=1.5
m1=.5
m2=.05
m3=.5
nsample=10


# prepare output
particles, weights = [], []
#create parent 4-vector
p_mother=LorentzVector(0,0,0,m0)

#integration boundary
q2min,q2max = (m2+m3)**2,(m0-m1)**2
cthmin,cthmax = m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0)
mass = m2

#numerical integration
integral=0
for i in range(nsample):

    #Get kinematic Variables
    q2 = random.uniform(q2min,q2max)
    cth = random.uniform(-1,1)
    th = np.arccos(cth)
    q  = math.sqrt(q2)

    #decay meson and V
    cosQ =cth
    phiQ =random.uniform(-math.pi,math.pi)
    cosM =random.uniform(-1.,1.)
    phiM =random.uniform(-math.pi,math.pi)
    p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
    p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

    #branching fraction
    brval  = eval(br)
    brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)

    #save
    #particles.append(p_3)
    #weights.append(brval)





#integrator for HNL decay channels
def integrate(self,br, coupling, m0, m1, m2, m3, nsample):

        # prepare output
        particles, weights = [], []
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #integration boundary
        q2min,q2max = (m2+m3)**2,(m0-m1)**2
        mass = m2

        integral=0

        for i in range(nsample):

            q2 = random.uniform(q2min,q2max)
            q  = math.sqrt(q2)
            E2st = (q**2 - m2**2 + m3**2)/(2*q)
            E3st = (m0**2 - q**2 - m1**2)/(2*q)
            m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
            m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
            cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            EN = random.uniform(ENmin,ENmax)
            #th = np.arccos(EN)
            
            #branching fraction
            brval  = eval(br)
            #print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
            brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)

            #save
            #particles.append(p_3)
            weights.append(brval)
        #print('branching frac', brval)
        print("sum of weights ",sum(weights))
        #return particles,weights
        return(sum(weights))






#branching fractions for HNL decays
#the I's need to be integrated
lambda12=np.sqrt(x**2+y**2-2*x*y-2*y*z-2*x*z)
I1integ=12*(s-x**2-y**2)*(1+z**2-s*lambda12_1*lambda12_2)
I2integ=24*y*z* (1+x**2-s)*lambda12_3*lambda12_4
#for N -> \ell_1^- \ell_2^+ \nu_{\ell 2}
kron=[[1,0,0],[0,1,0],[0,0,1]]
br1=(U**2*GF**2*MN**5/(192*np.pi**3))*I1*(1-kron(l1,l2))
#for N->\nu_{\ell_1} + \ell_2^- + \ell_2^+
br2=(U**2*GF**2*MN**5/(96*np.pi**3))*((glL*glR+kron(l1,l2)*glR)*I2+(glL**2+glR**2+kron(l1,l2)*(1+2*glL))*I1)



############Foresee Junk Code##########
    #this is for the decay into a 3 body with H' being a scalar
'''    def decay_in_restframe_3body1(self, br, coupling, m0, m1, m2, m3, nsample):
        # prepare output
        particles, weights = [], []
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #integration boundary
        q2min,q2max = (m2+m3)**2,(m0-m1)**2

        integral=0
        ####added
        #cth=EN
        ####
        for i in range(nsample):

            #Get kinematic Variables
            q2 = random.uniform(q2min,q2max)
            q  = math.sqrt(q2)
            E2st = (q**2 - m2**2 + m3**2)/(2*q)
            E3st = (m0**2 - q**2 - m1**2)/(2*q)
            m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
            m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
            cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            EN = random.uniform(ENmin,ENmax)
            #th = np.arccos(EN)


            #decay meson and V
            cosQ =random.uniform(-1,1)
            phiQ =random.uniform(-math.pi,math.pi)
            cosM =random.uniform(-1.,1.)
            phiM =random.uniform(-math.pi,math.pi)
            p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
            p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

            #branching fraction
            brval  = eval(br)
            #print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
            brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)

            #save
            particles.append(p_3)
            weights.append(brval)
        return particles,weights'''

'''    #modifying for tau particle added by Alec
    def decay_in_restframe_3body2(self, br, coupling, m0, m1, m2, m3, nsample):
        # prepare output
        particles, weights = [], []
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #integration boundary
        qmin,qmax = (m2+m3),(m0-m1)
        q2min,q2max=0,1 #for tau particles
        #cthmin,cthmax = m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0)
        #print("integration boundaries ",[q2min,q2max],[cthmin,cthmax])
        mass = m2
        #mass=1
        #numerical integration
        integral=0
        ####added
        #cth=EN
        ####
        for i in range(nsample):

            #Get kinematic Variables
            q2 = random.uniform(q2min,q2max)
            q  = math.sqrt(q2)
            E2stmin = (q**2 - m2**2 + m3**2)/(2*q)
            E3stmin = (m0**2 - q**2 - m1**2)/(2*q)
            E2stmax = (q**2 - m2**2 + m3**2)/(2*q)
            E3stmax = (m0**2 - q**2 - m1**2)/(2*q)
            m232min = (E2stmin + E3stmin)**2 - (np.sqrt(E2stmin**2 - m3**2) + np.sqrt(E3stmin**2 - m1**2))**2
            m232max = (E2stmax + E3stmax)**2 - (np.sqrt(E2stmax**2 - m3**2) - np.sqrt(E3stmax**2 - m1**2))**2
            cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            ENmax = (m232max + qmax**2 - m2**2 - m1**2)/(2*m0)
            ENmin = (m232min + qmin**2 - m2**2 - m1**2)/(2*m0)
            EN = random.uniform(ENmin,ENmax)
            #th = np.arccos(EN)


            #decay meson and V
            cosQ =random.uniform(-1,1)
            phiQ =random.uniform(-math.pi,math.pi)
            cosM =random.uniform(-1.,1.)
            phiM =random.uniform(-math.pi,math.pi)
            p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
            p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

            #branching fraction
            brval  = eval(br)
            #print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
            brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)

            #save
            particles.append(p_3)
            weights.append(brval)
        return particles,weights

    #this is for the 3 body decay with H' being a vector
    def decay_in_restframe_3body2(self, br, coupling, m0, m1, m2, m3, nsample):

            # prepare output
            particles, weights = [], []
            #create parent 4-vector
            p_mother=LorentzVector(0,0,0,m0)

            #integration boundary
            q2min,q2max = (m2+m3)**2,(m0-m1)**2
            cthmin,cthmax = m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0)
            print("integration boundaries ",[q2min,q2max],[cthmin,cthmax])
            mass = m2
            #mass=1
            #numerical integration
            integral=0
            ####added
            #cth=EN
            ####
            for i in range(nsample):

                #Get kinematic Variables
                q2 = random.uniform(q2min,q2max)
                EN = random.uniform(m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0))
                #th = np.arccos(EN)
                q  = math.sqrt(q2)

                #decay meson and V
                cosQ =random.uniform(-1,1)
                phiQ =random.uniform(-math.pi,math.pi)
                cosM =random.uniform(-1.,1.)
                phiM =random.uniform(-math.pi,math.pi)
                p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
                p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

                #branching fraction
                brval  = eval(br)
                print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
                brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)

                #save
                particles.append(p_3)
                weights.append(brval)
            print('branching frac', brval)
            print("sum of weights ",sum(weights))
            return particles,weights'''
def decay_in_restframe_3body(self, br, coupling, m0, m1, m2, m3, nsample,channel):


    # prepare output
    particles, weights = [], []

    #create parent 4-vector
    p_mother=LorentzVector(0,0,0,m0)

    #integration boundary
    q2min,q2max = (m2+m3)**2,(m0-m1)**2
    cthmin,cthmax = -1 , 1
    mass = m2

    #numerical integration
    integral=0
    if channel=='2 body':
        integral=0
        for i in range(nsample):

            #Get kinematic Variables
            q2 = random.uniform(q2min,q2max)
            cth = random.uniform(-1,1)
            th = np.arccos(cth)
            q  = math.sqrt(q2)

            #decay meson and V
            cosQ =cth
            phiQ =random.uniform(-math.pi,math.pi)
            cosM =random.uniform(-1.,1.)
            phiM =random.uniform(-math.pi,math.pi)
            p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
            p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

            #branching fraction
            brval  = eval(br)
            brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)

            #save
            particles.append(p_3)
            weights.append(brval)

        return particles,weights


    if channel=='3 body tau':
        # prepare output
        particles, weights = [], []
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #integration boundary
        qmin,qmax = (m1+m2),(m0-m3)
        q2min,q2max=0,1 #for tau particles
        '''E2st = (q**2 - m2**2 + m3**2)/(2*q)
        E3st = (m0**2 - q**2 - m1**2)/(2*q)
        m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
        m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
        cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
        ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)'''
        #cthmin,cthmax = m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0)
        #print("integration boundaries ",[q2min,q2max],[cthmin,cthmax])
        mass = m2
        #mass=1
        #numerical integration
        integral=0
        ####added
        #cth=EN
        ####
        for i in range(nsample):
            #Get kinematic Variables
            q2 = random.uniform(q2min,q2max)
            q  = math.sqrt(q2)
            E2stmin = (qmin**2 - m2**2 + m3**2)/(2*qmin)
            E3stmin = (m0**2 - qmin**2 - m1**2)/(2*qmin)
            E2stmax = (qmax**2 - m2**2 + m3**2)/(2*qmax)
            E3stmax = (m0**2 - qmax**2 - m1**2)/(2*qmax)
            m232min = (E2stmin + E3stmin)**2 - (np.sqrt(E2stmin**2 - m3**2) + np.sqrt(E3stmin**2 - m1**2))**2
            m232max = (E2stmax + E3stmax)**2 - (np.sqrt(E2stmax**2 - m3**2) - np.sqrt(E3stmax**2 - m1**2))**2
            #cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            #cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            ENmax = (m232max + qmax**2 - m2**2 - m1**2)/(2*m0)
            ENmin = (m232min + qmin**2 - m2**2 - m1**2)/(2*m0)
            ENmin=mass
            mh=self.masses('15')
            mv=self.masses('18')
            ml=self.masses('11')
            ENmax=(mh**2+mass**2-(ml+mv)**2)/(2*mh)
            #print('ENmin, ENmax: ',ENmin,ENmax)
            EN = random.uniform(ENmin,ENmax)
            th = np.arccos(EN)
            #decay meson and V
            cosQ =random.uniform(-1,1)
            phiQ =random.uniform(-math.pi,math.pi)
            cosM =random.uniform(-1.,1.)
            phiM =random.uniform(-math.pi,math.pi)
            p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
            p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

            #branching fraction
            brval  = eval(br)
            #print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
            brval *= (q2max-q2min)*(ENmax-ENmin)/float(nsample)

            #save
            particles.append(p_3)
            weights.append(brval)
        return particles,weights
    if channel=='3 body vector':
        # prepare output
        particles, weights = [], []
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #integration boundary
        qmin,qmax = (m1+m2),(m0-m3)
        q2min,q2max=0,1 #for tau particles
        '''E2st = (q**2 - m2**2 + m3**2)/(2*q)
        E3st = (m0**2 - q**2 - m1**2)/(2*q)
        m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
        m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
        cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
        ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)'''
        #cthmin,cthmax = m3,(m0**2+m3**2-(m2+m1)**2)/(2*m0)
        #print("integration boundaries ",[q2min,q2max],[cthmin,cthmax])
        mass = m2
        #mass=1
        #numerical integration
        integral=0
        ####added
        #cth=EN
        ####
        for i in range(nsample):
            #Get kinematic Variables
            q2 = random.uniform(q2min,q2max)
            q  = math.sqrt(q2)
            E2stmin = (qmin**2 - m2**2 + m3**2)/(2*qmin)
            E3stmin = (m0**2 - qmin**2 - m1**2)/(2*qmin)
            E2stmax = (qmax**2 - m2**2 + m3**2)/(2*qmax)
            E3stmax = (m0**2 - qmax**2 - m1**2)/(2*qmax)
            m232min = (E2stmin + E3stmin)**2 - (np.sqrt(E2stmin**2 - m3**2) + np.sqrt(E3stmin**2 - m1**2))**2
            m232max = (E2stmax + E3stmax)**2 - (np.sqrt(E2stmax**2 - m3**2) - np.sqrt(E3stmax**2 - m1**2))**2
            #cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
            #cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
            ENmax = (m232max + qmax**2 - m2**2 - m1**2)/(2*m0)
            ENmin = (m232min + qmin**2 - m2**2 - m1**2)/(2*m0)
            ENmin=mass
            ENmax=(mh**2+mass**2-(ml+mv)**2)/(2*mh)
            #print('ENmin, ENmax: ',ENmin,ENmax)
            EN = random.uniform(ENmin,ENmax)
            th = np.arccos(EN)
            #decay meson and V
            cosQ =random.uniform(-1,1)
            phiQ =random.uniform(-math.pi,math.pi)
            cosM =random.uniform(-1.,1.)
            phiM =random.uniform(-math.pi,math.pi)
            p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
            p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)

            #branching fraction
            brval  = eval(br)
            #print("q val ",q," EN val ",EN," Br ",brval, " Br ",eval(br))
            brval *= (q2max-q2min)*(ENmax-ENmin)/float(nsample)

            #save
            particles.append(p_3)
            weights.append(brval)
        return particles,weights

'''
        #for 3 body decay into vector meson (original)
        #D->V+l+\nu V here is a vector meson like K^*
        #K+ u \bar{s}; D0 c \bar{u}
        #in particular we are going to do D0->K*- + e+ + N
        #I am assuming Mp is the mass of Ds (scalar) looking at the reference this appears to be correct
        #MV is the mass of D*s
        #0~parent meson; 1~daughter meson; 2~lepton; 3~sterile neutrino
        #self=model
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=410.1*10**(-15)
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)
        VHV=.967 #Vcs matrix element
        pid0="421"
        pid1="-323"
        pid2="-11"
        #pid3=
        A00=.76
        Mp=1.97
        s1A0=.17
        s2A0=0
        V0=1.03
        MV=2.11
        #Ml=self.masses(pid2)
        s1V=.27
        s2V=0
        A10=.66
        s1A1=.3
        s2A1=.2
        A20=.49
        s1A2=.67
        s2A2=.16

        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        f1=f"({V}/(self.masses('{pid0}')+self.masses('{pid1}')))"
        f2=f"((self.masses('{pid0}')+self.masses('{pid1}'))*{A1})"
        f3=f"(-{A2}/(self.masses('{pid0}')+self.masses('{pid1}')))"
        f4=f"((self.masses('{pid1}')*(2*{A0}-{A1}-{A2})+self.masses('{pid0}')*({A2}-{A1}))/q**2)"
        f5=f"({f3}+{f4})"
        #form factors for A0 and V, the form at least
        #s1A0 is sigma_1(A0) etc.
        omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2+m3**2-self.masses('{pid2}')**2-2*self.masses('{pid0}')*EN)"
        #omega=f"self.masses('{pid0}')*m3*EN/q**2"
        Omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2-q**2)"
        prefactor=f"(({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*self.masses('{pid0}')**2))"
        term1=f"({f2}**2/2)*(q**2-m3**2-self.masses('{pid2}')**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
        term2=f"({f5}**2/2)*(m3**2+self.masses('{pid2}')**2)*(q**2-m3**2+self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
        term3=f"2*{f3}**2*self.masses('{pid1}')**2*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)*(m3**2+self.masses('{pid2}')**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
        term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
        term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-self.masses('{pid2}')**2))"
        term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/self.masses('{pid1}')**2)*(m3**2-self.masses('{pid2}')**2)+({Omegasqr}**2/self.masses('{pid1}')**2)*self.masses('{pid2}')**2+2*(m3**2-self.masses('{pid2}')**2)**2-2*q**2*(m3**2+self.masses('{pid2}')**2))"
        term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2)+2*{omegasqr}*(self.masses('{pid2}')**2-m3**2)+{Omegasqr}*(m3**2-self.masses('{pid2}')**2-q**2))"
        term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+self.masses('{pid2}')**2)-2*self.masses('{pid1}')**2*(q**4-(m3**2-self.masses('{pid2}')**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-self.masses('{pid2}')**2)+2*{omegasqr}**2*q**2)"
        bra=prefactor + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
        #MH=self.masses(pid0)
        model.add_production_3bodydecay(
            label= "5_di",
            pid0 = pid0,
            pid1 = "-323",
            pid2 = pid2,
            br = bra,
            generator ="pythia8",
            energy = energy,
            nsample = 100,
            scaling = 0, 
        )
        self=model
        #the bounds of integration are not correct rn.
        #term 8 seems huge
        #print(self.masses(pid0))
        #print(self.masses(pid1))
        #print(self.masses(pid2))

        #print(eval(term8))
        #print(eval(bra))
        m0=self.masses(f'{pid0}')
        m1=self.masses(f'{pid1}')
        m2=self.masses(f'{pid2}')
        m3=0
        nsample=100
        integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
        print(integ)
        #q=1
        #EN=1
        #m3=1
        #print(eval(bra))'''
    
f"{prefactor}**coupling**2*((1-(mass**2/self.masses('{pid0}')))**2-(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+(mass**2/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid0}')+mass)**2/self.masses('{pid0}')**2))))"




#just before integrating code into foresee


''' 
def br_2_body(pid0,pid2):
    if int(pid0)>0:
        pid2=str(-abs(int(pid2)))
    if int(pid0)<0:
        pid2=str(abs(int(pid2)))
    #H_attr=df.loc[df['pid0']==pid0]
    #a=df.loc[df['pid0']==pid0]
    #tauH=H_attr['tauH (sec)'].values[0]
    #MH=H_attr['MH (GeV)'].values[0]
    #VH=H_attr['VH (unitless)'].values[0]
    #fH=H_attr['fH (GeV)'].values[0]

    tauH=self.tau(pid0)
    MH=self.masses(pid0)
    VH=self.VH(pid0)
    fH=self.fH(pid0)

    SecToGev=1./(6.582122*pow(10.,-25.))
    tauH=tauH*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    prefactor=((tauH*GF**2*fH**2*VH**2)/(8*np.pi))
    br=str(prefactor)+f"*coupling**2*mass**2*self.masses('{pid0}')*(1.-(mass/self.masses('{pid0}'))**2 + 2.* (self.masses('{pid2}')/self.masses('{pid0}'))**2 + (self.masses('{pid2}')/mass)**2*(1.-(self.masses('{pid2}')/self.masses('{pid0}'))**2)) * np.sqrt((1.+(mass/self.masses('{pid0}'))**2 - (self.masses('{pid2}')/self.masses('{pid0}'))**2)**2-4.*(mass/self.masses('{pid0}'))**2)"
    return(br)

def br_2_body_tau_H(pid0,pid1):
    self=model
    SecToGev=1./(6.582122*pow(10.,-25.))
    tautau=self.tau(pid0)
    tautau=tautau*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    VH=self.VH(pid1)
    fH=self.VH(pid1)
    Mtau=self.masses(pid0)
    prefactor=tautau*GF**2*VH**2*fH**2*Mtau**3/(16*np.pi)
    br=f"{prefactor}*coupling**2*((1-(mass**2/self.masses('{pid0}')**2))**2-(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+(mass**2/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid1}')+mass)**2/self.masses('{pid0}')**2))))"
    return (br)

#rho meson is assumed to be rho(770)
def br_2_body_tau_rho(pid0,pid1):
    grho=.102
    self=model
    SecToGev=1./(6.582122*pow(10.,-25.))
    tautau=self.tau(pid0)
    tautau=tautau*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    Mtau=self.masses(pid0)
    Mrho=self.masses(pid1)
    VH=self.VH(pid1)
    prefactor=tautau*grho**2*GF**2*VH**2*Mtau**3/(8*np.pi*Mrho**2)
    br=f"{prefactor}*coupling**2*((1-(mass**2/self.masses('{pid0}')**2))**2+(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+((mass**2-2*self.masses('{pid1}')**2)/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid1}')+mass)**2/self.masses('{pid0}')**2))))"
    return(br)

def dbr_3_body_tau(pid0,pid1,pid2):
    SecToGev=1./(6.582122*pow(10.,-25.))
    self.model
    tautau=self.tau(pid0)*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    prefactor=f"({tautau}*coupling**2*{GF}**2*self.masses('{pid0}')**2*EN/(2*np.pi**3))"
    dbr=f"{prefactor}*(1+((mass**2-self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-2*(EN/self.masses('{pid0}')))*(1-(self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*EN*self.masses('{pid0}'))))*np.sqrt(EN**2-mass**2)"
    return(dbr)

def dbr_3_body_pseudo(pid0,pid1,pid2):
    #tauH=410.1*10**(-15) #GeV; lifetime of meson
    tauH=self.tau(pid0) #GeV; lifetime of meson
    #VHHp=0.997 #unitless Vcd matrix element
    VHHp=self.VHHp(pid0,pid1) #unitless Vcd matrix element
    SecToGev=1./(6.582122*pow(10.,-25.))
    tauH=tauH*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    prefactor=f"{tauH}*coupling**2*{VHHp}**2*{GF}**2/(64*np.pi**3*self.masses('{pid0}')**2)"
    #tauH=df.loc[df['pid0']==pid0]['tauH (sec)'].values[0]
    #VHHp=df.loc[df['pid0']==pid0]['VHHp'].values[0]
    #pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    if int(pid0[0])==4:
        channel="D"
    if pid0=='521' or pid0=='511':
        channel="B"
    if pid0=='531':
        channel='Bs'
    if pid0=='541' and pid1=='511':
        channel='Bc->B'
    if pid0=='541' and pid1=='531':
        channel='Bc->Bs'
    if channel=="D":
        f00=.747 #for D mesons
        fp0=f00 #for D mesons
        MV=2.01027 #mass in GeV for D mesons
        MS=2.318    #mass in GeV for D mesons
    if channel=="B":
        f00=0.66
        fp0=f00 #for B mesons
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        #MS=5.4154 for B meson
        MS=1.969
    if channel=="Bs":
        f00=0.57
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->B":
        f00=-0.58
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->Bs":
        f00=-0.61
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    pidk="-321"
    pidpi="211"
    fp=f"{f00}/(1-q**2/{MV}**2)"
    f0=f"{f00}/(1-q**2/{MS}**2)"
    fm=f"({f0}-{fp})*(self.masses('{pid0}')**2-self.masses('{pid1}')**2)/q**2"
    term1=f"({fm})**2*(q**2*(m3**2+self.masses('{pid2}')**2)-(m3**2-self.masses('{pid2}')**2)**2)"
    term2=f"2*({fp})*({fm})*m3**2*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term3=f"(2*({fp})*({fm})*self.masses('{pid2}')**2*(4*EN*self.masses('{pid0}')+ self.masses('{pid2}')**2-m3**2-q**2))"
    term4=f"({fp})**2*(4*EN*self.masses('{pid0}')+self.masses('{pid2}')**2-m3**2-q**2)*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term5=f"-({fp})**2*(2*self.masses('{pid0}')**2+2*self.masses('{pid1}')**2-q**2)*(q**2-m3**2-self.masses('{pid2}')**2)"
    bra=str(prefactor)  + "*(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
    return(bra)

def dbr_3_body_vector(pid0,pid1,pid2):
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
        A00=.76
        Mp=1.97
        s1A0=.17
        s2A0=0
        V0=1.03
        MV=2.11
        s1V=.27
        s2V=0
        A10=.66
        s1A1=.3
        s2A1=.2*0
        A20=.49
        s1A2=.67
        s2A2=.16*0
        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
    #'B^+ -> \bar{D}*^0 + e^+ + N' or 'B^0 -> D*^- + e^+ + N'
    if (pid0 in ['521','-521'] and pid1 in ['423','-423']) or (pid0 in ['511'] and pid1 in ['413','-413']):
        A00=0.69
        Mp=6.277
        s1A0=0.58
        s2A0=0
        V0=0.76
        MV=6.842
        s1V=0.57
        s2V=0
        A10=0.66
        s1A1=0.78
        s2A1=0
        A20=0.62
        s1A2=1.04
        s2A2=0
        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"   
    #'B^0_s -> D^*_s^- + e^+ + N'
    if pid0 in ['531'] and pid1 in ['433','-433']:
        A00=0.67
        Mp=6.842
        s1A0=0.35
        s2A0=0
        V0=0.95
        MV=6.842
        s1V=0.372
        s2V=0
        A10=0.70
        s1A1=0.463
        s2A1=0
        A20=0.75
        s1A2=1.04
        s2A2=0
        A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
        V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
        #form factors for A1 and A2
        A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
        A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
    
    #'B^+_c -> B*^0 + e^+ + N'
    if pid0 in ['541','-541'] and pid1 in ['513','-513']:
        A00=-.27
        mfitA0=1.86
        deltaA0=.13
        V0=3.27
        mfitV=1.76
        deltaV=-.052
        A10=.6
        mfitA1=3.44
        deltaA1=-1.07
        A20=10.8
        mfitA2=1.73
        deltaA2=0.09
        A0=f"{A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2)"
        V=f"{V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2)"
        #form factors for A1 and A2
        A1=f"{A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2)"
        A2=f"{A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2)"
    #'B^+_c -> B^*_s^0+ e^+ + N'
    if pid0 in ['541','-541'] and pid1 in ['533','-533']:
        A00=-.33
        mfitA0=1.86
        deltaA0=.13
        V0=3.25
        mfitV=1.76
        deltaV=-.052
        A10=.4
        mfitA1=3.44
        deltaA1=-1.07
        A20=10.4
        mfitA2=1.73
        deltaA2=0.09
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
    omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2+m3**2-self.masses('{pid2}')**2-2*self.masses('{pid0}')*EN)"
    Omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2-q**2)"
    prefactor=f"(({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*self.masses('{pid0}')**2))"
    term1=f"({f2}**2/2)*(q**2-m3**2-self.masses('{pid2}')**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
    term2=f"({f5}**2/2)*(m3**2+self.masses('{pid2}')**2)*(q**2-m3**2+self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
    term3=f"2*{f3}**2*self.masses('{pid1}')**2*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)*(m3**2+self.masses('{pid2}')**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
    term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
    term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-self.masses('{pid2}')**2))"
    term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/self.masses('{pid1}')**2)*(m3**2-self.masses('{pid2}')**2)+({Omegasqr}**2/self.masses('{pid1}')**2)*self.masses('{pid2}')**2+2*(m3**2-self.masses('{pid2}')**2)**2-2*q**2*(m3**2+self.masses('{pid2}')**2))"
    term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2)+2*{omegasqr}*(self.masses('{pid2}')**2-m3**2)+{Omegasqr}*(m3**2-self.masses('{pid2}')**2-q**2))"
    term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+self.masses('{pid2}')**2)-2*self.masses('{pid1}')**2*(q**4-(m3**2-self.masses('{pid2}')**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-self.masses('{pid2}')**2)+2*{omegasqr}**2*q**2)"
    bra=prefactor + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
    return(bra)

'''
#functions to show br curve
#need to figure out how to delete this function
'''
def integrate_pseud_3body(df,pid0,pid1,pid2,m3,coupling,nsample,channel="D"):
    self=model
    tauH=df.loc[df['pid0']==pid0]['tauH (sec)'].values[0]
    VHHp=df.loc[df['pid0']==pid0]['VHHp'].values[0]
    pid1=df.loc[df['pid0']==pid0]['pid1'].values[0]
    SecToGev=1./(6.582122*pow(10.,-25.))
    tauH=tauH*SecToGev
    GF=1.166378*10**(-5) #GeV^(-2)
    prefactor=f"{tauH}*{coupling}**2*{VHHp}**2*{GF}**2/(64*np.pi**3*self.masses('{pid0}')**2)"
    if channel=="D":
        f00=.747 #for D mesons
        fp0=f00 #for D mesons
        MV=2.01027 #mass in GeV for D mesons
        MS=2.318    #mass in GeV for D mesons
    if channel=="B":
        f00=0.66
        fp0=f00 #for B mesons
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
        #MS=1.969
    if channel=="Bs":
        f00=0.57
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->B":
        f00=-0.58
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    if channel=="Bc->Bs":
        f00=-0.61
        fp0=f00
        MV=6.2749 #for Bc meson, I think this should also work for Bc* meson (probly not actually)
        MS=5.4154 
    m0=self.masses(pid0)
    m1=self.masses(pid1)
    m2=self.masses(pid2)
    pidk="-321"
    pidpi="211"
    fp=f"{f00}/(1-q**2/{MV}**2)"
    f0=f"{f00}/(1-q**2/{MS}**2)"
    fm=f"({f0}-{fp})*(self.masses('{pid0}')**2-self.masses('{pid1}')**2)/q**2"
    term1=f"({fm})**2*(q**2*(m3**2+self.masses('{pid2}')**2)-(m3**2-self.masses('{pid2}')**2)**2)"
    term2=f"2*({fp})*({fm})*m3**2*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term3=f"(2*({fp})*({fm})*self.masses('{pid2}')**2*(4*EN*self.masses('{pid0}')+ self.masses('{pid2}')**2-m3**2-q**2))"
    term4=f"({fp})**2*(4*EN*self.masses('{pid0}')+self.masses('{pid2}')**2-m3**2-q**2)*(2*self.masses('{pid0}')**2-2*self.masses('{pid1}')**2-4*EN*self.masses('{pid0}')-self.masses('{pid2}')**2+m3**2+q**2)"
    term5=f"-({fp})**2*(2*self.masses('{pid0}')**2+2*self.masses('{pid1}')**2-q**2)*(q**2-m3**2-self.masses('{pid2}')**2)"
    bra=str(prefactor)  + "*(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
   
    integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
    return(integ)

#need to eventually delete this as well
def integrate_vec_3body(df,pid0,pid1,pid2,m3,coupling,bra,nsample):
    self=model
    bra=dbr_3_body_vector(df,pid0,pid1,pid2)
    m0=self.masses(f'{pid0}')
    m1=self.masses(f'{pid1}')
    m2=self.masses(f'{pid2}')
    integ=foresee.integrate(bra, 1, m0, m1, m2,m3, nsample)
    return(integ)


def integrate_q2EN(pid0,pid1,pid2,m3,coupling,dbr,nsample):
    m0=self.masses(f'{pid0}')
    m1=self.masses(f'{pid1}')
    m2=self.masses(f'{pid2}')
    integ=foresee.integrate(dbr, 1, m0, m1, m2,m3, nsample)
    return(integ)

#pid0 is parent particle, pid1 is produced meson pid2 is the smaller particle
#modified for felix
def show_br_curve(br,pid0,pid1,pid2,integration,nsample=100):
    if integration=='br':
        self=model
        coupling=1
        delm=.1
        mass0=0
        mass=0.01
        x=[]
        y=[]
        for n in range(1,int((3-mass0)/delm)):
            y.append(eval(br))
            x.append(mass)
            mass+=delm

        plt.plot(x,y)
        plt.xlim([0,3])
        plt.ylim([0,max(y)])
        plt.xlabel(r"$m_N (GeV)$")
        plt.ylabel(r"Br")

    if integration=='dEN':
        dbr=br
        self=model
        m0=self.masses(f"{pid0}")
        m1=self.masses(f"{pid1}")
        m2=self.masses(f"{pid2}")
        qmin,qmax=(m2+m3),(m0-m1)
        coupling=1
        delm=.1
        x=[]
        y=[]
        m3=.01
        for n in range(1,17):
            y.append(foresee.integrate_EN(dbr, 1, m0, m1, m2,m3, nsample))
            x.append(m3)
            m3+=delm
        plt.ylim([0,max(y)])
        plt.xlabel(r"$m_N (GeV)$")
        plt.ylabel(r"Br")
        plt.plot(x,np.array(y))

    if integration=='dq2dEN':
        self=model
        dbr=br
        m0=self.masses(f"{pid0}")
        m1=self.masses(f"{pid1}")
        m2=self.masses(f"{pid2}")
        coupling=1
        delm=.1
        x=[]
        y=[]
        m3=0
        for n in range(1,30):
            y.append(integrate_q2EN(pid0,pid1,pid2,m3,coupling,dbr,nsample))
            #y.append(integrate_pseud_3body(df,pid0,pid1,pid2,m3,1,nsample=nsample,channel="D"))
            x.append(m3)
            m3+=delm
        plt.ylim([0,max(y)])
        plt.xlabel(r"$m_N (GeV)$")
        plt.ylabel(r"Br")
        plt.plot(x,np.array(y))

    plt.show()
'''