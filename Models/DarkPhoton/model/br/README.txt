------------------------------------
---- Translation of File naming ----
------------------------------------

-----------------------------------
---- Combined Branching ratios ----
-----------------------------------
can be found in the model folder

invisible: invisible final states= neutrinos + DM (if considered)
visible: all hadronic state and charged leptons
qcd: combined results of non-perturbative and perturbative hadronic results with whatever is valid for a certain mass
leptons: all leptons (charged and neutral leptons)

-----------------------------
---- Individual Channels ----
-----------------------------
in both brs/ and widths/, you can find the Branching ratios/widths for individual channels

---------------------------
---- leptonic channels ----
---------------------------

elec: e+ e-
muon: mu+ mu-
tau: tau+ tau-
nue: nu_e (electron neutrino) nubar_e (electron anti-neutrino)
numu: nu_mu (muon neutrino) nubar_mu (muon anti-neutrino)
nutau: nu_tau (tau neutrino) nubar_tau (tau anti-neutrino)

---------------------------
---- hadronic channels ----
---------------------------

quarks: perturbative quark contribution, notice that this should only bet taken into account 
when the non-perturbative regime transitions into the perturbative regime

2pi: pi+ pi-
3pi: pi+ pi- pi0
4pi: 2pi+2pi- + pi+pi-2pi0
6pi: 3pi+3pi- + 2pi+2pi-2pi0
EtaGamma: eta gamma
EtaOmega: eta omega
EtaPhi: eta phi
EtaPiPi: eta pi+ pi-
EtaPrimePiPi: eta' pi+ pi-
KK: K+K- + Kbar0K0
KKpi: KLKSpi0 + K+K-pi0 + K+-pi-+K0
KKpipi: K+pi-K-pi+ + 3x (KS pi0 K+- pi-+)
OmPiPi: Omegapi+pi- + Omegapi0pi0
Omega Pion: omega pi0 where omega -> pi0 gamma, so effectively pi0 pi0 gamma
PhiPi: phi pi0
PhiPiPi: phipi+pi- + phipi0pi0
PiGamma: pi0 gamma
nnbar = neutron anti-neutron
ppbar = proton anti-proton

--------------------------
--hadronic sub-channels --
--------------------------
files starting with "single" contain branching ratios and widths of channels that have hadronic sub-channels

4pi_c: 2pi+ 2pi-
4pi_n: pi+ pi- 2pi0
6pi_c: 3pi+ 3pi-
6pi_n: 2pi+ 2pi- 2pi0
KK_c: K+ K-
KK_n: Kbar0 K0
KKpi_0: KL KS pi0
KKpi_1: K+ K- pi0
KKpi_2: K+- pi-+ K0
KKpipi_0: K*(890) K- pi+ -> K+ pi- K- pi+
KKpipi_1: K*0(890) K+- pi-+ -> KS pi0 K+- pi-+
KKpipi_2: K*+-(890) KS pi-+ -> K+- pi0 KS pi-+
KKpipi_3: K*+-(890) K-+ pi0 -> KS pi-+ K-+ pi0
OmPiPi_c: Omega pi+ pi-
OmPiPi_n: Omega pi0 pi0
PhiPiPi_c: phi pi+ pi-
PhiPiPi_n: phi pi0 pi0
nnbar = neutron anti-neutron
ppbar = proton anti-proton
