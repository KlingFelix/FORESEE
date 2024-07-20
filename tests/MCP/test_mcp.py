#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_DarkPhoton_pion_decay.py

import sys, os
src_path = "../../"
sys.path.append(src_path)

from src.foresee import Utility,Model,Foresee
import pytest
import numpy as np


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_MCP_jpsi():

    # generators
    generators=["Pythia8"]
    
    #Specify pion 2-body decay
    modelname="MCP"
    energy = "14"
    model = Model(modelname, path="./")
    model.add_production_2bodydecay(
        pid0 = "443",
        pid1 = "0",
        br = "2 * 0.0597 * coupling**2 * ((1+2*(mass/self.masses('pid0'))**2)*np.sqrt(1-4*(mass/self.masses('pid0'))**2) )/( (1+2*(self.masses('11')/self.masses('pid0'))**2)*np.sqrt(1-4*(self.masses('11')/self.masses('pid0'))**2) )",
        generator = generators,
        energy = energy,
        nsample = 30,
    )
    model.set_dsigma_drecoil_1d(
        dsigma_der="2.0*3.1415/(137.**2)/self.masses('11') * (1/recoil**2 - mass**2 / (2*self.masses('11')*recoil*energy**2))",
        recoil_max = "2 * self.masses('11') * (energy**2-mass**2) / (self.masses('11')*(2*energy+mass) + mass**2)",
        coupling_ref=1
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
    except: pass

    #Benchmark values
    mass, coupling, = 1, 1e-2
    
    #Init FORESEE corresponding to FASER
    foresee = Foresee(path=src_path)
    foresee.rng.seed(137)
    foresee.set_model(model=model)
    foresee.set_detector(
        distance=620,
        selection="abs(x.x)<0.5 and abs(x.y)<0.5",
        length=7,
        luminosity=3000,
        numberdensity=3.754e+29,
        ermin=0.03,
        ermax=1.,
    )
    
    #Find LLP spectra, save under model dir
    foresee.get_llp_spectrum(mass=mass, coupling=1, do_plot=False)
    
    #Find momenta and weights, write events to hepmc
    _, nsigs, _, _ = foresee.get_events_interaction(
        mass=mass,
        energy=energy,
        couplings=[coupling],
        nsample=1
    )
    
    for igen,gen in enumerate(generators):
        print(gen,round(nsigs[igen][0],3))
    
    #Compare result to expected numbers of events
    ref = {
        "Pythia8": 4670.767,
    }

    for igen,gen in enumerate(generators):
        assert np.isclose(round(nsigs[igen][0],3),ref[gen])
        
        
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_MCP_upsilon():

    # generators
    generators=["Pythia8"]
    
    #Specify pion 2-body decay
    modelname="MCP"
    energy = "14"
    model = Model(modelname, path="./")
    model.add_production_2bodydecay(
        pid0 = "553",
        pid1 = "0",
        br = "2 * 0.0238 * coupling**2 * ((1+2*(mass/self.masses('pid0'))**2)*np.sqrt(1-4*(mass/self.masses('pid0'))**2) )/( (1+2*(self.masses('11')/self.masses('pid0'))**2)*np.sqrt(1-4*(self.masses('11')/self.masses('pid0'))**2) )",
        generator = generators,
        energy = energy,
        nsample = 30,
    )
    model.set_dsigma_drecoil_1d(
        dsigma_der="2.0*3.1415/(137.**2)/self.masses('11') * (1/recoil**2 - mass**2 / (2*self.masses('11')*recoil*energy**2))",
        recoil_max = "2 * self.masses('11') * (energy**2-mass**2) / (self.masses('11')*(2*energy+mass) + mass**2)",
        coupling_ref=1
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
    except: pass

    #Benchmark values
    mass, coupling, = 1, 1e-2
    
    #Init FORESEE corresponding to FASER
    foresee = Foresee(path=src_path)
    foresee.rng.seed(137)
    foresee.set_model(model=model)
    foresee.set_detector(
        distance=620,
        selection="abs(x.x)<0.5 and abs(x.y)<0.5",
        length=7,
        luminosity=3000,
        numberdensity=3.754e+29,
        ermin=0.03,
        ermax=1.,
    )
    
    #Find LLP spectra, save under model dir
    foresee.get_llp_spectrum(mass=mass, coupling=1, do_plot=False)
    
    #Find momenta and weights, write events to hepmc
    _, nsigs, _, _ = foresee.get_events_interaction(
        mass=mass,
        energy=energy,
        couplings=[coupling],
        nsample=1
    )
    
    for igen,gen in enumerate(generators):
        print(gen,round(nsigs[igen][0],3))
    
    #Compare result to expected numbers of events
    ref = {
        "Pythia8": 2.167,
    }

    for igen,gen in enumerate(generators):
        assert np.isclose(round(nsigs[igen][0],3),ref[gen])

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_MCP_rho():

    # generators
    generators=["EPOSLHC","QGSJET","SIBYLL","SIBYLL2.3c"]
    
    #Specify pion 2-body decay
    modelname="MCP"
    energy = "14"
    model = Model(modelname, path="./")
    model.add_production_2bodydecay(
        pid0 = "113",
        pid1 = "0",
        br = "2 * 4.72e-5 * coupling**2 * ((1+2*(mass/self.masses('pid0'))**2)*np.sqrt(1-4*(mass/self.masses('pid0'))**2) )/( (1+2*(self.masses('11')/self.masses('pid0'))**2)*np.sqrt(1-4*(self.masses('11')/self.masses('pid0'))**2) )",
        generator = generators,
        energy = energy,
        nsample = 30,
    )
    model.set_dsigma_drecoil_1d(
        dsigma_der="2.0*3.1415/(137.**2)/self.masses('11') * (1/recoil**2 - mass**2 / (2*self.masses('11')*recoil*energy**2))",
        recoil_max = "2 * self.masses('11') * (energy**2-mass**2) / (self.masses('11')*(2*energy+mass) + mass**2)",
        coupling_ref=1
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
    except: pass

    #Benchmark values
    mass, coupling, = 0.3, 1e-2
    
    #Init FORESEE corresponding to FASER
    foresee = Foresee(path=src_path)
    foresee.rng.seed(137)
    foresee.set_model(model=model)
    foresee.set_detector(
        distance=620,
        selection="abs(x.x)<0.5 and abs(x.y)<0.5",
        length=7,
        luminosity=3000,
        numberdensity=3.754e+29,
        ermin=0.03,
        ermax=1.,
    )
    
    #Find LLP spectra, save under model dir
    foresee.get_llp_spectrum(mass=mass, coupling=1, do_plot=False)
    
    #Find momenta and weights, write events to hepmc
    _, nsigs, _, _ = foresee.get_events_interaction(
        mass=mass,
        energy=energy,
        couplings=[coupling],
        nsample=1
    )
    
    for igen,gen in enumerate(generators):
        print(gen,round(nsigs[0][igen],3))
    
    #Compare result to expected numbers of events
    ref = {
        "EPOSLHC": 65755.138,
        "QGSJET": 0.0,
        "SIBYLL": 44332.146,
        "SIBYLL2.3c": 41054.262,
    }

    for igen,gen in enumerate(generators):
        assert np.isclose(round(nsigs[0][igen],3),ref[gen])

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_MCP_pion():

    # generators
    generators=["EPOSLHC","QGSJET","SIBYLL","SIBYLL2.3c"]
    
    #Specify pion 2-body decay
    modelname="MCP"
    energy = "14"
    model = Model(modelname, path="./")
    model.add_production_3bodydecay(
        pid0 = "111",
        pid1 = "22",
        pid2 = "0",
        br = "2.*0.99 * coupling**2 * 1./137/4./3.1415/q**2 * (1-q**2/self.masses('pid0')**2)**3 * (1-4*mass**2/q**2)**0.5 * (2-(1-4*mass**2/q**2)*np.sin(th)**2)",
        generator = generators,
        energy = energy,
        nsample = 30,
    )
    model.set_dsigma_drecoil_1d(
        dsigma_der="2.0*3.1415/(137.**2)/self.masses('11') * (1/recoil**2 - mass**2 / (2*self.masses('11')*recoil*energy**2))",
        recoil_max = "2 * self.masses('11') * (energy**2-mass**2) / (self.masses('11')*(2*energy+mass) + mass**2)",
        coupling_ref=1
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
    except: pass

    #Benchmark values
    mass, coupling, = 0.02, 1e-3
    
    #Init FORESEE corresponding to FASER
    foresee = Foresee(path=src_path)
    foresee.rng.seed(137)
    foresee.set_model(model=model)
    foresee.set_detector(
        distance=620,
        selection="abs(x.x)<0.5 and abs(x.y)<0.5",
        length=7,
        luminosity=3000,
        numberdensity=3.754e+29,
        ermin=0.03,
        ermax=1.,
    )
    
    #Find LLP spectra, save under model dir
    foresee.get_llp_spectrum(mass=mass, coupling=1, do_plot=False)
    
    #Find momenta and weights, write events to hepmc
    _, nsigs, _, _ = foresee.get_events_interaction(
        mass=mass,
        energy=energy,
        couplings=[coupling],
        nsample=1
    )
    
    for igen,gen in enumerate(generators):
        print(gen,round(nsigs[0][igen],3))
    
    #Compare result to expected numbers of events
    ref = {
        "EPOSLHC": 686.871,
        "QGSJET": 622.675,
        "SIBYLL": 566.725,
        "SIBYLL2.3c": 546.146,
    }

    for igen,gen in enumerate(generators):
        assert np.isclose(round(nsigs[0][igen],3),ref[gen])

