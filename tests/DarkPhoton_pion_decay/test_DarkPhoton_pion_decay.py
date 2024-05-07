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
import import_ipynb

#Test select parts of DarkPhoton notebook, applying only pion 2-body decay
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_DarkPhoton_decay():
    
    #Specify pion 2-body decay
    modelname="DarkPhoton"
    energy = "13.6"
    model = Model(modelname, path="./")
    model.add_production_2bodydecay(
        pid0 = "221",
        pid1 = "22",
        br = "2.*0.39 * coupling**2 * pow(1.-pow(mass/self.masses('pid0'),2),3)",
        generator = ['EPOSLHC', 'SIBYLL', 'QGSJET', 'Pythia8-Forward'],
        energy = energy,
        nsample = 100, 
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
    except: pass
    try: os.mkdir('model/events')
    except: pass
    try: os.mkdir('model/results')
    except: pass
    
    #Symbolic links to model-specific tables
    try:
        os.unlink('model/br')
        os.unlink('model/ctau.txt')
    except: pass
    os.symlink(src = src_path + '../Models/' + modelname + '/model/br',\
               dst = 'model/br',\
               target_is_directory=True)
    os.symlink(src = src_path + '../Models/' + modelname + '/model/ctau.txt',\
               dst = 'model/ctau.txt',\
               target_is_directory=False)

    model.set_ctau_1d(filename="model/ctau.txt",)
    decay_modes = ["e_e", "mu_mu", "pi+_pi-", "pi0_gamma", "pi+_pi-_pi0", "K_K"] 
    model.set_br_1d(
        modes = decay_modes,
        finalstates=[[11,-11], [13,-13], [221,-211], [111,22], None, [321,-321]],
        filenames=["model/br/"+mode+".txt" for mode in decay_modes],
    )
    
    #Benchmark values
    mass, coupling, = 0.05, 3e-5
    
    #Init FORESEE corresponding to FASER2
    foresee = Foresee(path=src_path)
    foresee.rng.seed(137)
    foresee.set_model(model=model)
    foresee.set_detector(
        distance=620, 
        selection="np.sqrt((x.x)**2 + (x.y)**2)<1",    
        length=10, 
        luminosity=3000, 
    )
    setupnames = ['EPOSLHC_pT=1', 'SIBYLL_pT=2', 'QGSJET_pT=0.5', 'PYTHIA_pT=1']
    modes = {'221':  ['EPOSLHC', 'SIBYLL', 'QGSJET'  , 'Pythia8-Forward']}
    
    #Find LLP spectra, save under model dir
    foresee.get_llp_spectrum(mass=mass, coupling=1, do_plot=False)
    
    #Find momenta and weights, write events to hepmc
    momenta, weights, _ = foresee.write_events(
        mass = mass, 
        coupling = coupling, 
        energy = energy, 
        numberevent = 1000,
        filename = "model/events/test.hepmc", 
        return_data = True,
        weightnames=setupnames,
        modes=modes,
        seed=137,
    )
    
    #Compare result to expected numbers of events
    ref = {"EPOSLHC_pT=1":  165.52,\
           "SIBYLL_pT=2":    88.906,\
           "QGSJET_pT=0.5":  46.713,\
           "PYTHIA_pT=1":   122.05,}
    for isetup, setup in enumerate(setupnames):
        assert round(sum(weights[:,isetup]),3) == ref[setup]
