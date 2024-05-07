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
#import import_ipynb

#Test select parts of DarkPhoton notebook, applying only pion 2-body decay
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_DarkPhoton_decay():
    
    #Brem Masses

    #Specify mixing
    modelname="DarkPhoton"
    energy = "13.6"
    model = Model(modelname, path="./")
    model.add_production_mixing(
        pid = "113",
        mixing = "coupling * 0.3/5. * self.masses('pid')**2/abs(mass**2-self.masses('pid')**2+self.masses('pid')*self.widths('pid')*1j)",
        generator = ['EPOSLHC', 'SIBYLL', 'QGSJET', 'Pythia8-Forward'],
        energy = energy,
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
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
    
    #Benchmark values
    mass, coupling = 0.05, 1e-5
    
    #Init FORESEE corresponding to FASER
    foresee = Foresee(path=src_path)
    foresee.rng.seed(137)
    foresee.set_model(model=model)
    foresee.set_detector(
        distance=480,
        selection="np.sqrt((x.x)**2 + (x.y)**2)<.1",
        length=1,
        luminosity=300,
    )
    setupnames = ['EPOSLHC', 'SIBYLL', 'QGSJET', 'PYTHIA']
    modes = {'113':  ['EPOSLHC', 'SIBYLL', 'QGSJET', 'Pythia8-Forward']}
    
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
    )
    
    for isetup, setup in enumerate(setupnames):
        print(setup,round(sum(weights[:,isetup]),3))
    
    #Compare result to expected numbers of events
    ref = {
        "EPOSLHC": 0.219,
        "SIBYLL": 0.14,
        "QGSJET": 0,
        "PYTHIA": 0.165,
    }

    for isetup, setup in enumerate(setupnames):
        assert np.isclose(round(sum(weights[:,isetup]),3),ref[setup])

