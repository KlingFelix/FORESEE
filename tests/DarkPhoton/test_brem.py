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
    masses_brem = [
        0.01  ,  0.0126,  0.0158,  0.02  ,  0.0251,  0.0316,  0.0398,
        0.0501,  0.0631,  0.0794,  0.1   ,  0.1122,  0.1259,  0.1413,
        0.1585,  0.1778,  0.1995,  0.2239,  0.2512,  0.2818,  0.3162,
        0.3548,  0.3981,  0.4467,  0.5012,  0.5623,  0.6026,  0.631 ,
        0.6457,  0.6607,  0.6761,  0.6918,  0.7079,  0.7244,  0.7413,
        0.7586,  0.7762,  0.7943,  0.8128,  0.8318,  0.8511,  0.871 ,
        0.8913,  0.912 ,  0.9333,  0.955 ,  0.9772,  1.    ,  1.122 ,
        1.2589,  1.4125,  1.5849,  1.7783,  1.9953,  2.2387,  2.5119,
        2.8184,  3.1623,  3.9811,  5.0119,  6.3096,  7.9433, 10.
    ]
    
    #Specify pion 2-body decay
    modelname="DarkPhoton"
    energy = "13.6"
    model = Model(modelname, path="./")
    model.add_production_direct(
        label = "Brem",
        energy = energy,
        condition = ["p.pt<1", "p.pt<2", "p.pt<0.5"],
        coupling_ref=1,
        masses = masses_brem,
    )

    #Dir where to place links to branching ratios, ctau and output results
    try: os.mkdir('model')
    except: pass
    
    #Symbolic links to model-specific tables
    try:
        os.unlink('model/br')
        os.unlink('model/ctau.txt')
        os.unlink('model/direct')
    except: pass
    os.symlink(src = src_path + '../Models/' + modelname + '/model/br',\
               dst = 'model/br',\
               target_is_directory=True)
    os.symlink(src = src_path + '../Models/' + modelname + '/model/ctau.txt',\
               dst = 'model/ctau.txt',\
               target_is_directory=False)
    os.symlink(src = src_path + '../Models/' + modelname + '/model/direct',\
               dst = 'model/direct',\
               target_is_directory=True)

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
    setupnames = ["p.pt<1", "p.pt<2", "p.pt<0.5"]
    modes = {'Brem':  ["p.pt<1", "p.pt<2", "p.pt<0.5"]}
    
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
        "p.pt<1": 4.602,
        "p.pt<2": 4.686,
        "p.pt<0.5": 4.153,
    }

    for isetup, setup in enumerate(setupnames):
        assert np.isclose(round(sum(weights[:,isetup]),3),ref[setup])

