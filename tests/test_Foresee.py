#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_Foresee.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility,Model,Foresee
import pytest
import numpy as np
import import_ipynb

foresee = Foresee(path=src_path)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights():
    """
    Check the sum of xs values for pi+ using EPOSLHC & SIBYLL at various energies
    """
    pid = "211" #pi+ (211)
    
    #Expected values for the sum of flattened list_xs
    ref = {"EPOSLHC_13.6": 1259075965736.8040,\
           "SIBYLL_13.6" :  968262425157.5983,\
           "EPOSLHC_14":   1381338244262.9976,\
           "SIBYLL_14" :    980175192323.9988,\
           "EPOSLHC_27":   1900779251078.6904,\
           "SIBYLL_27" :   1442273621493.8990,\
           "EPOSLHC_100":  3494975701999.6006,\
           "SIBYLL_100" :  2390905532445.802}

    for energy in ["13.6","14","27","100"]:
        for generator in ["EPOSLHC","SIBYLL"]:
            
            #Open file, fetch list of xs
            dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
            filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
            _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
            
            #Check approx agreement of sum of flattened list_xs w/ expected ref
            assert np.isclose(sum(np.array(list_xs).flatten()),\
                              ref[generator+"_"+energy])
