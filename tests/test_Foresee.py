#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest tests.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility, Model,Foresee
import pytest
import numpy as np

foresee = Foresee(path="../")

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights():
    energy = "14"
    generator = "EPOSLHC"
    dirname = foresee.dirpath + "files/hadrons/"+energy+"TeV/"+generator+"/"
    
    #Expected values for the sum of flattened list_xs
    ref={"2212": 112332379150.4998, "211": 1.38133824e+12}
    
    #Check protons (2212) and pi+ (211)
    for pid in ["2212","211"]:
        
        #Open file, fetch list of xs
        filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
        _, _, list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
        print(pid,sum(np.array(list_xs).flatten()))
        
        #Check approx agreement of sum of flattened list_xs w/ expected ref
        assert np.isclose(sum(np.array(list_xs).flatten()), ref[pid])

