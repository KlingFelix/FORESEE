#Contains tests for the Utility class

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest tests.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility
import pytest
import numpy as np

util = Utility()

#List common pdg ids expected to appear in FORESEE computations
testpdgids = [2112, -2112, 2212, -2212, 211  ,-211 , 321 , -321 ,\
              310 ,  130 , 111        , 221        , 331        ,\
              3122, -3122, 3222, -3222, 3112 ,-3112, 3322, -3322,\
              3312, -3312, 3334, -3334, 113        , 223        ,\
              333        , 213 , -213 , 411  ,-411 , 421 , -421 ,\
              431  ,-431 , 4122, -4122, 511  ,-511 , 521 , -521 ,\
              531  ,-531 , 541 , -541 , 5122 ,-5122, 4   , -4   ,\
              5    ,-5   , 11  , -11  , 13   ,-13  , 15  , -15  ,\
              22         , 23         , 24   ,-24  , 25         ,\
              0          , 443        , 100443      , 553       ,\
              100553     , 200553     , 12, -12, 14, -14, 16, -16]

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_charges():
    
    #Photons and neutrinos should have no charge
    for id in [22,12,14,16]: assert util.charges(id)==0        
    
    #Anti-particle charges must be 0 or opposite to particle charge
    for id in [negid for negid in testpdgids if negid<0]:        
        assert util.charges(id) in [0, -1.0*util.charges(abs(id))]  #Check both...
        assert util.charges(abs(id)) in [0, -1.0*util.charges(id)]  #...ways


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_masses():
    #Particle/anti-particle masses must agree
    for id in [negid for negid in testpdgids if negid<0]:        
        assert util.masses(id)==util.masses(abs(id))


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_utility_nans():    
    #Properties must not return not-a-number (nan)
    for id in testpdgids:
        assert not np.isnan(util.charges(id))
        assert not np.isnan(util.masses(id))
        assert not np.isnan(util.ctau(id))
        assert not np.isnan(util.widths(id))


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_utility_infs():    
    #Properties must not return infinity (inf)
    for id in testpdgids:
        assert not np.isinf(util.charges(id))
        assert not np.isinf(util.masses(id))
        assert not np.isinf(util.ctau(id))
        assert not np.isinf(util.widths(id))

