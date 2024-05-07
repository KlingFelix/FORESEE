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
#import import_ipynb

foresee = Foresee(path=src_path)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights_1():
    """
    Check the sum of xs values for pi+ using EPOSLHC & SIBYLL at various energies
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {'EPOSLHC_13.6_211': 1259075965736.8040,
           'SIBYLL_13.6_211': 968262425157.5983,
           'EPOSLHC_14_211': 1381338244262.9976,
           'SIBYLL_14_211': 980175192323.9988,
           'EPOSLHC_27_211': 1900779251078.6904,
           'SIBYLL_27_211': 1442273621493.8990,
           'EPOSLHC_100_211': 3494975701999.6006,
           'SIBYLL_100_211': 2390905532445.802
    }

    for energy in ["13.6","14","27","100"]:
        for generator in ["EPOSLHC","SIBYLL"]:
            for pid in ["211"]:
            
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])

def test_read_list_momenta_weights_2():
    """
    Check the sum of xs values for various light hadrons using EPOSLHC &
    SIBYLL & QGSJET at 14 TeV
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {
        'EPOSLHC_14_111': 1540016521237.8992,
        'EPOSLHC_14_-211': 1354422658626.5984,
        'EPOSLHC_14_221': 168625794754.7002,
        'EPOSLHC_14_321': 163567242000.0,
        'EPOSLHC_14_310': 156484453688.50006,
        'EPOSLHC_14_223': 174369536956.30035,
        'SIBYLL_14_111': 1135037805475.7979,
        'SIBYLL_14_-211': 957309158089.9979,
        'SIBYLL_14_221': 193017427160.60062,
        'SIBYLL_14_321': 126329282125.99944,
        'SIBYLL_14_310': 122040615609.59941,
        'SIBYLL_14_223': 180030783726.6004,
        'QGSJET_14_111': 1620017887078.2053,
        'QGSJET_14_-211': 1337306648835.5032,
        'QGSJET_14_221': 240708917737.39948,
        'QGSJET_14_321': 158783905140.3007,
        'QGSJET_14_310': 155261268435.60068,
        'QGSJET_14_223': 0.0,
    }
    
    for energy in ["14"]:
        for generator in ["EPOSLHC"]:#,"SIBYLL","QGSJET"]:
            for pid in ["111","-211","221","321","310","223"]:
                
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])

def test_read_list_momenta_weights_3():
    """
    Check the sum of xs values for various charm hadrons using NLO-P8 &
    DPMJET & Pythia8 & Pythia8-Forward at 13.6 TeV
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {
        'NLO-P8_13.6_411': 1465801300.7249959,
        'NLO-P8_13.6_-421': 2833497451.5750012,
        'NLO-P8_13.6_4122': 211773228.74999905,
        'DPMJET_13.6_411': 757617140.1999979,
        'DPMJET_13.6_-421': 2402337888.900002,
        'DPMJET_13.6_4122': 221847601.10000074,
        'SIBYLL_13.6_411': 1526990065.6999984,
        'SIBYLL_13.6_-421': 3330583308.1999803,
        'SIBYLL_13.6_4122': 610248228.1000023,
        'Pythia8_13.6_411': 1763738803.700007,
        'Pythia8_13.6_-421': 3428131619.5999875,
        'Pythia8_13.6_4122': 249262647.19999886,
        'Pythia8-Forward_13.6_411': 1393189207.0,
        'Pythia8-Forward_13.6_-421': 2696044660.0,
        'Pythia8-Forward_13.6_4122': 1477589385.0
    }
    
    for energy in ["13.6"]:
        for generator in ["NLO-P8","DPMJET","SIBYLL","Pythia8","Pythia8-Forward"]:
            for pid in ["411","-421", "4122"]:
                
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])
                    
def test_read_list_momenta_weights_4():
    """
    Check the sum of xs values for various beauty hadrons using NLO-P8 &
    NLO-P8-min & NLO-P8-max at 13.6 TeV
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {
        'NLO-P8_13.6_-511': 100674964.26881358,
        'NLO-P8_13.6_521': 101239418.50559343,
        'NLO-P8-Min_13.6_-511': 74871136.7904587,
        'NLO-P8-Min_13.6_521': 75275592.57858993,
        'NLO-P8-Max_13.6_-511': 150104244.28716186,
        'NLO-P8-Max_13.6_521': 150995882.83722088
    }
    
    for energy in ["13.6"]:
        for generator in ["NLO-P8","NLO-P8-Min","NLO-P8-Max"]:
            for pid in ["-511", "521"]:
                
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])
