#To run the tests, isntall pytest and jupyter notebook importer:
#  python3 -m pip install pytest
#  python3 -m pip install import_ipynb
#Then do
#  pytest tests.py

import sys, os
src_path = "../../"
sys.path.append(src_path)
import pytest
import numpy as np
import import_ipynb
import matplotlib as mpl
mpl.use('Agg')  #Supress GUI output

def notebookRunner(notebook,ref):
    """
    Runs the codes provided in model notebooks, automatizes comparison
    of output energy distributions w/o/ need to run notebooks manually
    Parameters
    ----------
        noteboook: str
            Name of the notebook and the model directory, e.g. "DarkPhoton"
        ref: dict{str: float}
            Expected numbers of events, with setupnames as keys.
    Returns
    -------
        None
    """

    #Close previously existing symbolic links to avoid conflicts
    try:
        print('Removing existing symbolic links')
        os.unlink('tmp_notebook.ipynb')
        os.unlink('model')
    except:
        print('No existing links detected')

    #Links from the working directory to relevant Model directory
    os.symlink(src = src_path + 'Models/'+notebook+'/'+notebook+'.ipynb',\
               dst = 'tmp_notebook.ipynb',\
               target_is_directory=False)
    os.symlink(src = src_path + 'Models/'+notebook+'/model',\
               dst = 'model',\
               target_is_directory=True)

    #Import codes from notebook (symbolic link generalizes import cmd)
    from tmp_notebook import setupnames,weights
    
    #Verify output against reference values
    Nevts = [round(sum(weights[:,i]),3) for i in range(len(setupnames))]
    for i in range(len(Nevts)):
        print(setupnames[i],Nevts[i])
        assert np.isclose(Nevts[i],ref[setupnames[i]])


def test_DarkPhoton():
    """
    Main function calls to notebookRunner. 
    Specify the notebooks to be run and the expected values here.
    """

    #Expected numbers of events
    ref = {"EPOSLHC_pT=1":   970.308,\
           "SIBYLL_pT=2":   1412.706,\
           "QGSJET_pT=0.5":  431.748,\
           "PYTHIA_pT=1":    956.652,}
    
    #Run the notebook, compare output to expectation
    notebookRunner(notebook='DarkPhoton',ref=ref)
