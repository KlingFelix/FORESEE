import matplotlib
import os
from matplotlib import pyplot as plt
import random
import time
import types
from .utils.vectors import *
from .utils.utility import Utility
from .utils.model import Model
from .utils.decay import Decay
from matplotlib import gridspec
from numba import jit


##############################################
##############################################
#  FORESEE Class
##############################################
##############################################

class Foresee(Utility, Decay):

    def __init__(self, path="../../"):

        #Consistently initiate random number generators
        self.rng = random.Random()
        Utility.__init__(self,self.rng)
        Decay.__init__(self,self.rng)

        # initiate properties
        self.model = None
        self.shortlived = {"321": 20, "-321": 20, "321": 20,  }
        self.selection = "np.sqrt(x.x**2 + x.y**2)< 1"
        self.length = 5
        self.luminosity = 3000
        self.distance = 480
        self.channels = None
        self.dirpath = path
        #initiate jit functions by running with dummy input
        _ = self.boostlist(np.array([[0,0,0,1]]),np.array([[0,0,0]]))

    ###############################
    #  Model
    ###############################

    def set_model(self,model):
        self.model = model

    ###############################
    #  Decay in Flight Probability
    ###############################

    def get_decay_prob(self, pid, momentum):
        """
        Get decay probability for a particle

        Parameters
        ----------
        pid: str, int
            Particle PDG ID
        momentum: LorentzVector
            Particle 4-momentum
        Returns
        -------
            The probability of the particle decaying in-flight as a float
        """

        # return 1 when decaying promptly or 0 if negative pz.
        if str(pid) not in ["211","-211","321","-321","310","130"]: return 1
        if momentum.pz<0: return 0

        # lifetime and kinematics
        ctau = self.ctau(str(pid))
        theta=math.atan(momentum.pt/momentum.pz)
        dbarz = ctau * momentum.pz / momentum.m
        dbart = ctau * momentum.pt / momentum.m

        # probability to decay in beampipe
        # TODO: Here the LHC values are hardcoded. Should probably be defined in set_detector()
        if str(pid) in ["130", "310"]:
            ltan, ltas, rpipe = 140., 20., 0.05
            if (theta < 0.017/ltas): probability = 1.- np.exp(- ltan/dbarz)
            elif (theta < 0.05/ltas): probability = 1.- np.exp(- ltas/dbarz)
            else: probability = 1.- np.exp(- rpipe /dbart)
        if str(pid) in ["321","-321","211","-211"]:
            ltas, rpipe = 20., 0.05
            if (theta < 0.05/ltas): probability = 1.- np.exp(- ltas/dbarz)
            else: probability = 1.- np.exp(- rpipe /dbart)
        return probability

    ###############################
    #  Efficient Boost Function
    ###############################

    @staticmethod
    @jit
    def boostlist(arr_particle, arr_boost):
        """
        Boost all 4-momenta in a list

        Parameters
        ----------
        arr_particle: [ [float,float,float,float] , ... ]
            Array of particle 4 momenta to be boosted
        arr_boost: [float,float,float]
            The amounts to boost in x,y,z directions.
            The number of boosts does not need to equal the number of particles

        Returns
        -------
            The theta angles and magnitudes of the boosted particle momenta in a numpy array:
            [ [particle[0] to boost[0]],
              [particle[1] to boost[0]],
              ... ,
              [particle[n] to boost[0]],
              [particle[0] to boost[1]],
              [particle[1] to boost[1]],
              ... ,
              [particle[n] to boost[1]],
              ... ,
              [particle[0] to boost[n']],
              ... ,
              [particle[n] to boost[n']]]
            with theta,|momentum|
        """

        # intialize output
        out, i = np.zeros((len(arr_particle)*len(arr_boost),2)), 0

        # loop over 3D boost vectors
        for bx, by, bz in arr_boost:

            b2 = bx**2 + by**2 + bz**2
            gamma = 1.0 / (1.0 - b2)**0.5
            if b2 > 0.0: gamma2 = (gamma - 1.0) / b2
            else: gamma2 = 0.0

            # Loop over LorentzVectors
            for xx, xy, xz, xt in arr_particle:

                bp = bx * xx + by * xy + bz * xz
                xp = xx + gamma2 * bp * bx - gamma * bx * xt
                yp = xy + gamma2 * bp * by - gamma * by * xt
                zp = xz + gamma2 * bp * bz - gamma * bz * xt
                tp = gamma * (xt - bp)

                pt = np.sqrt(xp**2+yp**2)
                th = math.pi/2 if zp==0 else np.arctan(pt/zp)
                pm = np.sqrt(pt**2+zp**2)

                out[i,0]= th
                out[i,1]= pm
                i+=1
        return out

    ###############################
    #  LLP production
    ###############################

    def get_spectrum_decays(self, mass, coupling, key):
        """
        Get the spectrum corresponding to enabled 2- and 3-body decay modes

        Parameters
        ----------
        mass: float
            The mass of the considered particle
        coupling: float
            Coupling strength
        key: str
            Production mode label

        Returns
        -------
            Momenta and weights in the lab frame as numpy arrays
        """

        # load details of production channel
        pid0 = self.model.production[key]["pid0"]
        pid1 = self.model.production[key]["pid1"]
        pid2 = self.model.production[key]["pid2"]
        br = self.model.production[key]["br"]
        generator = self.model.production[key]["production"]
        energy = self.model.production[key]["energy"]
        nsample_had = self.model.production[key]["nsample_had"]
        nsample = self.model.production[key]["nsample"]
        massrange = self.model.production[key]["massrange"]
        preselectioncut = self.model.production[key]["preselectioncut"]
        integration = self.model.production[key]["integration"]

        # check if in mass range
        if massrange is not None:
            if mass<massrange[0] or mass>massrange[1]: return [], []
        if (self.model.production[key]["type"]=="2body") and (self.masses(pid0)<=self.masses(pid1,mass)+mass): return [], []
        elif (self.model.production[key]["type"]=="3body") and (self.masses(pid0)<=self.masses(pid1,mass)+self.masses(pid2,mass)+mass): return [], []

        # load mother particle spectrum
        filename = self.dirpath + "files/hadrons/"+energy+"TeV.txt.gz"
        keys = [f"{pid0}({gen})" for gen in generator]
        momenta_mother, weights_mother = self.read_list_4momenta_weights(filename, keys,mass=self.masses(pid0), preselectioncut=preselectioncut, nsample=nsample_had)

        # get sample of LLP momenta in the mother's rest frame
        if self.model.production[key]["type"] == "2body":
            m0, m1, m2 = self.masses(pid0), self.masses(pid1,mass), mass
            momenta_llp, weights_llp = self.decay_in_restframe_2body(eval(br), m0, m1, m2, nsample)
        if self.model.production[key]["type"] == "3body":
            m0, m1, m2, m3 = self.masses(pid0), self.masses(pid1,mass), self.masses(pid2,mass), mass
            momenta_llp, weights_llp = self.decay_in_restframe_3body(br, coupling, m0, m1, m2, m3, nsample, integration)

        # boost
        arr_minus_boostvectors = LorentzVectors_to_f_arr(momenta=momenta_mother,mode='boost',boostf=-1)
        arr_momenta_llp = LorentzVectors_to_f_arr(momenta_llp)
        momenta_lab = self.boostlist(arr_momenta_llp, arr_minus_boostvectors)

        # weights
        w_decays = np.array([self.get_decay_prob(pid0, p_mother)*w_mother for w_mother, p_mother in zip(weights_mother,momenta_mother)])
        weights_llp = np.array(weights_llp)
        weights_lab = (weights_llp * w_decays[:, :, np.newaxis])
        weights_lab = np.concatenate([w.T for w in weights_lab])

        #return
        return momenta_lab, weights_lab

    def get_spectrum_mixing(self, mass, coupling, key):
        """
        Get the spectrum corresponding to mixing

        Parameters
        ----------
        mass: float
            The mass of the considered particle
        coupling: float
            Coupling strength
        key: str
            Production mode label

        Returns
        -------
            Momenta as [theta, |p3|] and weights in the lab frame as numpy arrays
        """

        # load details of production channel
        pid0 = self.model.production[key]["pid0"]
        mixing = self.model.production[key]["mixing"]
        generator = self.model.production[key]["production"]
        energy = self.model.production[key]["energy"]
        massrange = self.model.production[key]["massrange"]

        # check if in mass range
        if massrange is not None:
            if mass<massrange[0] or mass>massrange[1]: return [], []

        # load mother particle spectrum
        filename = self.dirpath + "files/hadrons/"+energy+"TeV.txt.gz"
        keys = [f"{pid0}({gen})" for gen in generator]
        momenta_mother, weights_mother = self.read_list_4momenta_weights(filename, keys,mass=self.masses(pid0))

        # z-axis angles and 3-momentum magnitudes from momenta
        momenta_lab = theta_p3_f_arr(momenta=momenta_mother)

        # weights
        if type(mixing)==str:
            mixing_angle = eval(mixing)
            weights_lab = np.array([w_mother*mixing_angle**2 for w_mother in weights_mother])
        else:
            weights_lab = np.array([w_mother*mixing(mass, coupling, p_mother)**2 for p_mother,w_mother in zip(momenta_mother,weights_mother)])

        #return
        return momenta_lab, weights_lab

    def get_spectrum_direct(self, mass, coupling, key):
        """
        Get the spectrum corresponding to direct production

        Parameters
        ----------
        mass: float
            The mass of the considered particle
        coupling: float
            Coupling strength
        key: str
            Production mode label

        Returns
        -------
            Momenta and weights in the lab frame as numpy arrays
        """
        # load details of production channel
        label = key
        energy = self.model.production[key]["energy"]
        coupling_ref =  self.model.production[key]["coupling_ref"]
        condition =  self.model.production[key]["condition"]
        configuration =  self.model.production[key]["configuration"]
        masses =  self.model.production[key]["masses"]

        #determined mass benchmark below / above mass
        if mass<masses[0] or mass>masses[-1]: return [], []
        mass0, mass1 = 0, 1e10
        for xmass in masses:
            if xmass<=mass and xmass>mass0: mass0=xmass
            if xmass> mass and xmass<mass1: mass1=xmass

        #load benchmark data
        filename0 = self.model.modelpath+"model/direct/"+energy+"TeV/"+energy+"TeV_"+str(mass0)+".txt.gz"
        filename1 = self.model.modelpath+"model/direct/"+energy+"TeV/"+energy+"TeV_"+str(mass1)+".txt.gz"
        try:
            momenta_llp0, weights_llp0 = self.read_list_4momenta_weights(filename0, configuration,mass=mass0,nocuts=True)
            momenta_llp1, weights_llp1 = self.read_list_4momenta_weights(filename1, configuration, mass=mass1,nocuts=True)
        except:
            print ("did not find file:", filename0, "or", filename1)
            return [], []

        # z-axis angles and 3-momentum magnitudes from momenta
        momenta_lab = theta_p3_f_arr(momenta=momenta_llp0)

        # weights
        if len(condition)>1:
            factors = np.array([[0 if (c is not None) and (eval(c)==0) else 1 if c is None else eval(c) for p in momenta_llp0] for c in condition]).T
            weights_llp = [ w_lpp0 + (w_lpp1-w_lpp0)/(mass1-mass0)*(mass-mass0) for  w_lpp0, w_lpp1 in zip(weights_llp0.T[0], weights_llp1.T[0])]
            weights_lab = np.array([w*coupling**2/coupling_ref**2*factor for w,factor in zip(weights_llp, factors)])
        else:
            c = condition[0]
            factors = np.array([0 if (c is not None) and (eval(c)==0) else 1 if c is None else eval(c) for p in momenta_llp0])
            weights_llp = [ w_lpp0 + (w_lpp1-w_lpp0)/(mass1-mass0)*(mass-mass0) for  w_lpp0, w_lpp1 in zip(weights_llp0.T, weights_llp1.T)]
            weights_lab = np.array([w*coupling**2/coupling_ref**2*factor for w,factor in zip(weights_llp, factors)]).T

        #return
        return momenta_lab, weights_lab

    def get_llp_spectrum(self, mass, coupling, channels=None, do_plot=False, save_file=True):
        """
        Get the spectrum of LLPs

        Parameters
        ----------
        mass: float
            The mass of the considered particle
        coupling: float
            Coupling strength
        channels: [str]
            List of modes to consider, used as production dictionary keys
        do_plot: bool
            Flag whether to produce a plot based on the resulting histogram or not
        save_file: bool
            Flag whether to call convert_to_hist_list saving the results into a numpy file

        Returns
        -------
            If do_plot, the output of convert_to_hist_list. Else None.
        """
        # prepare output
        if channels is None: channels = [key for key in self.model.production.keys()]
        momenta_all, weights_all = np.array([[0.1,0.1]]), [0 ]
        dirname = self.model.modelpath+"model/LLP_spectra/"
        if not os.path.exists(dirname): os.mkdir(dirname)

        list_w, keys_llp, energy = [], [], 0

        # loop over channels
        for key in self.model.production.keys():

            # selected channels only
            if key not in channels: continue
            if self.model.production[key]["type"] in ["2body", "3body"]:
                momenta, weights = self.get_spectrum_decays(mass=mass,coupling=coupling,key=key)
            if self.model.production[key]["type"]=="mixing":
                momenta, weights = self.get_spectrum_mixing(mass=mass,coupling=coupling,key=key)
            if self.model.production[key]["type"]=="direct":
                momenta, weights = self.get_spectrum_direct(mass=mass,coupling=coupling,key=key)

            #return statistcs
            if save_file==True and len(momenta)>0:
                energy = self.model.production[key]["energy"]
                for iproduction, production in enumerate(self.model.production[key]["production"]):
                    key_llp = f"{key}({production})"
                    data = self.convert_to_hist_list(momenta, weights[:,iproduction], do_plot=False)
                    list_w.append(data[2])
                    keys_llp.append(key_llp)

            #store mome
            if do_plot and len(momenta)>0:
                momenta_all = np.concatenate((momenta_all, momenta), axis=0)
                weights_all = np.concatenate((weights_all, weights[:,0]), axis=0)

        if save_file==True and energy != 0:
            logth, logp = data[0], data[1]
            filename = dirname+energy+"TeV_"+"m_"+str(mass)+".txt.gz"
            self.write_list_angle_momenta_weights(logth, logp, list_w, keys_llp, filename)

        #return
        if do_plot:
            return self.convert_to_hist_list(momenta_all, weights_all, do_plot=do_plot)[0]

    ###############################
    #  Detector Specifics
    ###############################

    def set_detector(
            self,
            distance=480,
            distance_prod=0,
            selection="np.sqrt(x.x**2 + x.y**2)< 1",
            length=5,
            luminosity=3000,
            channels=None,
            numberdensity=3.754e+29,
            ermin=0.03,
            ermax=1,
            efficiency=1,
            photon_yield = 17.4e+3*0.64,
            n_layer = 4,
            length_layer = 100,
            efficiency_layer = 0.1,
            density_layer=1.023,
        ):
        """
        Specify the detector configuration

        Parameters
        ----------
        distance: float
            Detector distance from collider central experiment interaction point
        distance_prod: float
            Distance where LLPs are produced if not at the collider interaction point
        selection: str
            Expression quantifying the selection rule
        length: float
            Detector length in meters, along z-direction i.e. line of sight
        luminosity: float
            Expected luminosity in fb^-1
        channels: [str]
            Decay channels to consider. Default None implies all.
        numberdensity: float
            The number density of target particles in the detector in m^-3
        ermin: float
            Minimum particle energy
        ermax: float
            Maximum particle energy
        efficiency: str, float, int, types.FunctionType
            Detector efficiency function

        Returns
        -------
            None
        """

        self.distance=distance
        self.distance_prod=distance_prod
        self.selection=selection
        self.length=length
        self.lfront=distance-distance_prod
        self.lback=distance-distance_prod+length
        self.luminosity=luminosity
        self.channels=channels
        self.numberdensity=numberdensity
        self.ermin=ermin
        self.ermax=ermax
        self.efficiency=efficiency

        # for MCPs only
        self.photon_yield=photon_yield
        self.n_layer=n_layer
        self.length_layer=length_layer
        self.efficiency_layer=efficiency_layer
        self.density_layer=density_layer

        #make evaluation of selection faster
        selection = selection.replace("x.x", "x").replace("x.y", "y").replace("x.z", "z")
        selection = selection.replace("p.x", "px").replace("p.y", "py").replace("p.z", "pz")
        lambdastr_selection = f'lambda x,y,z,px,py,pz: {selection}'
        lambdafunc_selection = eval(lambdastr_selection)
        self.numbafunc_selection = jit(nopython=True)(lambdafunc_selection)

        #make evaluation of efficiency faster
        lambdastr_efficiency = f'lambda energy,x,y: {efficiency}'
        lambdafunc_efficiency = eval(lambdastr_efficiency)
        self.numbafunc_efficiency = jit(nopython=True)(lambdafunc_efficiency)


    def event_passes(self,momentum):
        """
        Check if an event passes momentum criteria

        Parameters
        ----------
        momentum: LorentzVector
            The momentum vector to compare against the selection criteria specified for Foresee
        Returns
        -------
            The result as a bool
        """
        # obtain 3-momentum
        p=Vector3D(momentum.px,momentum.py,momentum.pz)
        # get position of
        x=float(self.distance/p.z)*p
        if type(x) is np.ndarray: x=Vector3D(x[0],x[1],x[2])
        # check if it passes
        if eval(self.selection): return True
        else:return False

    ###############################
    #  Get Events in Detector
    ###############################

    def get_events(self, mass, energy,
            modes = None,
            couplings = np.logspace(-8,-3,51),
            nsample = 1,
            preselectioncuts = "th<0.01",
            coup_ref = 1,
        ):
        """
        The numbers of expected events in the specified detector,
        assuming given production modes, couplings and cuts.

        Parameters
        ----------
        mass: float
            Particle mass
        energy: str
            Collider sqrt(S) in TeV
        modes: dict
            Production modes to consider as keys,
            list of prediction labels (e.g. generator names) as values
        couplings: numpy array
            The couplings to scan over
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
        preselectioncuts: str
            Expression defining cuts to be used e.g. "th<0.01 and p>100"
        coup_ref: float
            Reference coupling value
        Returns

        -------
            Lists of couplings, ctaus, sum of output weights, output momenta, output weight array
        """

        # setup different couplings to scan over
        model = self.model
        if modes is None: modes = {key: model.production[key]["production"] for key in model.production.keys()}
        nprods = max([len(modes[key]) for key in modes.keys()])
        for key in modes.keys(): modes[key] += [modes[key][0]] * (nprods - len(modes[key]))

        #setup ctau, branching fractions
        ctaus = np.array([model.get_ctau(mass, coupling) for coupling in couplings])
        if self.channels is None: brs = np.array([1 for coupling in couplings])
        else: brs = np.array([sum([model.get_br(channel, mass, coupling) for channel in self.channels]) for coupling in couplings])

        # setup output arrays
        output_p, output_w = [LorentzVector(0,0,0,0)], [np.array([[0 for _ in range(nprods)] for _ in couplings])]

        # loop over production modes
        for key in modes.keys():

            productions = model.production[key]["production"]
            dirname = self.model.modelpath+"model/LLP_spectra/"
            filename = dirname+energy+"TeV_"+"m_"+str(mass)+".txt.gz"
            keys_llp  = [f"{key}({production})" for production in modes[key]]

            # try Load Flux file
            try:
                momenta, weights =self.read_list_4momenta_weights(filename=filename, keys=keys_llp, mass=mass, nsample=nsample, preselectioncut=preselectioncuts)
            except:
                continue
            
            # get coupling factors
            cfacs = np.array([model.get_production_scaling(key, mass, coupling, coup_ref) for coupling in couplings])

            # filter events that pass selection
            momenta = LorentzVectors_to_f_arr(momenta)
            #TODO below could likely be optimized with skheparrays, if momenta not turned into arrays just yet
            position = [ [self.distance/p[2]*p[0], self.distance/p[2]*p[1], self.distance] for p in momenta]
            filtered = [(p, x, w) for p,x,w in zip(momenta, position, weights) if self.numbafunc_selection(x[0],x[1],x[2],p[0],p[1],p[2])]
            if not filtered: continue
            momenta, positions, weights = zip(*filtered)

            # weight of this event incl. lumi and efficiency
            weights = [w * self.numbafunc_efficiency(p[3],x[0],x[1]) * self.luminosity * 1000 for (p,x,w) in zip(momenta, positions, weights)]

            # loop over particles, and record probablity to decay in volume
            # TODO could this be optimized?
            for p,w in zip(momenta, weights):
                dbars = ctaus * p[2] / mass
                prob_decays = np.exp(-self.lfront / dbars) - np.exp(-self.lback / dbars)
                wgts = np.outer(cfacs * prob_decays * brs,w)
                output_w.append(wgts)

            #TODO do we want to return a list of LorentzVectors or a skheparray w/ new skhep?
            #TODO could also have an auxiliary function doing the inverse of LorentzVectors_to_f_arr
            output_p += [LorentzVector(p[0],p[1],p[2],p[3]) for p in momenta]

        # prepare results directory
        # TODO: THIS SHOULD NOT BE HERE
        dirname = self.model.modelpath+"model/results/"
        if not os.path.exists(dirname): os.mkdir(dirname)

        #reshape
        return couplings, ctaus, sum(output_w), output_p, np.transpose(np.array(output_w), (1, 0, 2))

    def get_events_interaction(self, mass, energy,
            modes = None,
            couplings = np.logspace(-8,-3,51),
            nsample = 1,
            preselectioncuts = "th<0.01 and p>100",
            coup_ref = 1,
        ):
        """
        Get the expected number of signal events in the specified detector

        Parameters
        ----------
        mass: float
            Particle mass
        energy: str
            Collider sqrt(S) in TeV
        modes: None, dict
            If specified, a dictionary with production modes to consider as keys,
            and lists of prediction labels (e.g. generator names) as values
        couplings: numpy array
            The couplings to scan over
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
            Relevant for non-cylindrical or off-axis detectors
        preselectioncuts: str
            Expression defining cuts to be used e.g. "th<0.01 and p>100"
        coup_ref: float
            Reference coupling value

        Returns
        -------
            List of couplings, umber of nsignals as numpy array, stat momenta, stat weights as numpy array
        """

        # setup different couplings to scan over
        model = self.model
        if modes is None: modes = {key: model.production[key]["production"] for key in model.production.keys()}
        nprods = max([len(modes[key]) for key in modes.keys()])
        for key in modes.keys(): modes[key] += [modes[key][0]] * (nprods - len(modes[key]))

        # setup output arrays
        output_p, output_w = [LorentzVector(0,0,0,0)], [np.array([[0 for _ in range(nprods)] for _ in couplings])]

        # unit conversion
        GeV2_in_invmeter2 = (5e15)**2

        # loop over production modes
        for key in modes.keys():

            productions = model.production[key]["production"]
            dirname = self.model.modelpath+"model/LLP_spectra/"
            filename = dirname+energy+"TeV_"+"m_"+str(mass)+".txt.gz"
            keys_llp  = [f"{key}({production})" for production in modes[key]]

            # try Load Flux file
            try:
                momenta, weights=self.read_list_4momenta_weights(
                    filename=filename, keys=keys_llp, mass=mass, nsample=nsample, preselectioncut=preselectioncuts)
            except:
                continue
            
            # setup coupling-factors
            cfacs = np.array([model.get_production_scaling(key, mass, coupling, coup_ref) for coupling in couplings])

            # filter events that pass selection
            momenta = LorentzVectors_to_f_arr(momenta)
            #TODO the below could likely be optimized w/ skheparray
            position = [ [self.distance/p[2]*p[0], self.distance/p[2]*p[1], self.distance] for p in momenta]
            filtered = [(p, w) for p,x,w in zip(momenta, position, weights) if self.numbafunc_selection(x[0],x[1],x[2],p[0],p[1],p[2])]
            if not filtered: continue
            momenta, weights = zip(*filtered)

            # weight of this event incl. lumi
            weights = [w * self.luminosity * 1000 for (p,w) in zip(momenta, weights)]

            # loop over particles, and record interaction probablity
            #TODO could this be optimized?
            for p,w in zip(momenta, weights):
                sigmaint = np.array(model.get_sigmaints(mass, couplings, p[3], self.ermin, self.ermax))
                lamdaint = 1. / self.numberdensity / sigmaint * GeV2_in_invmeter2
                prob_int = self.length / lamdaint
                wgts = np.outer(cfacs * prob_int, w)
                output_w.append(wgts)

            #TODO reconsider output format? Use skheparray for new skhep?
            output_p += [LorentzVector(p[0],p[1],p[2],p[3]) for p in momenta]

        return couplings, sum(output_w), output_p, np.transpose(np.array(output_w), (1, 0, 2))


    def get_events_ionisation(self, mass, energy,
            modes=None,
            couplings = np.logspace(-5,0,51),
            nsample=1,
            preselectioncuts="th<0.01 and p>100",
            coup_ref=1,
        ):
        """
        Get the expected number of signal events in the specified detector

        Parameters
        ----------
        mass: float
            Particle mass
        energy: str
            Collider sqrt(S) in TeV
        modes: None, dict
            If specified, a dictionary with production modes to consider as keys,
            and lists of prediction labels (e.g. generator names) as values
        couplings: numpy array
            The couplings to scan over
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
            Relevant for non-cylindrical or off-axis detectors
        preselectioncuts: str
            Expression defining cuts to be used e.g. "th<0.01 and p>100"
        coup_ref: float
            Reference coupling value

        Returns
        -------
            List of couplings, number of nsignals as numpy array, stat momenta, stat weights as numpy array
        """

        # setup different couplings to scan over
        model = self.model
        if modes is None: modes = {key: model.production[key]["production"] for key in model.production.keys()}
        nprods = max([len(modes[key]) for key in modes.keys()])
        for key in modes.keys(): modes[key] += [modes[key][0]] * (nprods - len(modes[key]))

        # setup output arrays
        output_p, output_w = [LorentzVector(0,0,0,0)], [np.array([[0 for _ in range(nprods)] for _ in couplings])]

        # loop over production modes
        for key in modes.keys():

            productions = model.production[key]["production"]
            dirname = self.model.modelpath+"model/LLP_spectra/"
            filename = dirname+energy+"TeV_"+"m_"+str(mass)+".txt.gz"
            keys_llp  = [f"{key}({production})" for production in modes[key]]
            # try Load Flux file
            try:
                momenta, weights=self.read_list_4momenta_weights(
                    filename=filename, keys=keys_llp, mass=mass, nsample=nsample, preselectioncut=preselectioncuts)
            except:
                continue

            #setup coupling-factors
            cfacs = np.array([model.get_production_scaling(key, mass, coupling, coup_ref) for coupling in couplings])

            # filter events that pass selection
            momenta = LorentzVectors_to_f_arr(momenta)
            #TODO could the rest of this function be optimized further w/ skheparrays? Similar to above comments
            position = [ [self.distance/p[2]*p[0], self.distance/p[2]*p[1], self.distance] for p in momenta]
            filtered = [(p, w) for p,x,w in zip(momenta, position, weights) if self.numbafunc_selection(x[0],x[1],x[2],p[0],p[1],p[2])]
            if not filtered: continue
            momenta, weights = zip(*filtered)

            # weight of this event incl. lumi
            weights = [w * self.luminosity * 1000 for (p,w) in zip(momenta, weights)]

            #factor
            factor = self.density_layer * self.efficiency_layer * self.photon_yield * self.length_layer

            # loop over particles, and record probablity to interact in volume
            for p,w in zip(momenta, weights):
                dEdxs = model.get_dEdx(mass, couplings, p[3])
                nPhotoElec = factor*np.array(dEdxs)
                Prob_det = (1 - np.exp(-abs(nPhotoElec)))**self.n_layer
                wgts = np.outer(cfacs * Prob_det, w)
                output_w.append(wgts)

            output_p += [LorentzVector(p[0],p[1],p[2],p[3]) for p in momenta]

        return couplings, sum(output_w), output_p, np.transpose(np.array(output_w), (1, 0, 2))

    ###############################
    #  Export Results as HEPMC File
    ###############################

    def decay_llp(self, momentum, pids):
        """
        Handle to call the appropriate decay functions (2-body, 3-body, ...) based on the length of input pids

        Parameters
        ----------
        momentum: LorentzVector
            Initial state particle 4-momentum
        pids: [str] / [int]
            Final state particle PDG IDs

        Returns
        -------
            Lists of final state particle PDG IDs and momenta as LorentzVectors
        """

        # unspecified decays - can't do anything
        if pids==None:
            return None, []
        # 1-body decays
        elif len(pids)==1:
            p1 = LorentzVector(momentum.x,momentum.y,momentum.z,np.sqrt(momentum.p**2 + self.masses(str(pids[0]))**2 ) )
            return pids, [p1]
        # 2-body decays
        elif len(pids)==2:
            phi = self.rng.uniform(-math.pi,math.pi)
            cos = self.rng.uniform(-1.,1.)
            m0, m1, m2 = momentum.m, self.masses(str(pids[0])), self.masses(str(pids[1]))
            p1, p2 = self.twobody_decay(momentum,m0,m1,m2,phi,cos)
            return pids, [p1,p2]
        # 3-body decays
        elif len(pids)==3:
            m0 = momentum.m
            m1, m2, m3 = self.masses(str(pids[0])), self.masses(str(pids[1])), self.masses(str(pids[2]))
            p1, p2, p3 = self.threebody_decay_pure_phase_space(momentum,m0,m1,m2,m3)
            return pids, [p1,p2,p3]
        # not 2/3 body decays - not yet implemented
        else:
            return None, []

    def write_hepmc_file(self, data, filename, weightnames):
        """
        Store the resulting evevnts into a hepmc file

        Parameters
        ----------
        data: [[float, LorentzVector, LorentzVector, [str] or [int], [LorentzVector]]
            A table of events, with each event entry specified in terms of
            weights, position, momentum, pids and finalstate particle momenta
        filename: str
            The name of the output file
        weightnames: [str]
            Labels for the weights, to be included in header line

        Returns
        -------
            None
        """

        # open file
        f= open(filename,"w")
        f.write("HepMC::Version 2.06.09\n")
        f.write("HepMC::IO_GenEvent-START_EVENT_LISTING\n")

        # loop over events
        for ievent, (weights, position, momentum, pids, finalstate) in enumerate(data):

            #TODO assert equal numbers of weights and weightnames

            # Event Info
            # int: event number / int: number of multi particle interactions [-1] / double: event scale [-1.] / double: alpha QCD [-1.] / double: alpha QED [-1.] / int: signal process id [0] / int: barcode for signal process vertex [-1] / int: number of vertices in this event [1] /  int: barcode for beam particle 1 [1] / int: barcode for beam particle 2 [0] /  int: number of entries in random state list (may be zero) [0] / long: optional list of random state integers [-] /  int: number of entries in weight list (may be zero) [0] / double: optional list of weights [-]
            f.write("E "+str(ievent)+" -1 -1. -1. -1. 0 -1 1 1 0 0 " +str(len(weightnames))+ " "+" ".join([str(w) for w in weights])+"\n")
            # int: number of entries in weight name list [0] /  std::string: list of weight names enclosed in quotes
            f.write("N "+str(len(weightnames))+" "+" ".join(["\""+name+"\"" for name in weightnames]) + "\n")
            # std::string: momentum units (MEV or GEV) [GeV] /  std::string: length units (MM or CM) [MM]
            f.write("U GEV MM\n")
            # double: cross section in pb /  double: error associated with this cross section in pb [0.]
            f.write("C "+str(weights[0])+" 0.\n")
            # PDF info - doesn't apply here
            f.write("F 0 0 0 0 0 0 0 0 0\n")

            #vertex
            npids= "0" if pids==None else str(len(pids))
            f.write("V -1 0 ")
            f.write(str(round(position.x*1000,10))+" ")
            f.write(str(round(position.y*1000,10))+" ")
            f.write(str(round(position.z*1000,10))+" ")
            f.write(str(round(position.t*1000,10))+" ")
            f.write("1 "+npids+" 0\n")

            # LLP
            status= "1" if pids==None else "2"
            f.write("P 1 32 ") # First particle, ID for Z'
            f.write(str(round(momentum.px,10))+" ")
            f.write(str(round(momentum.py,10))+" ")
            f.write(str(round(momentum.pz,10))+" ")
            f.write(str(round(momentum.e, 10))+" ")
            f.write(str(round(momentum.m, 10))+" ")
            f.write(status+ " 0 0 -1 0\n")

            #decay products
            if pids is None: continue
            for iparticle, (pid, particle) in enumerate(zip(pids, finalstate)):
                f.write("P "+str(iparticle+2)+" "+str(pid)+" ")
                f.write(str(round(particle.px,10))+" ")
                f.write(str(round(particle.py,10))+" ")
                f.write(str(round(particle.pz,10))+" ")
                f.write(str(round(particle.e, 10))+" ")
                f.write(str(round(particle.m, 10))+" ")
                f.write("1 0 0 0 0\n")

        # close file
        f.write("HepMC::IO_GenEvent-END_EVENT_LISTING\n")
        f.close()

    def write_csv_file(self, data, filename):
        """
        Write results into a comma-separated-values format file

        Parameters
        ----------
        data: [[float, LorentzVector, LorentzVector, [str] or [int], [LorentzVector]]
            A table of events, with each event entry specified in terms of
            weights, position, momentum, pids and finalstate particle momenta
        filename: str
            The name of the output file

        Returns
        -------
            None
        """

        # open file
        f= open(filename,"w")
        f.write("particle_id,particle_type,process,vx,vy,vz,vt,px,py,pz,m,q\n")

        # loop over events
        for ievent, (weights, position, momentum, pids, finalstate) in enumerate(data):

            #vertex
            vx, vy = round(position.x*1000,10), round(position.y*1000,10)
            vz, vt = round(position.z*1000,10), round(position.t*1000,10)

            # LLP
            px, py = round(momentum.px,10), round(momentum.py,10)
            pz, m, q = round(momentum.pz,10), round(momentum.m ,10), 0
            particle_id, particle_type, process = ievent, 32, 0
            f.write(str(particle_id)+","+str(particle_type)+","+str(process)+",")
            f.write(str(vx)+","+str(vy)+","+str(vz)+","+str(vt)+",")
            f.write(str(px)+","+str(py)+","+str(pz)+","+str(m)+","+str(q)+"\n")

            #decay products
            if pids is None: continue
            for iparticle, (pid, particle) in enumerate(zip(pids, finalstate)):
                px, py = round(particle.px,10), round(particle.py,10)
                pz, m, q = round(particle.pz,10), round(particle.m ,10), self.charges(str(pid))
                particle_id, particle_type, process = ievent, pid, 0
                f.write(str(particle_id)+","+str(particle_type)+","+str(process)+",")
                f.write(str(vx)+","+str(vy)+","+str(vz)+","+str(vt)+",")
                f.write(str(px)+","+str(py)+","+str(pz)+","+str(m)+","+str(q)+"\n")

        # close file
        f.close()


    def write_events(self, mass, coupling, energy, filename=None, numberevent=10, zfront=0, nsample=1,
        notime=True, t0=0, modes=None, return_data=False,
        filetype="hepmc", preselectioncuts="th<0.01", weightnames=None):
        """
        A handle to the file writing functions

        Parameters
        ----------
        mass: float
            The particle mass
        coupling: float
            Coupling strength
        energy: str
            Collider sqrt(S) in TeV
        filename: str, None
            The name of the output file to produce. If None, defaults to mass_coupling.suffix
        numberevent: int
            Number of events
        zfront: float
            Advance z-axis position by a constant, default 0
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
        notime: bool
            If false, time information included in position vectors
        t0=0, modes=None
        return_data: bool
            Flag whether to return data and weight information
        filetype: str
            Specify "hepmc" or "csv"
        preselectioncuts: str
            Expression defining cuts to be used e.g. "th<0.01"
        weightnames:
            Labels for the weights, written into hepmc file header

        Returns
        -------
            If return_data: weighted raw data, baseweights, unweighted data
            Else None
        """
        #initialize weightnames if not defined
        model = self.model
        if modes is None: modes = {key: model.production[key]["production"] for key in model.production.keys()}
        nprods = max([len(modes[key]) for key in modes.keys()])
        for key in modes.keys(): modes[key] += [modes[key][0]] * (nprods - len(modes[key]))
        if weightnames is None: weightnames = modes[list(modes.keys())[0]]

        # get weighted sample of LLPs
        _, _, _, weighted_raw_data, weights = self.get_events(mass=mass, energy=energy, couplings = [coupling],
            nsample=nsample, modes=modes, preselectioncuts=preselectioncuts)
        baseweights = weights[0].T[0]

        # unweight sample
        weighted_combined_data = [[p,0 if w[0]==0 else w/w[0]] for p,w in zip(weighted_raw_data, weights[0])]
        unweighted_raw_data = self.rng.choices(weighted_combined_data, weights=baseweights, k=numberevent)
        eventweight = sum(baseweights)/float(numberevent)

        # setup decay channels
        decaymodes = self.model.br_functions.keys()
        branchings = [float(self.model.get_br(mode,mass,coupling)) for mode in decaymodes]
        finalstates = [self.model.br_finalstate[mode] for mode in decaymodes]
        channels = [[[fs, mode], br] for mode, br, fs in zip(decaymodes, branchings, finalstates)]
        br_other = 1-sum(branchings)
        if br_other>0: channels.append([[None,"unspecified"], br_other])
        channels=np.array(channels,dtype='object').T

        # get LLP momenta and decay location
        unweighted_data = []
        for momentum, weight in unweighted_raw_data:
            # determine choice of final state
            while True:
                pids, mode = self.rng.choices(channels[0], weights=channels[1], k=1)[0]
                if (self.channels is None) or (mode in self.channels): break
            # position
            thetax, thetay = 0 if momentum.pz==0 else momentum.px/momentum.pz, 0 if momentum.pz==0 else momentum.py/momentum.pz
            posz = self.rng.uniform(0,self.length)
            posx = thetax*self.distance
            posy = thetay*self.distance
            post = posz + t0
            if notime: position = LorentzVector(posx,posy,posz+zfront,0)
            else     : position = LorentzVector(posx,posy,posz+zfront,post)
            # decay
            pids, finalstate = self.decay_llp(momentum, pids)
            # save
            unweighted_data.append([eventweight*weight, position, momentum, pids, finalstate])

        # prepare output filename
        dirname = self.model.modelpath+"model/events/"
        if not os.path.exists(dirname): os.mkdir(dirname)
        if filename==None: filename = dirname+str(mass)+"_"+str(coupling)+"."+filetype
        else: filename = self.model.modelpath + filename

        # write to file file
        if filetype=="hepmc": self.write_hepmc_file(filename=filename, data=unweighted_data, weightnames=weightnames)
        if filetype=="csv": self.write_csv_file(filename=filename, data=unweighted_data)

        #return
        if return_data: return weighted_raw_data, weights[0], unweighted_data

    ###############################
    #  Plotting and other final processing
    ###############################

    def extract_contours(self,
            inputfile, outputfile,
            nevents=3,
            icontour=0,
        ):
        """
        Export information of contour lines into text files

        Parameters
        ----------
        inputfile: str
            Load data from this file
        outputfile: str
            Filename for result output
        nevents: int
            Number of events
        icontour: int
            Number of Contour

        Returns
        -------
            None
        """
        # load data
        masses,couplings,nsignals=np.load(inputfile, allow_pickle=True, encoding='latin1')
        m, c = np.meshgrid(masses, couplings)
        n = np.log10(np.array(nsignals).T+1e-20)

        # extract line
        cs = plt.contour (m,c,n, levels=[np.log10(nevents)])
        #FIXME ContourSet definition changed in matplotlib ver 3.8
        p = cs.collections[0].get_paths()[icontour]
        v = p.vertices
        xvals, yvals = v[:,0], v[:,1]
        plt.close()

        # save to fole
        f= open(outputfile,"w")
        for x, y in zip(xvals,yvals): f.write(str(x)+" "+str(y)+"\n")
        f.close()

    def plot_reach(self,
            setups, bounds, projections, bounds2=[], grids=[],
            title=None, linewidths=None, xlabel=r"Mass [GeV]", ylabel=r"Coupling",
            xlims=[0.01,1],ylims=[10**-6,10**-3], figsize=(7,5), legendloc=None,
            branchings=None, branchingsother=None,
            fs_label=14, confidence_interval=False,
        ):
        """
        Produce reach plot

        Parameters
        ----------
        setups: [[str,str,str,str,float,int]]
            List of arrays, with each array containing the filename in model/results directory,
            label, color, linestyle, opacity alpha for filled contours and required number of events
        bounds: [[str,str,float,float,float]]
            List of arrays specifying the existing bounds to plot. Drawn in dark gray.
            Each array contains:
            filename in model/bounds directory, label, label x and y positions, label rotation
        projections: [[str,str,str,float,float,float]]
            List of arrays specifying other projections to include in the plot. Each array contains:
            filename in model/bounds directory, color, label, label x and y positiond, label rotation
        bounds2: [[str,str,float,float,float]]
            List of arrays specifying further existing bounds to plot. Drawn in light gray.
            Each array contains:
            filename in model/bounds directory, label, label x and y positions, label rotation
        grids: [[str, 2D ndarray, ndarray, str,str]]
            List of arrays specifying any irregular grids. See scipy.interpolate.griddata
            Each array contains:
            label, 2D grid points, values, color, linestyle
        title: str
            Main title above the plot
        linewidths: float, [float]
            The linewidths for the contours, optionally specified case-by-case in an array.
            See matplotlib.axes.Axes.contour
        xlabel: str
            Horizontal axis label in plot
        ylabel: str
            Vertical axis label in plot
        xlims: [float,float]
            Lower and higher limits on the horizontal axis
        ylims: [float,float]
            Lower and higher limits on the vertical axis

        Returns
        -------
            Pyplot object
        """

        # initiate figure
        matplotlib.rcParams.update({'font.size': 15})

        if branchings is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
            spec = gridspec.GridSpec(nrows=2,ncols=1,width_ratios=[1],height_ratios=[1,0.3],wspace=0,hspace=0)
            ax = fig.add_subplot(spec[0])
        zorder=-100

        # Existing Constraints
        for bound in bounds2:
            filename, label, posx, posy, rotation = bound
            data=np.loadtxt(self.model.modelpath+"model/lines/"+filename)
            ax.fill(data.T[0], data.T[1], color="#efefef",zorder=zorder)
            ax.plot(data.T[0], data.T[1], color="darkgray"  ,zorder=zorder,lw=1)
            zorder+=1

        # Future sensitivities
        for projection in projections:
            filename, color, label, posx, posy, rotation = projection
            data=np.loadtxt(self.model.modelpath+"model/lines/"+filename)
            ax.plot(data.T[0], data.T[1], color=color, ls="dashed", zorder=zorder, lw=1)
            zorder+=1

        # Existing Constraints
        for bound in bounds:
            filename, label, posx, posy, rotation = bound
            data=np.loadtxt(self.model.modelpath+"model/lines/"+filename)
            ax.fill(data.T[0], data.T[1], color="gainsboro",zorder=zorder)
            ax.plot(data.T[0], data.T[1], color="dimgray"  ,zorder=zorder,lw=1)
            zorder+=1

        # labels
        for bound in bounds2:
            filename, label, posx, posy, rotation = bound
            if label is None: continue
            ax.text(posx, posy, label, fontsize=fs_label, color="darkgray", rotation=rotation)
        for projection in projections:
            filename, color, label, posx, posy, rotation = projection
            if label is None: continue
            ax.text(posx, posy, label, fontsize=fs_label, color=color, rotation=rotation)
        for bound in bounds:
            filename, label, posx, posy, rotation = bound
            if label is None: continue
            ax.text(posx, posy, label, fontsize=fs_label, color="dimgray", rotation=rotation)

        # forward experiment sensitivity
        for setup in setups:
            filename, label, color, ls, alpha, level = setup
            if type(level)==list: level_up, level, level_down = level
            else: level_up, level_down = None, None
            masses,couplings,nsignals=np.load(self.model.modelpath+"model/results/"+filename, allow_pickle=True, encoding='latin1')
            m, c = np.meshgrid(masses, couplings)
            n = np.log10(np.array(nsignals).T+1e-20)
            ax.contour (m,c,n, levels=[np.log10(level)]       ,colors=color,zorder=zorder, linestyles=ls, linewidths=linewidths)
            if level_up is not None: ax.contourf(m,c,n, levels=[np.log10(level_up),np.log10(level_down)],colors=color,zorder=zorder, alpha=alpha)
            ax.plot([0,0],[0,0], color=color,zorder=-1000, linestyle=ls, label=label)
            zorder+=1

        # irregular grids
        for label, points, values, color, ls in grids:
            masses = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), 101)
            couplings = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]) ,101)
            m, c = np.meshgrid(masses, couplings)
            v = np.log10(np.array(values)+1e-20)
            n = interpolate.griddata(points, v, (m,c), method='linear')
            ax.contour (m,c,n, levels=[np.log10(level)] ,colors=color, zorder=zorder, linestyles=ls, linewidths=linewidths)
            ax.plot([0,0],[0,0], color=color,zorder=-1000, linestyle=ls, label=label)
            zorder+=1

        #frame
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", bbox_to_anchor=legendloc, frameon=False, labelspacing=0)

        if branchings is not None:
            ax.tick_params(axis="x",direction="in", pad=-15)
            ax.set_xticklabels([])
            ax2 = fig.add_subplot(spec[1])
            for channel, color, ls, label, posx, posy in branchings:
                masses = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),1000)
                brvals = [self.model.get_br(channel, mass, 1) for mass in masses]
                ax2.plot(masses, brvals, color=color, ls=ls)
                ax2.text(posx, posy, label, fontsize=fs_label, color=color)
            if branchingsother is not None:
                color, ls, label, posx, posy, range = branchingsother
                masses = np.logspace(np.log10(range[0]),np.log10(range[1]),1000)
                brvals = [1-sum([self.model.get_br(branching[0], mass, 1) for branching in branchings])for mass in masses]
                ax2.plot(masses, brvals, color=color, ls=ls)
                ax2.text(posx, posy, label, fontsize=fs_label, color=color)
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlim(xlims[0],xlims[1])
            ax2.set_ylim(0.01, 1.5)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("BR")
            return plt, ax, ax2

        return plt

    def plot_production(self,
        masses, productions, condition="True", energy="14",
        xlims=[0.01,1],ylims=[10**-6,10**-3],
        xlabel=r"Mass [GeV]", ylabel=r"\sigma/\epsilon^2$ [pb]",
        figsize=(7,5), fs_label=14, title=None, legendloc=None, dolegend=True, ncol=1, normalization_factor=1,
    ):
        """
        Plot the production modes

        Parameters
        ----------
        masses: [float]
            List of mass values to loop over
        productions: [ dict, ... ]
            List of dictionaries specifying each production mode, e.g.
            {"channels": "111",
             "color": "red",
             "label": r"$\\pi^0 \to \\gamma A'$",
             "generators": ["EPOSLHC"]},
        condition: str
            Add event weight to total if this condition is satisfied
        energy: str
            The collider sqrt(S) in TeV
        xlims: [float,float]
            Lower and higher limits on the horizontal axis
        ylims: [float,float]
            Lower and higher limits on the vertical axis
        xlabel: str
            Horizontal axis label in plot
        ylabel: str
            Vertical axis label in plot
        figsize: (float,float)
            The (horizontal,vertical) dimensions of the figure to produce
        fs_label: float
            Label font size
        title: str, None
            Main plot title
        legendloc: BboxBase, 2-tuple, 4-tuple of floats
            Bbox to anchor legend to
        dolegend: bool
            Flag whether to include legend in plot
        ncol: int
            Number of columns for legend formatting
        Returns
        -------
            Pyplot object
        """
        # initiate figure
        matplotlib.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(figsize=figsize)

        # loop over production channels
        dirname = self.model.modelpath+"model/LLP_spectra/"
        for production in productions:

            # get arguments
            channels = production['channels']
            if 'massrange' in production.keys(): massrange = production['massrange']
            else: massrange=xlims
            if 'color' in production.keys(): color = production['color']
            else: color=None
            if 'ls' in production.keys(): ls = production['ls']
            else: ls=None
            if 'label' in production.keys(): label = production['label']
            else: label=None
            if 'generators' in production.keys(): generators = production['generators']
            else: generators=None

            # fix format
            if isinstance(generators, (list, tuple, np.ndarray))== False: channels=[generators]
            if isinstance(channels, (list, tuple, np.ndarray))== False: channels=[channels]

            # loop over generators
            xvals, yvals = [], [[] for _ in generators]
            for igen, generator in enumerate(generators):
                # loop over masses
                for mass in masses:
                    if mass<massrange[0]: continue
                    if mass>massrange[1]: continue
                    # loop over channels
                    total = 0
                    for channel in channels:

                        filename = dirname+energy+"TeV_"+"m_"+str(mass)+".txt.gz"
                        key_llp  = f"{channel}({generator})"

                        try:
                            data = self.read_list_angle_momenta_weights(filename, keys = [key_llp])

                            for i in range(len(data[0])):
                                logth, logp, w = data[0][i],data[1][i],data[2][i][0]
                                if eval(condition): total+=w
                        except:
                            continue
                    if igen==0: xvals.append(mass)
                    yvals[igen].append(total+1e-10)

            # add to plot
            yvals = np.array(yvals)*float(normalization_factor)
            yvals_min = [min(row) for row in yvals.T]
            yvals_max = [max(row) for row in yvals.T]
            ax.plot(xvals, yvals[0], color=color, label=label, ls=ls)
            ax.fill_between(xvals, yvals_min, yvals_max, color=color, alpha=0.2)

        # finalize
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if dolegend: ax.legend(loc="upper right", bbox_to_anchor=legendloc, frameon=False, labelspacing=0, fontsize=fs_label, ncol=ncol)

        # return
        return plt

    # show 2d hadronspectrum
    def get_spectrumplot(self, pid="111", generator="EPOSLHC", energy="14", prange=[[-6, 0, 60],[ 0, 4, 40]]):
        """
        Plot the spectrum of a given particle type as predicted by a given generator

        Parameters
        ----------
        pid: str
            Plot the spectrum of particles with this PDG ID
        generator: str
            Plot the spectrum corresponding to this prediction
        energy: str
            The collider sqrt(s) in TeV
        prange: [[float, float, float], [float,float,float]]
            Lists of min, max and num for t (prange[0]) and p (prange[1])

        Returns
        -------
            Pyplot object
        """
        filename = self.dirpath + "files/hadrons/"+energy+"TeV.txt.gz"
        keys = [f"{pid}({generator})"]
        p,w = self.read_list_4momenta_weights(filename, keys,mass=self.masses(pid))
        plt,_,_,_ =self.convert_to_hist_list(p,w[:,0], do_plot=True, prange=prange)
        return plt

    def plot_production_branchings(self,
        masses, productions,
        xlims=[0.01,1],ylims=[10**-1,1],
        xlabel=r"Mass [GeV]", ylabel=r"BR/g^2$",
        figsize=(7,5), fs_label=14, title=None, legendloc=None, dolegend=True, ncol=1, xlog=True, ylog=True,
        nsample=100):
        """
        Plot branching fractions for given production modes

        Parameters
        ----------
        masses: [float]
            List of mass values to loop over
        productions: [[str,str,str]]
            List of lists specifying the production modes. Each entry contains:
            key in productions dict, color, description/label
        xlims: [float,float]
            Lower and higher limits on the horizontal axis
        ylims: [float,float]
            Lower and higher limits on the vertical axis
        xlabel: str
            Horizontal axis label in plot
        ylabel: str
            Vertical axis label in plot
        figsize: (float,float)
            The (horizontal,vertical) dimensions of the figure to produce
        fs_label: float
            Label font size
        title=None
        legendloc: BboxBase, 2-tuple, or 4-tuple of floats
            Bbox to anchor legend to
        dolegend: bool
            Flag whether to include legend in plot
        ncol: int
            Number of columns for legend formatting
        xlog: bool
            Flag whether to use logarithmic horizontal axis
        ylog: bool
            Flag whether to use logarithmic vertical axis
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by

        Returns
        -------
            Pyplot object
        """
        # initiate figure
        matplotlib.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(figsize=figsize)

        # loop over production channels
        model = self.model
        coupling = 1
        for key, color, label in productions:

            # load details of production channel
            pid0 = model.production[key]["pid0"]
            pid1 = model.production[key]["pid1"]
            pid2 = model.production[key]["pid2"]
            br = model.production[key]["br"]
            nsample = model.production[key]["nsample"]
            massrange = model.production[key]["massrange"]
            integration = model.production[key]["integration"]

            # loop over masses
            xvals, yvals = [], []
            for mass in masses:
                xvals.append(mass)
                if model.production[key]["type"]=="2body":
                    if (self.masses(pid0)<=self.masses(pid1,mass)+mass): yvals.append(0)
                    else: yvals.append(eval(br))
                elif model.production[key]["type"]=="3body":
                    if (self.masses(pid0)<=self.masses(pid1,mass)+self.masses(pid2,mass)+mass): yvals.append(0)
                    else:
                        m0, m1, m2, m3 = self.masses(pid0), self.masses(pid1,mass), self.masses(pid2,mass), mass
                        _, weights = self.decay_in_restframe_3body(br, 1, m0, m1, m2, m3, nsample=nsample, integration=integration)
                        yvals.append(sum(weights))

            # add to plot
            ax.plot(xvals, yvals, color=color, label=label)

        # finalize
        ax.set_title(title)
        if xlog: ax.set_xscale("log")
        if ylog: ax.set_yscale("log")
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if dolegend: ax.legend(loc="upper right", bbox_to_anchor=legendloc, frameon=False,
            labelspacing=0, fontsize=fs_label, ncol=ncol)

        # return
        return plt
