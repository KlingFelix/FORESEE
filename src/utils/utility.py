from particle import Particle
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math, gzip
from .vectors import *

class Utility():

    def __init__(self, rng=None):
        self.rng = rng

    ###############################
    #  Hadron Masses, lifetimes etc
    ###############################

    def charges(self, pid):
        """
        Retrieve particle charges from scikit-particle API

        Parameters
        ----------
        pid:  int / str
            The PDG ID for which to request charge

        Returns
        -------
        Particle charge as float
        """
        try:
            charge = Particle.from_pdgid(int(pid)).charge
        except:
            charge = 0.0
        return charge if charge!=None else 0.0

    def masses(self,pid,mass=0):
        """
        Retrieve particle masses from scikit-particle API

        Parameters
        ----------
        pid:  int / str
            The PDG ID for which to request mass
        mass: float
            Default value returned if pid==0

        Returns
        -------
        Particle mass as float
        """
        pidabs = abs(int(pid))
        #Treat select entries separately
        if   pidabs==0: return mass
        elif pidabs==4: return 1.5   #GeV, scikit-particle returns 1.27 for c quark
        elif pidabs==5: return 4.5   #GeV, scikit-particle returns 4.18 for b quark
        #General case: fetch values from scikit-particle
        else:
            mret = Particle.from_pdgid(pidabs).mass   #MeV
            return mret*0.001 if mret!=None else 0.0  #GeV

    def ctau(self,pid):
        """
        Retrieve particle lifetimes tau multiplied by the speed of light c
        from scikit-particle API

        Parameters
        ----------
        pid:  int / str
            The PDG ID for which to request c*tau

        Returns
        -------
        Particle c*tau as float
        """
        pidabs = abs(int(pid))
        ctau = 0.0
        try:
            ctau = Particle.from_pdgid(pidabs).ctau
        except:
            ctau = 0.0
            print('WARNING '+str(pid)+' ctau not obtained from scikit-particle')
        if ctau==None: ctau=0.0
        if np.isinf(ctau): ctau=8.51472e+48  #Avoid inf return value in code
        return ctau*0.001

    def widths(self, pid):
        """
        Retrieve particle widths from scikit-particle API

        Parameters
        ----------
        pid:  int / str
            The PDG ID for which to request width

        Returns
        -------
        Particle width as float
        """
        try:
            width = Particle.from_pdgid(int(pid)).width
        except:
            width = 0.0
            print('WARNING '+str(pid)+' width not obtained from scikit-particle, returning 0')
        return width*1e-3 if width!=None else 0.0
        
    ###############################
    #  Import Function
    ###############################

    def readfile(self,filename):
        """
        Function that reads a table in a .txt file and converts it to a numpy array

        Parameters
        ----------
        filename:  str
            The name/path of the file to be read

        Returns
        -------
        The recovered table as a numpy array. Do be de-preciated, kept for historial reasons.
        """
        print ("Warning: Foresee.readfile(filename) will be depreciated soon. Replace it with np.loadtxt(filename).")
        array = []
        with open(filename) as f:
            for line in f:
                if line[0]=="#":continue
                words = [float(elt.strip()) for elt in line.split( )]
                array.append(words)
        return np.array(array)

    ###############################
    #  Reading/Plotting Particle Tables
    ###############################

    def read_list_angle_momenta_weights(self,filename, keys):
        """
        Read a flattened grid of (logth, logp) points and associated weights from a
        (optionally gzipped) text file written by write_list_angle_momenta_weights.
    
        Parameters
        ----------
        filename : str
            Path to the file to read.  If it ends with '.gz' it is treated as
            gzip-compressed.
        keys : [str]
            Column labels to extract (e.g. '111(EPOSLHC)' or 'Brem_FWW(p.pt<1)').
            Must be a subset of the columns present in the file.
    
        Returns
        -------
        list_th : [float]
            The log-theta grid values.
        list_p : [float]
            The log-momentum grid values.
        list_w : [[float]]
            Weights for every requested key at every grid point.
            list_w[i] corresponds to keys[i].
    
        Raises
        ------
        KeyError
            If any of the requested keys are not found in the file header.
        """
        open_func = gzip.open if filename.endswith(".gz") else open
    
        with open_func(filename, "rb") as f:
            # --- parse header ---
            raw_header = f.readline().decode().strip()
            # Strip quotes that were added around key names during writing
            header_cols = [col.strip('"') for col in raw_header.split()]
    
            idx_th = header_cols.index("logth")
            idx_p  = header_cols.index("logp")
    
            missing = [k for k in keys if k not in header_cols]
            if missing:
                raise KeyError(f"Requested key(s) not found in file header: {missing}")
    
            idx_keys = [header_cols.index(k) for k in keys]
    
            # --- read data rows ---
            list_th, list_p, list_w = [], [], [[] for _ in keys]
    
            for line in f:
                line = line.decode().strip()
                if not line:
                    continue
                parts = line.split()
                list_th.append(float(parts[idx_th]))
                list_p.append(float(parts[idx_p]))
                
                for out_idx, col_idx in enumerate(idx_keys):
                    w = parts[col_idx]
                    if w != 'NULL': list_w[out_idx].append(float(w))
                    else: list_w[out_idx].append(np.nan)
                        
        mask = ~np.isnan(np.array(list_w)).any(axis=0)
        list_th = [list_th[i] for i in range(len(list_th)) if mask[i]]
        list_p = [list_p[i] for i in range(len(list_p)) if mask[i]]
        list_w = np.array(list_w)[:, mask]
    
        return list_th, list_p, np.array(list_w).T

    def read_list_4momenta_weights(self,filename, keys, mass,nsample=1,preselectioncut=None, nocuts=False):
        """
        Function that converts input files under files/hadrons/ into meson spectra

        Parameters
        ----------
        filename : str
            Path to the file to read.  If it ends with '.gz' it is treated as
            gzip-compressed.
        keys : [str]
            Column labels to extract (e.g. '111(EPOSLHC)' or 'Brem_FWW(p.pt<1)').
            Must be a subset of the columns present in the file.
        mass: float
            The mass of the considered particle
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by.
            Each entry in the filename(s) then results in nsample particles, so the total number
            of particles returned in the end will be [the amount in list] x nsample
        preselectioncuts: str / None
            Expression defining cuts to be used e.g. "th<0.01 and p>100"
        nocuts: bool
            Flag whether to skip applying cuts

        Returns
        -------
            Particles as a list of LorentzVectors (old skhep) / skheparray (new), 
            and weights as an np.array of np.arrays. The weight subarray index 
            corresponds to alternative cross sections / weights per particle
        """
        #read file
        try:
            list_logth, list_logp, list_xs = self.read_list_angle_momenta_weights(filename=filename, keys=keys)
        except:
            if filename.endswith(".gz"): filename = filename[:-3]  #Check for uncompressed filename variant
            list_logth, list_logp, list_xs = self.read_list_angle_momenta_weights(filename=filename, keys=keys)
            
        phis,ths,pts,ens,weights = [],[],[],[],[]
        for logth,logp,xs in zip(list_logth,list_logp, list_xs):
            
            if nocuts==False and max(xs) < 10.**-6: continue
            p  = 10.**logp
            th = 10.**logth
            
            if nocuts==False and preselectioncut is not None:
                if not eval(preselectioncut): continue

            #Sample random variables
            phis.append(np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample))))
            fthrand = np.array(list(map(self.rng.uniform,[-0.025]*nsample,[0.025]*nsample)))
            fprand  = np.array(list(map(self.rng.uniform,[-0.025]*nsample,[0.025]*nsample)))
            fth = np.power(10,fthrand)
            fp  = np.power(10,fprand )
            
            #Angles, 3-momentum magnitudes and transverse momenta for constructing 4-momenta
            th_smeared = np.multiply(th,fth)
            p_smeared  = np.multiply(p, fp )
            ths.append(th_smeared)
            pts.append(np.multiply(p_smeared, np.sin(th_smeared)))
            ens.append(np.sqrt(np.add(p_smeared**2,mass**2)))
            
            weights.append( np.ones((nsample,1)) * np.array(xs)/float(nsample) )
                
        #Flatten
        phis = np.concatenate(phis)
        ths  = np.concatenate(ths)
        ens  = np.concatenate(ens)
        pts  = np.concatenate(pts)

        #Construct particle 4-momentum list/array
        particles = LorentzArray({"pt": pts, "theta": ths, "phi": phis, "energy": ens})

        return particles, np.concatenate(weights)

    def write_list_angle_momenta_weights(self, list_th, list_p, list_w, keys, filename="output.txt.gz"):
        """
        Write a flattened grid of (logth, logp) points and associated weights to a
        gzipped text file.
    
        Parameters
        ----------
        list_th : array-like of float
            The log-theta grid values.
        list_p : array-like of float
            The log-momentum grid values.
        list_w : list of list of float
            Weights for every key at every grid point.
        keys : list of str
            Column labels (e.g. '111(EPOSLHC)' or 'Brem_FWW(p.pt<1)').
        filename : str
            Output path.  If it ends with '.gz' the file is gzip-compressed.
        """
        n_points = len(list_th)
    
       
        # Build the header line
        header_parts = ["logth", "logp"] + [f'{k}' for k in keys]
        header = " ".join(header_parts)
    
        open_func = gzip.open if filename.endswith(".gz") else open
    
        with open_func(filename, "wb") as f:
            f.write((header + "\r\n").encode())
    
            for i in range(n_points):
                row_parts = [str(list_th[i]), str(list_p[i])]
                for k in range(len(keys)):
                    row_parts.append(str(list_w[k][i]))
                f.write((" ".join(row_parts) + "\r\n").encode())
        print(f"spectra saved:\t{filename}")



    
    def convert_list_to_momenta(self,filename, keys,mass,filetype="txt",nsample=1,preselectioncut=None, nocuts=False):
        """
        Old name of function "read_list_4momenta_weights".
        Please replace by "read_list_4momenta_weights".
        Will be depreciated soon.
        """
        ## TODO: remove function when its safe to do so
        print ("Warning: Foresee.convert_list_to_momenta() will be depreciated soon. Replace it with Foresee.read_list_4momenta_weights().")
        return self.read_list_4momenta_weights(filename, keys, mass,nsample,preselectioncut,nocuts)


    def get_hist_list(self, tx, px, weights, prange):
        """
        Fetch the contents of a 2D histo given in terms of angles and momenta in list format

        Parameters
        ----------
        tx: numpy array of floats
            Values for the angle w.r.t. z-axis, for producing the 2D grid
        px: numpy array of floats
            Momentum values for producing the 2D grid
        weights: numpy array of floats
            Weights for each entry in the histo
        prange: [[float,float,float],[float,float,float]]
            Lists of min, max and num for t (prange[0]) and p (prange[1])

        Returns
        -------
            Lists of angles w.r.t z-axis, momenta and weights
        """
        
        # define histogram
        tmin, tmax, tnum = prange[0]
        pmin, pmax, pnum = prange[1]
        dt = (tmax - tmin) / tnum
        dp = (pmax - pmin) / pnum
        t_edges = np.logspace(tmin, tmax, num=tnum + 1)
        p_edges = np.logspace(pmin, pmax, num=pnum + 1)
        log_t_centers = np.linspace(tmin + 0.5 * dt, tmax - 0.5 * dt, num=tnum)
        log_p_centers = np.linspace(pmin + 0.5 * dp, pmax - 0.5 * dp, num=pnum)

        # fill histogram
        w, _, _ = np.histogram2d(tx, px, weights=weights, bins=(t_edges, p_edges))

        # build grid of centers
        T, P = np.meshgrid(log_t_centers, log_p_centers, indexing="ij")

        # convert to desired output
        list_t = T.ravel().tolist()
        list_p = P.ravel().tolist()
        list_w = w.ravel().tolist()

        return list_t, list_p, list_w

    def convert_to_hist_list(self,momenta,weights, do_plot=False, filename=None, prange=[[-5, 0, 100],[ 0, 4, 80]], vmin=None, vmax=None):
        """
        Convert list of momenta to 2D histogram, and plot

        Parameters
        ----------
        momenta: [LorentzVector] / skheparray (new skhep) / ndarray of length 4 or 2
            List of 4-momenta
        weights: numpy array of floats
            Weights for each entry in the histo
        do_plot: bool
            Flag whether to produce a spectrum plot based on the resulting lists or not
        filename: str / None
            Output filename for saving results
        prange: [[float,float,float],[float,float,float]]
            Lists of min, max and num for t (prange[0]) and p (prange[1])
        vmin: float
            Value mapped to 0 for the color map. See matplotlib.colors.LogNorm
        vmax: float
            Value mapped to 1 for the color map. See matplotlib.colors.LogNorm

        Returns
        -------
            If do_plot, return pyplot object first, then lists of angles w.r.t z-axis, momenta
            and weights. If do_plot false, only return the lists.
        """

        #preprocess data
        if type(momenta[0])==LorentzVector:
            tx = np.array([np.arctan(mom.pt/mom.pz) for mom in momenta])
            px = np.array([mom.p for mom in momenta])
        elif type(momenta) == np.ndarray and len(momenta[0]) == 4:
            tx = np.array([math.pi/2 if zp==0 else np.arctan(np.sqrt(xp**2+yp**2)/zp) for xp,yp,zp,_ in momenta])
            px = np.array([np.sqrt(xp**2+yp**2+zp**2) for xp,yp,zp,_ in momenta])
        elif type(momenta) == np.ndarray and len(momenta[0]) == 2:
            tx, px = momenta.T
        else:
            try:
                #Covers new skhep skheparray case
                tx = momenta.theta
                px = momenta.p
            except:
                tx,px = np.array([]), np.array([])
                print ("Error: momenta provided in unknown format: "+str(type(momenta)))

        # get standard weighted list
        list_t, list_p, list_w = self.get_hist_list(tx, px, weights, prange=prange )

        # save file ?
        if filename is not None:
            print ("save data to file:", filename)
            np.save(filename,[list_t,list_p,list_w])

        # plot ?
        if do_plot:
            plotobj=self.make_spectrumplot(list_t, list_p, list_w, prange, vmin=vmin, vmax=vmax)
            return plotobj, list_t,list_p,list_w
        else:
            return list_t,list_p,list_w

    def make_spectrumplot(self, list_t, list_p, list_w, prange=[[-5, 0, 100],[ 0, 4, 80]], vmin=None, vmax=None):
        """
        A colormap spectrum in terms of z-axis angles and momenta

        Parameters
        ----------
        list_t: [float]
            List of angles w.r.t z-axis
        list_p: [float]
            List of momenta
        list_w: [float]
            List of weights
        prange: [[float,float,float],[float,float,float]]
            Lists of min, max and num for t (prange[0]) and p (prange[1])
        vmin: float
            Value mapped to 0 for the color map. See matplotlib.colors.LogNorm
        vmax: float
            Value mapped to 1 for the color map. See matplotlib.colors.LogNorm

        Returns
        -------
            Pyplot object
        """
        matplotlib.rcParams.update({'font.size': 15})
        fig = plt.figure(figsize=(7,5.5))

        #get plot
        tmin, tmax, tnum = prange[0]
        pmin, pmax, pnum = prange[1]
        ticks = np.array([[np.linspace(10**(j),10**(j+1),9)] for j in range(-7,6)]).flatten()
        ticks = [np.log10(x) for x in ticks]
        ticklabels = np.array([[r"$10^{"+str(j)+"}$","","","","","","","",""] for j in range(-7,6)]).flatten()

        ax = plt.subplot(1,1,1)
        h=ax.hist2d(x=list_t,y=list_p,weights=list_w,
                    bins=[tnum,pnum],range=[[tmin,tmax],[pmin,pmax]],
                    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap="rainbow",
        )
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(r"angle wrt. beam axis $\theta$ [rad]")
        ax.set_ylabel(r"momentum $p$ [GeV]")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(pmin, pmax)
        return plt
