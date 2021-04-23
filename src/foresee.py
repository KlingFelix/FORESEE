import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
import random
from skhep.math.vectors import LorentzVector, Vector3D
from scipy import interpolate

class Utility():

    ###############################
    #  Hadron Masses and Lifetimes
    ###############################
    
    def masses(self,pid,mass=0):
        if   pid in ["2112","-2112"]: return 0.938
        elif pid in ["2212","-2212"]: return 0.938
        elif pid in ["211" ,"-211" ]: return 0.13957
        elif pid in ["321" ,"-321" ]: return 0.49368
        elif pid in ["310" ,"130"  ]: return 0.49761
        elif pid in ["111"         ]: return 0.135
        elif pid in ["221"         ]: return 0.547
        elif pid in ["331"         ]: return 0.957
        elif pid in ["3122","-3122"]: return 1.11568
        elif pid in ["3222","-3222"]: return 1.18937
        elif pid in ["3112","-3112"]: return 1.19745
        elif pid in ["3322","-3322"]: return 1.31486
        elif pid in ["3312","-3312"]: return 1.32171
        elif pid in ["3334","-3334"]: return 1.67245
        elif pid in ["113"         ]: return 0.77545
        elif pid in ["223"         ]: return 0.77524
        elif pid in ["333"         ]: return 1.019461
        elif pid in ["411" ,"-411" ]: return 1.86961
        elif pid in ["421" ,"-421" ]: return 1.86484
        elif pid in ["431" ,"-431" ]: return 1.96830
        elif pid in ["4122","-4122"]: return 2.28646
        elif pid in ["511" ,"-511" ]: return 5.27961
        elif pid in ["521" ,"-521" ]: return 5.27929
        elif pid in ["531" ,"-531" ]: return 5.36679
        elif pid in ["541" ,"-541" ]: return 6.2749
        elif pid in ["5"   ,"-5"   ]: return 4.5
        elif pid in ["22"          ]: return 0
        elif pid in ["23"          ]: return 91.
        elif pid in ["24"  ,"-24"  ]: return 80.4
        elif pid in ["25"          ]: return 125.
        elif pid in ["553"         ]: return 9.46
        elif pid in ["0"           ]: return mass
    
    def ctau(self,pid):
        if   pid in ["2112","-2112"]: tau = 10**8
        elif pid in ["2212","-2212"]: tau = 10**8
        elif pid in ["211","-211"  ]: tau = 2.603*10**-8
        elif pid in ["321","-321"  ]: tau = 1.238*10**-8
        elif pid in ["310"         ]: tau = 8.954*10**-11
        elif pid in ["130"         ]: tau = 5.116*10**-8
        elif pid in ["3122","-3122"]: tau = 2.60*10**-10
        elif pid in ["3222","-3222"]: tau = 8.018*10**-11
        elif pid in ["3112","-3112"]: tau = 1.479*10**-10
        elif pid in ["3322","-3322"]: tau = 2.90*10**-10
        elif pid in ["3312","-3312"]: tau = 1.639*10**-10
        elif pid in ["3334","-3334"]: tau = 8.21*10**-11
        return 3*10**8 * tau
    
    ###############################
    #  Utility Functions
    ###############################
    
    #function that reads a table in a .txt file and converts it to a numpy array
    def readfile(self,filename):
        array = []
        with open(filename) as f:
            for line in f:
                if line[0]=="#":continue
                words = [float(elt.strip()) for elt in line.split( )]
                array.append(words)
        return np.array(array)

class Model(Utility):
    
    def __init__(self,name):
        self.model_name = name
        self.lifetime_coupling_ref = None
        self.lifetime_function = None
        self.br_mode=None
        self.br_functions = {}
        self.production = {}
    
    ###############################
    #  Lifetime
    ###############################
    
    def set_ctau_1d(self,filename, coupling_ref=1):
        data=self.readfile(filename).T
        self.ctau_coupling_ref=coupling_ref
        self.ctau_function=interpolate.interp1d(data[0], data[1])
    
    def set_ctau_2d(self,filename):
        data=self.readfile(filename).T
        self.ctau_coupling_ref=None
        self.ctau_function=interpolate.interp2d(data[0], data[1], data[2], kind="linear")
    
    def get_ctau(self,mass,coupling):
        if self.ctau_function==None:
            print "No lifetime specified. You need to specify lifetime first!"
            return 10**10
        elif self.ctau_coupling_ref is None:
            return self.ctau_function(mass,coupling)
        else:
            return self.ctau_function(mass) / coupling**2 *self.ctau_coupling_ref**2

    ###############################
    #  BR
    ###############################

    def set_br_1d(self,modes,filenames):
        self.br_mode="1D"
        self.br_functions = {}
        for mode, filename in zip(modes, filenames):
            data = self.readfile(filename).T
            function = interpolate.interp1d(data[0], data[1])
            self.br_functions[mode] = function
                    
    def set_br_2d(self,modes,filenames):
        self.br_mode="2D"
        self.br_functions = {}
        for mode, filename in zip(modes, filenames):
            data = self.readfile(filename).T
            function = interpolate.interp2d(data[0], data[1], data[2], kind="linear")
            self.br_functions[channel] = function

    def get_br(self,mode,mass,coupling=1):
        if self.br_mode==None:
            print "No branching fractions specified. You need to specify branching fractions first!"
            return 0
        elif mode not in self.br_functions.keys():
            print "No branching fractions into ", mode, " specified. You need to specify BRs for this channel!"
            return 0
        elif self.br_mode == "1D":
            return self.br_functions[mode](mass)
        elif self.br_mode == "2D":
            return self.br_functions[mode](mass, coupling)


    ###############################
    #  Production
    ###############################

    def add_production_2bodydecay(self, pid0, pid1, br, generator, energy, nsample=1, label=None):
        if label is None: label=pid0
        self.production[label]=["2body", pid0, pid1, br, generator, energy, nsample]
        
    def add_production_3bodydecay(self, pid0, pid1, pid2, br, generator, energy, nsample=1, label=None):
        if label is None: label=pid0
        self.production[label]=["3body", pid0, pid1, pid2, br, generator, energy, nsample]
        
    def add_production_mixing(self, pid, mixing, generator, energy, label=None):
        if label is None: label=pid
        self.production[label]=["mixing", pid, mixing, generator, energy]
        
    def add_production_direct(self, label, energy, coupling_ref=1, condition=None, massrange=None):
        self.production[label]=["direct", energy, coupling_ref, condition, massrange]


class Foresee(Utility):

    def __init__(self):
        self.model = None
        self.shortlived = {"321": 20, "-321": 20, "321": 20,  }
        self.selection="np.sqrt(x.x**2 + x.y**2)< 1"
        self.length=5
        self.luminosity=3000
        self.distance=480
        self.channels=None
    
    ###############################
    #  Reading/Plotting Particle Tables
    ###############################

    # convert a table into input for contour plot
    def table2contourinput(self,data,idz=2):
        ntotal=len(data)
        ny=sum( 1 if d[0]==data[0][0] else 0 for d in data)
        nx=sum( 1 if d[1]==data[0][1] else 0 for d in data)
        xval = [data[ix*ny,0] for ix in range(nx)]
        yval = [data[iy,1] for iy in range(ny)]
        zval = [ [ data[ix*ny+iy,idz] for iy in range(ny) ] for ix in range(nx)]
        return np.array(xval),np.array(yval),np.array(zval)

    # function that converts input file into meson spectrum
    def convert_list_to_momenta(self,filename,mass,filetype="txt",nsample=1,preselectioncut=None,):
        if filetype=="txt":
            list_logth, list_logp, list_xs = self.readfile(filename).T
        elif filetype=="npy":
            list_logth, list_logp, list_xs = np.load(filename)
        else:
            print ("ERROR: cannot rtead file type")
        particles=[]
        weights  =[]
        
        for logth,logp,xs in zip(list_logth,list_logp, list_xs):
            
            if xs < 10.**-6: continue
            p  = 10.**logp
            th = 10.**logth
            
            if preselectioncut is not None:
                if not eval(preselectioncut): continue

            for n in range(nsample):
                phi= random.uniform(-math.pi,math.pi)
                if nsample == 1:
                    fth, fp = 1,1
                else:
                    fth = np.random.normal(1, 0.05, 1)[0]
                    fp  = np.random.normal(1, 0.05, 1)[0]
                
                th_sm=th*fth
                p_sm=p*fp
                
                en = math.sqrt(p_sm**2+mass**2)
                pz = p_sm*np.cos(th_sm)
                pt = p_sm*np.sin(th_sm)
                px = pt*np.cos(phi)
                py = pt*np.sin(phi)
                part=LorentzVector(px,py,pz,en)
                
                particles.append(part)
                weights.append(xs/float(nsample))

        return particles,weights

    # convert list of momenta to 2D histogram, and plot
    def convert_to_hist_list(self,momenta,weights, do_plot=False, filename=None, do_return=False, prange=[[-6, 0, 120],[ 0, 5, 50]]):
        
        #get data
        tmin, tmax, tnum = prange[0]
        pmin, pmax, pnum = prange[1]
        t_edges = np.logspace(tmin, tmax, num=tnum+1)
        p_edges = np.logspace(pmin, pmax, num=pnum+1)
        
        tx = [np.arctan(mom.pt/mom.pz) for mom in momenta]
        px = [mom.p for mom in momenta]
        
        w, t_edges, p_edges = np.histogram2d(tx, px, weights=weights,  bins=(t_edges, p_edges))
        
        t_centers = (t_edges[:-1] + t_edges[1:]) / 2
        p_centers = (p_edges[:-1] + p_edges[1:]) / 2
        
        list_t = []
        list_p = []
        list_w = []
        
        for it,t in enumerate(t_centers):
            for ip,p in enumerate(p_centers):
                list_t.append(np.log10 ( t_centers[it] ) )
                list_p.append(np.log10 ( p_centers[ip] ) )
                list_w.append(w[it][ip])
    
        if filename is not None:
            print "save data to file:", filename
            np.save(filename,[list_t,list_p,list_w])
        if do_plot==False:
            return list_t,list_p,list_w

        #get plot
        ticks = np.array([[np.linspace(10**(j),10**(j+1),9)] for j in range(-7,6)]).flatten()
        ticks = [np.log10(x) for x in ticks]
        ticklabels = np.array([[r"$10^{"+str(j)+"}$","","","","","","","",""] for j in range(-7,6)]).flatten()
        matplotlib.rcParams.update({'font.size': 15})
        fig = plt.figure(figsize=(8,5.5))
        ax = plt.subplot(1,1,1)
        h=ax.hist2d(x=list_t,y=list_p,weights=list_w,
                    bins=[tnum,pnum],range=[[tmin,tmax],[pmin,pmax]],
                    norm=matplotlib.colors.LogNorm(), cmap="hsv",
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
        
        return plt, list_t,list_p,list_w

    # show 2d hadronspectrum
    def get_spectrumplot(self, pid="111", generator="EPOSLHC", energy="14", prange=[[-6, 0, 120],[ 0, 5, 50]]):
        dirname = "files/hadrons/"+energy+"TeV/"+generator+"/"
        filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
        p,w = self.convert_list_to_momenta(filename,mass=self.masses(pid))
        plt,_,_,_ =self.convert_to_hist_list(p,w, do_plot=True, prange=prange)
        return plt

    ###############################
    #  Model
    ###############################

    def set_model(self,model):
        self.model = model

    ###############################
    #  LLP production
    ###############################
    
    def get_decay_prob(self, pid, momentum):
    
        # return 1 when decaying promptly
        if pid not in ["211","-211","321","-321","310","130"]: return 1
    
        # lifetime and kinematics
        ctau = self.ctau(pid)
        theta=math.atan(momentum.pt/momentum.pz)
        dbarz = ctau * momentum.pz / momentum.m
        dbart = ctau * momentum.pt / momentum.m
        
        # probability to decay in beampipe
        if pid in ["130", "310"]:
            ltan, ltas, rpipe = 140., 20., 0.05
            if (theta < 0.017/ltas): probability = 1.- np.exp(- ltan/dbarz)
            elif (theta < 0.05/ltas): probability = 1.- np.exp(- ltas/dbarz)
            else: probability = 1.- np.exp(- rpipe /dbart)
        if pid in ["321","-321","211","-211"]:
            ltas, rpipe = 20., 0.05
            if (theta < 0.05/ltas): probability = 1.- np.exp(- ltas/dbarz)
            else: probability = 1.- np.exp(- rpipe /dbart)
        return probability
    
    def twobody_decay(self, p0, m0, m1, m2, phi, costheta):
        """
        function that decays p0 > p1 p2 and returns p1,p2
        """
        
        #get axis of p0
        zaxis=Vector3D(0,0,1)
        rotaxis=zaxis.cross(p0.vector).unit()
        rotangle=zaxis.angle(p0.vector)
        
        #energy and momentum of p2 in the rest frame of p0
        energy1   = (m0*m0+m1*m1-m2*m2)/(2.*m0)
        energy2   = (m0*m0-m1*m1+m2*m2)/(2.*m0)
        momentum1 = math.sqrt(energy1*energy1-m1*m1)
        momentum2 = math.sqrt(energy2*energy2-m2*m2)
        
        #4-momentum of p1 and p2 in the rest frame of p0
        en1 = energy1
        pz1 = momentum1 * costheta
        py1 = momentum1 * math.sqrt(1.-costheta*costheta) * np.sin(phi)
        px1 = momentum1 * math.sqrt(1.-costheta*costheta) * np.cos(phi)
        p1=LorentzVector(-px1,-py1,-pz1,en1)
        if rotangle!=0: p1=p1.rotate(rotangle,rotaxis)
        
        en2 = energy2
        pz2 = momentum2 * costheta
        py2 = momentum2 * math.sqrt(1.-costheta*costheta) * np.sin(phi)
        px2 = momentum2 * math.sqrt(1.-costheta*costheta) * np.cos(phi)
        p2=LorentzVector(px2,py2,pz2,en2)
        if rotangle!=0: p2=p2.rotate(rotangle,rotaxis)
        
        #boost p2 in p0 restframe
        p1_=p1.boost(-1.*p0.boostvector)
        p2_=p2.boost(-1.*p0.boostvector)
        return p1_,p2_
    
    def decay_in_restframe_2body(self, br, m0, m1, m2, nsample):
        
        # prepare output
        particles, weights = [], []
        
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)
        
        #MC sampling of angles
        for i in range(nsample):
            cos =random.uniform(-1.,1.)
            phi =random.uniform(-math.pi,math.pi)
            p_1,p_2=self.twobody_decay(p_mother,m0,m1,m2,phi,cos)
            particles.append(p_2)
            weights.append(br/nsample)
        
        return particles,weights

    def decay_in_restframe_3body(self, br, m0, m1, m2, m3, nsample):
        
        # prepare output
        particles, weights = [], []
        
        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)
        
        #integration boundary
        q2min,q2max = (m2+m3)**2,(m0-m1)**2
        cthmin,cthmax = -1 , 1
        mass = m2
        
        #numerical integration
        integral=0
        for i in range(nsample):
            
            #Get kinematic Variables
            q2 = random.uniform(q2min,q2max)
            cth = random.uniform(-1,1)
            th = np.arccos(cth)
            q  = math.sqrt(q2)
            
            #decay meson and V
            cosQ =cth
            phiQ =random.uniform(-math.pi,math.pi)
            cosM =random.uniform(-1.,1.)
            phiM =random.uniform(-math.pi,math.pi)
            p_1,p_q=self.twobody_decay(p_mother,m0 ,m1,q  ,phiM,cosM)
            p_2,p_3=self.twobody_decay(p_q     ,q  ,m2,m3 ,phiQ,cosQ)
            
            #branching fraction
            brval  = eval(br)
            brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)
            
            #save
            particles.append(p_3)
            weights.append(brval)
        
        return particles,weights

    def get_llp_spectrum(self, mass, coupling, channels=None, do_plot=False, filenamesave=None, print_stats=False):

        # prepare output
        model = self.model
        if channels is None: channels = [key for key in model.production.keys()]
        momenta_lab, weights_lab = [LorentzVector(0,0,-mass,mass)], [0]

        # loop over channels
        for key in model.production.keys():
            
            # selected channels only
            if key not in channels: continue
            
            # summary statistics
            weight_sum, weight_sum_f=0,0
            
            # 2 body decays
            if model.production[key][0]=="2body":
                
                # load details of decay channel
                pid0, pid1, br =  model.production[key][1], model.production[key][2], model.production[key][3]
                generator, energy, nsample = model.production[key][4], model.production[key][5], model.production[key][6]
                if self.masses(pid0) <= self.masses(pid1, mass) + mass: continue
                
                # load mother particle spectrum
                filename = "files/hadrons/"+energy+"TeV/"+generator+"/"+generator+"_"+energy+"TeV_"+pid0+".txt"
                momenta_mother, weights_mother = self.convert_list_to_momenta(filename,mass=self.masses(pid0))
                
                # get sample of LLP momenta in the mother's rest frame
                m0, m1, m2 = self.masses(pid0), self.masses(pid1,mass), mass
                momenta_llp, weights_llp = self.decay_in_restframe_2body(eval(br), m0, m1, m2, nsample)
                
                # loop through all mother particles, and decay them
                for p_mother, w_mother in zip(momenta_mother, weights_mother):
                    # if mother is shortlived, add factor that requires them to decay before absorption
                    w_decay = self.get_decay_prob(pid0, p_mother)
                    for p_llp,w_lpp in zip(momenta_llp, weights_llp):
                        p_llp_lab=p_llp.boost(-1.*p_mother.boostvector)
                        momenta_lab.append(p_llp_lab)
                        weights_lab.append(w_mother*w_lpp*w_decay)
                        # statistics
                        weight_sum+=w_mother*w_lpp*w_decay
                        if p_llp_lab.pz>100 and p_llp_lab.pt/np.abs(p_llp_lab.pz)<0.1/480.: weight_sum_f+=w_mother*w_lpp*w_decay
        
            # 3 body decays
            if model.production[key][0]=="3body":
                
                # load details of decay channel
                pid0, pid1, pid2, br = model.production[key][1], model.production[key][2], model.production[key][3], model.production[key][4]
                generator, energy, nsample = model.production[key][5], model.production[key][6], model.production[key][7]
                if self.masses(pid0) <= self.masses(pid1, mass) + self.masses(pid2, mass) + mass: continue
                   
                # load mother particle
                filename = "files/hadrons/"+energy+"TeV/"+generator+"/"+generator+"_"+energy+"TeV_"+pid0+".txt"
                momenta_mother, weights_mother = self.convert_list_to_momenta(filename,mass=self.masses(pid0))
                    
                # get sample of LLP momenta in the mother's rest frame
                m0, m1, m2, m3= self.masses(pid0), self.masses(pid1,mass), self.masses(pid2,mass), mass
                momenta_llp, weights_llp = self.decay_in_restframe_3body(br, m0, m1, m2, m3, nsample)
                    
                # loop through all mother particles, and decay them
                for p_mother, w_mother in zip(momenta_mother, weights_mother):
                    # if mother is shortlived, add factor that requires them to decay before absorption
                    w_decay = self.get_decay_prob(pid0, p_mother)
                    for p_llp,w_lpp in zip(momenta_llp, weights_llp):
                        p_llp_lab=p_llp.boost(-1.*p_mother.boostvector)
                        momenta_lab.append(p_llp_lab)
                        weights_lab.append(w_mother*w_lpp*w_decay)
                        # statistics
                        weight_sum+=w_mother*w_lpp*w_decay
                        if p_llp_lab.pz>100 and p_llp_lab.pt/np.abs(p_llp_lab.pz)<0.1/480.: weight_sum_f+=w_mother*w_lpp*w_decay
    
            # mixing with SM particles
            if model.production[key][0]=="mixing":
                if mass>1.699: continue
                pid, mixing = model.production[key][1], model.production[key][2]
                generator, energy = model.production[key][3], model.production[key][4]
                filename = "files/hadrons/"+energy+"TeV/"+generator+"/"+generator+"_"+energy+"TeV_"+pid+".txt"
                momenta_mother, weights_mother = self.convert_list_to_momenta(filename,mass=self.masses(pid))
                mixing_angle = eval(mixing)
                for p_mother, w_mother in zip(momenta_mother, weights_mother):
                    momenta_lab.append(p_mother)
                    weights_lab.append(w_mother*mixing_angle**2)
                    # statistics
                    weight_sum+=w_mother*mixing_angle**2
                    if p_mother.pz>100 and p_mother.pt/np.abs(p_mother.pz)<0.1/480.: weight_sum_f+=w_mother*mixing_angle**2
    
            # direct production
            if model.production[key][0]=="direct":
                label, energy, coupling_ref = key, model.production[key][1], model.production[key][2]
                condition, massrange =  model.production[key][3], model.production[key][4]
                if massrange is not None:
                    if mass<massrange or mass>massrange[1]: continue
                filename="files/direct/"+energy+"TeV/"+self.model.model_name+"/"+label+"_"+energy+"TeV_"+str(mass)+".txt"
                try:
                    momenta_llp, weights_llp = self.convert_list_to_momenta(filename,mass=mass)
                except:
                    print "did not find file:", filename
                    continue
                for p,w_lpp in zip(momenta_llp, weights_llp):
                    if condition is not None and eval(condition)==0: continue
                    momenta_lab.append(p)
                    weights_lab.append(w_lpp*coupling**2/coupling_ref**2)
                    # statistics
                    weight_sum+=w_lpp*coupling**2/coupling_ref**2
                    if p.pz>100 and p.pt/np.abs(p.pz)<0.1/480.: weight_sum_f+=w_lpp*coupling**2/coupling_ref**2

            #return statistcs
            if print_stats: print key, "{:.2e}".format(weight_sum),"{:.2e}".format(weight_sum_f)
                
        #return
        output=self.convert_to_hist_list(momenta_lab, weights_lab, do_plot=do_plot, filename=filenamesave)
        if do_plot: return output[0]


    ###############################
    #  Counting Events
    ###############################

    def set_detector(self,distance=480, selection="np.sqrt(x.x**2 + x.y**2)< 1", length=5, luminosity=3000, channels=None):
        self.distance=distance
        self.selection=selection
        self.length=length
        self.luminosity=luminosity
        self.channels=channels
    
    def event_passes(self,momentum):
        # obtain 3-momentum
        p=Vector3D(momentum.px,momentum.py,momentum.pz)
        # get position of
        x=float(self.distance/p.z)*p
        if type(x) is np.ndarray: x=Vector3D(x[0],x[1],x[2])
        # check if it passes
        if eval(self.selection): return True
        else:return False
     
    def get_events(self,filename, mass,
            couplings = np.logspace(-8,-3,51),
            nsample=1,
            preselectioncuts="th<0.01 and p>100",
        ):
                    
        # Load Flux file
        particles_llp,weights_llp=self.convert_list_to_momenta(filename=filename, mass=mass,
            filetype="npy", nsample=nsample, preselectioncut=preselectioncuts
        )
        
        # setup different couplings to scan over
        ctaus, brs, nsignals, stat_t, stat_e, stat_w = [], [], [], [], [], []
        model = self.model
        for coupling in couplings:
            ctau = model.get_ctau(mass, coupling)
            if self.channels is None: br = 1.
            else:
                br = 0.
                for channel in self.channels: br+=model.get_br(channel, mass, coupling)
            ctaus.append(ctau)
            brs.append(br)
            nsignals.append(0.)
            stat_t.append([])
            stat_e.append([])
            stat_w.append([])
        
        # loop over particles, and record probablity to decay in volume
        for p,w in zip(particles_llp,weights_llp):
            # check if event passes
            if not self.event_passes(p): continue
            # weight of this event
            weight_event = w*self.luminosity*1000.
            
            #loop over couplings
            for icoup,coup in enumerate(couplings):
                #add event weight
                ctau, br =ctaus[icoup], brs[icoup]
                dbar = ctau*p.p/mass
                prob_decay = math.exp(-self.distance/dbar)-math.exp(-(self.distance+self.length)/dbar)
                nsignals[icoup] += weight_event* coup**2 * prob_decay * br
                stat_t[icoup].append(p.pt/p.pz)
                stat_e[icoup].append(p.e)
                stat_w[icoup].append(weight_event* coup**2 *prob_decay * br)

        return couplings, ctaus, nsignals, stat_e, stat_w, stat_t

    ###############################
    #  Plotting
    ###############################
            
    def plot_reach(self,
            setups,bounds,projections,
            title=None, xlabel=r"Mass [GeV]", ylabel=r"Coupling",
            xlims=[0.01,1],ylims=[10**-6,10**-3], figsize=(7,5), legendloc=None,
        ):
        
        # initiate figure
        matplotlib.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(figsize=figsize)
        zorder=-100
        
        # Future sensitivities
        for projection in projections:
            filename, color, label, posx, posy, rotation = projection
            data=self.readfile("files/models/"+self.model.model_name+"/lines/"+filename)
            ax.plot(data.T[0], data.T[1], color=color, ls="dashed", zorder=zorder, lw=1)
            zorder+=1
        
        
        # Existing Constraints
        for bound in bounds:
            filename, label, posx, posy, rotation = bound
            data=self.readfile("files/models/"+self.model.model_name+"/lines/"+filename)
            ax.fill(data.T[0], data.T[1], color="gainsboro",zorder=zorder)
            ax.plot(data.T[0], data.T[1], color="dimgray"  ,zorder=zorder,lw=1)
            zorder+=1
        
        # labels
        for projection in projections:
            filename, color, label, posx, posy, rotation = projection
            if label is None: continue
            plt.text(posx, posy, label, fontsize=14, color=color, rotation=rotation)
        for bound in bounds:
            filename, label, posx, posy, rotation = bound
            if label is None: continue
            plt.text(posx, posy, label, fontsize=14, color="dimgray", rotation=rotation)
        
        # forward experiment sensitivity
        for setup in setups:
            filename, label, color, ls, alpha, level = setup
            masses,couplings,nsignals=np.load("files/models/"+self.model.model_name+"/results/"+filename)
            m, c = np.meshgrid(masses, couplings)
            n = np.log10(np.array(nsignals).T+1e-20)
            ax.contour (m,c,n, levels=[np.log10(level)]       ,colors=color,zorder=zorder, linestyles=ls)
            ax.contourf(m,c,n, levels=[np.log10(level),10**10],colors=color,zorder=zorder, alpha=alpha)
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
        
        return plt
