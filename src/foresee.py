import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
import random
from skhep.math.vectors import LorentzVector, Vector3D


class Foresee():

    def __init__(self):
        self.started= True
    
    
    ###############################
    #  Hadron Masses and Lifetimes
    ###############################
    
    def masses(self,pid):
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
        list_of_lists = []
        with open(filename) as f:
            for line in f:
                if line[0]=="#":continue
                inner_list = [float(elt.strip()) for elt in line.split( )]
                list_of_lists.append(inner_list)
        return np.array(list_of_lists)

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
    def convert_to_hist_list(self,momenta,weights, do_plot=False, filename=None, do_return=False, ):
        
        #get data
        tmin, tmax, tnum = -6, 0, 120
        pmin, pmax, pnum =  0, 5, 50
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
            np.save(filename,[list_t,list_p,list_w])

        #get plot
        ticks = np.array([[np.linspace(10**(j),10**(j+1),9)] for j in range(-7,6)]).flatten()
        ticks = [np.log10(x) for x in ticks]
        ticklabels = np.array([[r"$10^{"+str(j)+"}$","","","","","","","",""] for j in range(-7,6)]).flatten()
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(10,6))
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
    def get_spectrumplot(self, pid="111", generator="EPOSLHC", energy="14"):
        dirname = "files/hadrons/"+energy+"TeV/"+generator+"/"
        filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
        p,w = self.convert_list_to_momenta(filename,mass=self.masses(pid))
        plt,_,_,_ =self.convert_to_hist_list(p,w)
        return plt



"""
def plot_reach(model,setups,include_default=False, label=None, show_future=False,
    figsize=(7,5),bbox_to_anchor=None, frameon=False, do_legend=True,setuplabels=None):
    
    #load data
    if include_default: setups.insert(0,"default")
    massesv, couplingsv, nsignalsv = [],[],[]
    for setup in setups:
        masses,couplings,nsignals=np.load("files/results/"+model+"_"+setup+".npy")
        massesv.append(masses)
        couplingsv.append(couplings)
        nsignalsv.append(nsignals)
    if setuplabels==None: setuplabels=setups
    
    #initiate figure
    matplotlib.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=figsize)
    zorder=-100
    colors = ["dodgerblue""firebrick"]
    
    #existing bounds
    if model == "DarkPhoton":
        bounds=["LSND","E137","Charm","NuCal","E141","NA64","BaBar","NA48"]
    elif model == "DarkHiggs":
        bounds=["1508.04094","1612.08718","E949","NA62","Charm"]
    elif model == "ALP-W":
        bounds=["SN1987","E137","NuCal","LEP","E949_displ","NA62_1","NA62_2","KOTO","KTEV","NA6264","E949_prompt","CDF","PbPb",]

    #existing bounds
    if model == "DarkPhoton":
        limits=[["FASER1",       "dashed",  "red",          1 ],
                ["FASER2",       "dashed",  "firebrick",    1 ],
                ["SeaQuest",     "dashed",  "lime",         1 ],
                ["NA62",         "dashed",  "limegreen",    1 ],
                ["SHiP",         "dashed",  "forestgreen",  1 ],
                ["HPS",          "dashed",  "deepskyblue",  1 ],
                ["HPS-1",        "dashed",  "deepskyblue",  0 ],
                ["Belle2",       "dashed",  "blue",         1 ],
                ["LHCb",         "dashed",  "dodgerblue",   1 ],
                ["LHCb-mumu1",   "dashed",  "dodgerblue",   0 ],
                ["LHCb-mumu2",   "dashed",  "dodgerblue",   0 ],
        ]
    if model == "DarkHiggs":
        limits=[["SHiP",         "dashed",  "teal",         1 ],
                ["MATHUSLA",     "dashed",  "dodgerblue",   1 ],
                ["CodexB",       "dashed",  "deepskyblue",  1 ],
                ["LHCb",         "dashed",  "cyan",         1 ],
                ]
    if model == "ALP-W":
        limits=[["Belle2-3gamma", "dashed", "royalblue",   0],
                ["KOTO-2gamma",   "dashed", "cyan",        0],
                ["KOTO-4gamma",   "dashed", "blue",        0],
                ["NA62-0gamma1",  "dashed", "dodgerblue",  0],
                ["NA62-0gamma2",  "dashed", "dodgerblue",  0],
                ["NA62-2gamma",   "dashed", "deepskyblue", 0],
                ["LHC",           "dashed", "teal",        0],
        ]

    if model in ["DarkHiggs"]:
        bound_file=readfile("files/"+model+"/bounds/anomaly_KOTO.txt")
        ax.plot(bound_file.T[0], bound_file.T[1], color="limegreen",zorder=zorder, lw=1)
        ax.fill(bound_file.T[0], bound_file.T[1], color="limegreen",zorder=zorder, alpha=0.15)
    
    if show_future:
        for limit in limits:
            bound_file=readfile("files/"+model+"/bounds/limits_"+limit[0]+".txt")
            ax.plot(bound_file.T[0], bound_file.T[1], color=limit[2], ls=limit[1], zorder=zorder, lw=1)
            if limit[3]==1: ax.plot([10**-10],[10**-10],color=limit[2], ls=limit[1],label=limit[0])
            zorder+=1

    for bound in bounds:
        bound_file=readfile("files/"+model+"/bounds/bounds_"+bound+".txt")
        ax.fill(bound_file.T[0], bound_file.T[1], color="gainsboro",zorder=zorder)
        ax.plot(bound_file.T[0], bound_file.T[1], color="dimgray"  ,zorder=zorder,lw=1)
        zorder+=1
        
    #text
    if model == "DarkHiggs":
        plt.text(0.330, 2.2*10**-3, r"LHCb $B^0$", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.430, 2.2*10**-3, r"LHCb $B^+$", fontsize=12,color="dimgray",rotation=90)
        plt.text(2.500, 2.2*10**-3, r"LHCb $B^+$", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.250, 4.0*10**-4, "CHARM", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.102, 9.2*10**-4, "E949", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.180, 8.2*10**-4, "NA62", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.120, 1.8*10**-4, "KOTO", fontsize=12,color="limegreen",rotation=0)
    if model == "DarkPhoton":
        plt.text(0.630, 5.0*10**-3, "BaBar", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.080, 5.0*10**-3, "NA48", fontsize=12,color="dimgray",rotation=60)
        plt.text(0.014, 3.4*10**-4, "NA64", fontsize=12,color="dimgray",rotation=-22)
        plt.text(0.011, 5.4*10**-5, "E141", fontsize=12,color="dimgray",rotation=22)
        plt.text(0.022, 2.2*10**-5, "NuCal", fontsize=12,color="dimgray",rotation=-17)
        plt.text(0.120, 3.3*10**-8, "LSND", fontsize=12,color="dimgray",rotation=0)
        plt.text(0.120, 1.4*10**-7, "CHARM", fontsize=12,color="dimgray",rotation=-5)
        plt.text(0.020, 4.1*10**-8, "E137", fontsize=12,color="dimgray",rotation=5)
        plt.text(0.120, 3.3*10**-8, "LSND", fontsize=12,color="dimgray",rotation=0)
    if model == "ALP-W":
        plt.text(4.000, 6.7*10**-4, "LEP", fontsize=12,color="dimgray",rotation=0)
        plt.text(0.065, 7.5*10**-4, "CDF", fontsize=12,color="dimgray",rotation=-12)
        plt.text(0.200, 1.3*10**-3, "KTEV", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.235, 1.3*10**-3, "NA62", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.270, 1.3*10**-3, "+NA48/2", fontsize=12,color="dimgray",rotation=90)
        plt.text(0.065, 9.0*10**-5, "E949", fontsize=12,color="dimgray",rotation=-9)
        plt.text(0.090, 3.4*10**-5, "KOTO", fontsize=12,color="dimgray",rotation=9)
        plt.text(0.065, 9.2*10**-6, "NA62", fontsize=12,color="dimgray",rotation=2)
        plt.text(0.065, 3.0*10**-6, "E949", fontsize=12,color="dimgray",rotation=-5)
        plt.text(0.100, 1.2*10**-6, "E137", fontsize=12,color="dimgray",rotation=-8)
        plt.text(0.100, 1.9*10**-7, "SN1987", fontsize=12,color="dimgray",rotation=25)
    if model == "ALP-W" and show_future:
        plt.text(0.370, 3.0*10**-4, r"KOTO $4\gamma$", fontsize=12,color="blue",rotation=0)
        plt.text(1.400, 1.7*10**-4, r"Belle2 $3\gamma$", fontsize=12,color="royalblue",rotation=0)
        plt.text(0.370, 1.0*10**-4, r"NA62 $2\gamma$", fontsize=12,color="deepskyblue",rotation=0)
        plt.text(0.220, 2.0*10**-5, r"NA62 $0\gamma$", fontsize=12,color="dodgerblue",rotation=0)
        plt.text(0.060, 2.0*10**-4, r"KOTO $2\gamma$", fontsize=12,color="cyan",rotation=0)
        plt.text(2.500, 3.5*10**-6, r"LHC $Z\to3\gamma$", fontsize=12,color="teal",rotation=0)
    
    #FASER limits
    linestyles=["solid","dashdot","dashed","dotted"]
    colors = ["red","firebrick","darkorange","magenta","gold","darkmagenta"]
    levels = [np.log10(3)]
    if model in ["Pospelov"]: levels = [np.log10(3),1,2,3]
    for i,(setup,color,linelabel) in enumerate(zip(setups,colors,setuplabels)):
        masses=massesv[i]
        couplings=couplingsv[i]
        nsignals=nsignalsv[i]
        m, c = np.meshgrid(masses, couplings)
        n=np.log10(np.array(nsignals).T+1e-20)
        ax.contour(m,c,n, levels=levels,colors=color,zorder=zorder, linestyles=linestyles)
        #ax.contourf(m, c, n, levels=[levels[0],10**10],colors=colors[i],zorder=zorder,alpha=0.1)
        ax.plot([10**-10],[10**-10],color=color,label=linelabel)
        #ax.plot([10**-10],[10**-10],color=color_def,linestyle="dashed",label="default")
        zorder+=1
    
    #frame
    if label is not None: ax.set_title(label)
    if model in ["DarkPhoton"]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01,2)
        ax.set_ylim(10**-8,0.01)
        ax.set_xlabel(r"dark photon mass $m_{A'}$ [GeV]")
        ax.set_ylabel(r"kinetic mixing $\epsilon$")
    if model in ["DarkHiggs"]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1,10)
        ax.set_ylim(3*10**-6,3*0.001)
        ax.set_xlabel("mass [GeV]")
        ax.set_ylabel("coupling")
    if model in ["ALP-W"]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.06,6)
        ax.set_ylim(2*10**-8,0.002)
        ax.set_xlabel("$m_a$ [GeV]")
        ax.set_ylabel(r"$g_{aWW}$ [GeV$^{-1}$]")
    if do_legend: ax.legend(loc="upper right", bbox_to_anchor=bbox_to_anchor, frameon=frameon, labelspacing=0.2)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.11, top=0.98)
    if label == None: label = model
    plt.savefig("/Users/felixkling/Downloads/"+label+".pdf")

"""
