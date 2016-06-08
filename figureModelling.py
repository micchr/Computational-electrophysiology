# in vestibular ganglion neurons.
# The code has partly served to construct the last figure of the paper
# This file presents the last part of a study published in Michel CB et al., Eur J Neurosci. 2015 Nov;42(10):2867-77. 
# doi: 10.1111/ejn.13021. Epub 2015 Aug 6. Identification and modelling of fast and slow Ih current components 

from __future__ import division
from numpy import *
from pylab import *
from scipy.integrate import odeint
from detect_peaks import detect_peaks

close('all')

celsius = 37
# the following represent the temperature issues to take account the recording have been made 
# at room temperature (22 C), and to model the neurons activity at physiological temperature (37 C)
q10 = 3**((celsius - 22)/10)
q10_Ihf = 3**((celsius - 22)/10) # temperature adaptation for the fast component
q10_Ihs = 3**((celsius - 22)/10) # temperature adaptation for the slow component (in some case, it can be different)

# the recording have been made on vestibular neurons so the neuron model has to reproduce their 
# electrophysiological pattern to be meaningfull. The vestibular neuron model is constructed with known 
# ionic channel, I had only had to adjust the maximal conductances to obtain correct vestibular models
# (reproducing both transient and sustained activity in the paper, here only sustained)
####################

# Na channel (activation)
m_inf   = lambda v: 1/(1+exp(-(v + 38)/7))
tm_inf  = lambda v: (10/(5*exp((v + 60)/18) + 36*exp(-(v + 60)/25)) + 0.04)/q10
h_inf   = lambda v: 1/(1+exp((v + 65)/6))
th_inf  = lambda v: (100/(7*exp((v + 60)/11) + 10*exp(-(v + 60)/25)) + 0.6)/q10

# KH channel (high voltage activated potassium channel)
n_inf   = lambda v: 1/((1+exp(-(v + 15)/5))**0.5)
p_inf   = lambda v: 1/(1+exp(-(v + 23)/6))
tn_inf  = lambda v: (100/(11*exp((v + 60)/24) + 21*exp(-(v + 60)/23)) + 0.7)/q10
tp_inf  = lambda v: (100/(4*exp((v + 60)/32) + 5*exp(-(v + 60)/22)) + 5)/q10

# KL channel (low voltage activated potassium channel)
w_inf   = lambda v: 1/((1+exp(-(v + 48)/6))**0.25)
z_inf   = lambda v: 0.5/(1+exp((v + 71)/10)) + 0.5
tw_inf  = lambda v: (100/(6*exp((v + 60)/6) + 16*exp(-(v + 60)/45)) + 1.5)/q10
tz_inf  = lambda v: (1000/(exp((v + 60)/20) + exp(-(v + 60)/8)) + 50)/q10

# h channel (the ones we are currently testing)
rs_inf   = lambda v: pow(1/(1+exp((v + 111)/5.5)),1)
trs  = lambda v: (75+1030*exp(-((-83-v)**2)/(61**2)))/q10_Ihs
rf_inf   = lambda v: pow(1/(1+exp((v + 130)/9.1)),1)
trf  = lambda v: (69+265*exp(-((-80-v)**2)/(31**2)))/q10_Ihf

### channel activity ###
v = arange(-150,51) # mV

V_rest  = -67     # mV
# the maximal conductances have been settled to reproduce the electrophysiological patterns of actual 
# vestibular neurons
Cm      = 1      # pF
gbar_Na = 1000
gbar_KH = 140
gbar_KL = 0

gbar_l  = 2       # nS
# Reversal potentials of ionic channels and leak
E_Na    = 50      # mV
E_K     = -70     # mV
E_h     = -43     # mV
E_l     = -67     # mV

## Simulate Model

def RMsolve(x,t):
  # ode representing the openning and closing of each conductance involved in the vestibular neuron 
  # electrophysiological activity (following Hodgkin-Huxley formalism)
  # sodium gate
  dx0 = (m_inf(x[8]) - x[0])/tm_inf(x[8])
  dx1 = (h_inf(x[8]) - x[1])/th_inf(x[8])
  # KH gate (high voltage activated potassium channel)
  dx2 = (n_inf(x[8]) - x[2])/tn_inf(x[8])
  dx3 = (p_inf(x[8]) - x[3])/tp_inf(x[8])
  # KH gate (low voltage activated potassium channel)
  dx4 = (w_inf(x[8]) - x[4])/tw_inf(x[8])
  dx5 = (z_inf(x[8]) - x[5])/tz_inf(x[8])
  # hyperpolarization activated cation channels gate
  dx6 = (rs_inf(x[8]) - x[6])/trs(x[8])
  dx7 = (rf_inf(x[8]) - x[7])/trf(x[8])

  # associated currents
  I_Na = gbar_Na*(x[0]**3)*x[1]*(x[8] - E_Na)
  I_KH = gbar_KH*(0.85*(x[2]**2) + 0.15*x[3])*(x[8] - E_K)
  I_KL = gbar_KL*(x[4]**4)*x[5]*(x[8] - E_K)
  I_hs = gbar_hs*x[6]*(x[8] - E_h)
  I_hf = gbar_hf*x[7]*(x[8] - E_h)
  I_l = gbar_l*(x[8] - E_l)
  
  # current clamp stimulation
  if int(t/Te) < len(time):
    I = int(t/Te)
  else:
    I = len(time)-1
  
  # potential variations
  dx8 = (Istim[I] - I_Na - I_KH - I_KL - I_hs - I_hf - I_l) / Cm

  return [dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8]

# duration and sample frequency of the simulation
T     = 300    # ms
Te    = 0.01   # ms
time  = arange(0,T+Te,Te)

V0 = -65

# settling of initial conditions
X0 = [m_inf(V0),h_inf(V0),n_inf(V0),p_inf(V0),w_inf(V0),z_inf(V0),rs_inf(V0),rf_inf(V0),V0]

figure(1,facecolor=[1,1,1], figsize=(8, 9))
rc('font',size=9)
rcParams['mathtext.default'] = 'regular'

for k in [1,2]:
  # a triple loop is launched to present the response of the channels for 3 different stimulation durations (loop l)
  # and two different stimulation amplitudes (loop j) for both Ih components (loop k)
  for j in [1,2]:

    subplot(3,4,4*(k-1)+j)
    hold

    NbTest = 3
    # loop on the amplitude and duration current stimulation
    for l in arange(1,NbTest+1):
    	
      Istim = zeros(len(time))
      # current clamp negative stimulations (se paper for duration and amplitude physiological relevence) 
      Imax = -250-50*(j-1) # amplitude setting
      Istim[50/Te:(50+l*60)/Te] = Imax # duration setting
            
      if k == 1:
      	# here the choice is made of the Ih conductance component activated
      	# fast component (the slow one is settled to 0)
        gbar_hf = 3.43
        gbar_hs = 0
        text(50,90,r'$G_{hf}$ = '+str(gbar_hf)+' nS')
          
        xticks(arange(0,350,50))

      elif k == 2:
      	# slow component (the fast one is settled to 0)
        gbar_hf = 0
        gbar_hs = 3.34
        text(50,90,r'$G_{hs}$ = '+str(gbar_hs)+' nS')
          
        xlabel('t (ms)')
          
      text(80,-180,'$I_{max}$ = '+str(int(round(Imax)))+' pA')

      y1 = odeint(RMsolve, X0, time)
	  
      if l==1:
      	# color choice for clear result display
        plot(time,y1[:,8],color=[0.6,0.6,0.6])
        plot(time,-Istim/Imax*15-220,color=[0.6,0.6,0.6]) 
      if l==2:
        plot(time,y1[:,8],color=[0.3,0.3,0.3])
        plot(time,-Istim/Imax*15-220,color=[0.3,0.3,0.3]) 
      if l==3:
        plot(time,y1[:,8],'k')
        plot(time,-Istim/Imax*15-220,color='k') 
		
    xticks(arange(0,400,100))

    ylim([-250,150])
    
    if j == 1:
      text(-112,-10,'Vm (mV)',rotation=90,fontsize=10)
    else:
      yticks(arange(-250,150,50),{})

show()





