import numpy as np
import Tau

n = np.load('/Users/TJW/BerkeleyResearchTau/ionHist.npz')

tau_adrian = np.array([0.066])
tau_adrian_error = np.array([0.012])

x_HI_error = n['upperCurve95'] - n['nf0']

y =  n['nf0']

sigma = x_HI_error

def m(ap,bp,cp,dp):
	tau, Q, z, Q_adrian, rho_uv, rho_ir = Tau.calc_tau_Q_rho(ap=ap,bp=bp,cp=cp,dp=dp)
	return Q_adrian
