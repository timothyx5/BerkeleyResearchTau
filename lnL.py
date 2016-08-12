import numpy as np
import Tau

n = np.load('/Users/TJW/BerkeleyResearchTau/ionHist.npz')

tau_adrian = np.array([tau[0]])
tau_adrian_error = np.array([0.012])

x_HI_error = np.zeros(len(n['nf0']))
for i in range(len(n['nf0'])):
	x_HI_error[i] = 0.5*np.mean(n['upperCurve95'][i] - n['nf0'][i])

rho_SFR_err = np.concatenate((np.array(Tau.uv_err), np.array(Tau.ir_err)))

y =  np.concatenate((np.ones(len(Tau.Q_adrian)) - Tau.Q_adrian, tau_adrian, np.array(np.log10(Tau.uv_data)), np.array(np.log10(Tau.ir_data))))

sigma = np.concatenate((x_HI_error, tau_adrian_error, rho_SFR_err))

def m(ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	tau, Q, z, Q_adrian, rho_uv, rho_ir = Tau.calc_tau_Q_rho(ap=ap,bp=bp,cp=cp,dp=dp)
	return np.concatenate((Tau.calc_x_HI(ap,bp,cp,dp)[700:1410][::10], [tau[0]], np.log10(rho_uv), np.log10(rho_ir)))
