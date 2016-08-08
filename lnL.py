
import numpy as np
import Tau

n = np.load('/Users/TJW/BerkeleyResearchTau/ionHist.npz')

rho_uv_y_error = (np.log10(Tau.uv_err_up) - np.log10(Tau.uv_err_low))*0.5

rho_ir_y_error = (np.log10(Tau.ir_err_up) - np.log10(Tau.ir_err_low))*0.5

rho_SFR_error = np.concatenate((rho_uv_y_error, rho_ir_y_error))

y =  np.concatenate((np.array(np.log10(Tau.uv_data)), np.array(np.log10(Tau.ir_data))))

sigma = rho_SFR_error

def m(ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	tau, Q, z, Q_adrian, rho_uv, rho_ir = Tau.calc_tau_Q_rho(ap=ap,bp=bp,cp=cp,dp=dp)
	return np.concatenate((np.log10(rho_uv), np.log10(rho_ir)))
