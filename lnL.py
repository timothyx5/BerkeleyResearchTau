import numpy as np
import Tau

n = np.load('c:/Users/Timot/Desktop/Research/ionHist.npz')

tau_adrian = np.array([0.066])
tau_adrian_error = np.array([0.012])

x_HI_error = n['upperCurve95'] - n['nf0']

rho_uv_y_error = [ 0.06, 0.05, 0.12, 0.22, 0.22, 0.09, 0.18, 0.12, 0.10, 0.10, 
				   0.09, 0.09, 0.12, 0.29, 0.21, 0.27, 0.08, 0.12, 0.09, 0.10,
				   0.13, 0.05, 0.06, 0.08, 0.10, 0.11, 0.11, 0.14 ]

rho_ir_y_error = [ 0.03, 0.20, 0.17, 0.17, 0.24, 0.19, 0.23, 0.09, 0.12, 0.17,
				   0.16, 0.17, 0.10, 0.04, 0.05, 0.06, 0.06, 0.05, 0.04, 0.03,
				   0.11, 0.19, 0.37 ]

rho_SFR_error = np.concatenate((np.array(rho_uv_y_error), np.array(rho_ir_y_error)))

rho_uv = Tau.rho(np.array(Tau.uv_z))
rho_ir = Tau.rho(np.array(Tau.ir_z))

y =  np.concatenate((n['nf0'], tau_adrian, np.array(np.log10(Tau.uv_data)), np.array(np.log10(Tau.ir_data))))

sigma = np.concatenate((x_HI_error, tau_adrian_error, rho_SFR_error))

def m(ap,bp,cp,dp):
	tau, Q, z, Q_adrian, rho_uv, rho_ir = Tau.calc_tau_Q_rho(ap=ap,bp=bp,cp=cp,dp=dp)
	return np.concatenate((Q, [tau[0]], np.log10(rho_uv), np.log10(rho_ir)))
