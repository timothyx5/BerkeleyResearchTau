#!/usr/bin/env python

'''
Thomson Optical Depth

Author: Tim Wilson
'''

from __future__ import division
from scipy import integrate
from astropy.cosmology import Planck15 as cosmo
import numpy as np

###################################################
##########Constants and conversions################
###################################################

MPC3 = (3.1e24)**3 #conversion factor from Megaparsec cubed to cm cubed
ALPHA_B = 2.59e-13 #recombination coefficient at T=10^4 in units of cubic cm per second, (Osterbrock & Ferland 2006).
NH = 1.881e-7 #current hydrogen density in cm^-3
GYR_S = 3.16e16 # to convert Gyr to seconds
KM_PER_MPC = 3.1e19 #for converting Hubble parameter to inverse time only
c = 3.0e10 #c is the speed of light in cm/s
SIGMA_T = 6.65e-25 #SIGMA_T is the Thomson cross section (astrobaki).

###################################################
#################### DATA #########################
###################################################

#UV data from MD14
uv_z	= [0.055,0.3,0.5,0.7,1.0,0.05,0.125,0.3,0.5,0.7,0.9,1.1,1.45,2.1,3.0,4.0,1.25,1.75,2.23,
		   2.3,3.1,3.8,4.9,5.9,7.0,7.9,7.0,8.]  #Averaged for values in a range
uv_data = [10**-1.82,10**-1.50,10**-1.39,10**-1.20,10**-1.25,10**-1.77,10**-1.75,10**-1.55,
		   10**-1.44,10**-1.24,10**-0.99,10**-0.94,10**-0.95,10**-0.75,10**-1.04,10**-1.69,
		   10**-1.02,10**-0.75,10**-0.87,10**-0.75,10**-0.97,10**-1.29,10**-1.42,10**-1.65,
		   10**-1.79,10**-2.09,10**-2.00,10**-2.21]
uv_err_up = [ uv_data[0]*10**0.09,uv_data[1]*10**0.05,uv_data[2]*10**0.15,uv_data[3]*10**0.31,uv_data[4]*10**0.31,uv_data[5]*10**0.08,
			  uv_data[6]*10**0.18,uv_data[7]*10**0.12,uv_data[8]*10**0.10,uv_data[9]*10**0.10,uv_data[10]*10**0.09,uv_data[11]*10**0.09, 
			  uv_data[12]*10**0.15,uv_data[13]*10**0.49,uv_data[14]*10**0.26,uv_data[15]*10**0.22,uv_data[16]*10**0.08,uv_data[17]*10**0.12,
			  uv_data[18]*10**0.09,uv_data[19]*10**0.09,uv_data[20]*10**0.11,uv_data[21]*10**0.05,uv_data[22]*10**0.06,uv_data[23]*10**0.08, 
			  uv_data[24]*10**0.10,uv_data[25]*10**0.11,uv_data[26]*10**0.10,uv_data[27]*10**0.14]
uv_err_low = [ uv_data[0]*10**-0.02,uv_data[1]*10**-0.05,uv_data[2]*10**-0.08,uv_data[3]*10**-0.13,uv_data[4]*10**-0.13,uv_data[5]*10**-0.09,
			   uv_data[6]*10**-0.18,uv_data[7]*10**-0.12,uv_data[8]*10**-0.10,uv_data[9]*10**-0.10,uv_data[10]*10**-0.08,uv_data[11]*10**-0.09, 
			   uv_data[12]*10**-0.08,uv_data[13]*10**-0.09,uv_data[14]*10**-0.15,uv_data[15]*10**-0.32,uv_data[16]*10**-0.08,uv_data[17]*10**-0.12,
			   uv_data[18]*10**-0.09,uv_data[19]*10**-0.11,uv_data[20]*10**-0.15,uv_data[21]*10**-0.05,uv_data[22]*10**-0.06,uv_data[23]*10**-0.08, 
			   uv_data[24]*10**-0.10,uv_data[25]*10**-0.11,uv_data[26]*10**-0.11,uv_data[27]*10**-0.14]

#IR data from MD14
ir_z	= [0.03,0.03,0.55,0.85,1.15,1.55,2.05,0.55,0.85,1.15,1.55,2.05,0.15,0.38,0.53,0.70,0.90,1.10,1.45,
		   1.85,2.25,2.75,3.60] #Averaged for values in a range
ir_data = [10**-1.72,10**-1.95,10**-1.34,10**-0.96,10**-0.89,10**-0.91,10**-0.89,10**-1.22,10**-1.10,
		   10**-0.96,10**-0.94,10**-0.80,10**-1.64,10**-1.42,10**-1.32,10**-1.14,10**-0.94,10**-0.81,
		   10**-0.84,10**-0.86,10**-0.91,10**-0.86,10**-1.36]
ir_err_up = [ ir_data[0]*10**0.02,ir_data[1]*10**0.20,ir_data[2]*10**0.22,ir_data[3]*10**0.15,ir_data[4]*10**0.27,ir_data[5]*10**0.17, 
			  ir_data[6]*10**0.21,ir_data[7]*10**0.08,ir_data[8]*10**0.10,ir_data[9]*10**0.13,ir_data[10]*10**0.13,ir_data[11]*10**0.18,
			  ir_data[12]*10**0.09,ir_data[13]*10**0.03,ir_data[14]*10**0.05,ir_data[15]*10**0.06,ir_data[16]*10**0.05,ir_data[17]*10**0.04,
			  ir_data[18]*10**0.04,ir_data[19]*10**0.02,ir_data[20]*10**0.09,ir_data[21]*10**0.15,ir_data[22]*10**0.23 ]
ir_err_low = [ ir_data[0]*10**-0.03,ir_data[1]*10**-0.20,ir_data[2]*10**-0.11,ir_data[3]*10**-0.19,ir_data[4]*10**-0.21,ir_data[5]*10**-0.21, 
			   ir_data[6]*10**-0.25,ir_data[7]*10**-0.11,ir_data[8]*10**-0.13,ir_data[9]*10**-0.20,ir_data[10]*10**-0.18,ir_data[11]*10**-0.15,
			   ir_data[12]*10**-0.11,ir_data[13]*10**-0.04,ir_data[14]*10**-0.05,ir_data[15]*10**-0.06,ir_data[16]*10**-0.06,ir_data[17]*10**-0.05,
			   ir_data[18]*10**-0.04,ir_data[19]*10**-0.03,ir_data[20]*10**-0.12,ir_data[21]*10**-0.23,ir_data[22]*10**-0.50 ]

###################################################
################## Functions ######################
###################################################

def rho(z, ap=0.01376, bp=3.26, cp=2.59, dp=5.68): #Planck2015
#def rho(z, ap=0.01306, bp=3.66, cp=2.28, dp=5.29): #WMAP
	'''rho(z) represents rhoSFR as a function of redshift,
	   returns star formation rate density in M_Sun yr^-1 Mpc^-3,
	   ap, bp, cp, and dp are the parameters of a ML fitting Planck's tau findings.'''
	return ap * (((1 + z)**bp)/(1+((1 + z)/cp)**dp))

def niondot(z, f_esc=0.2, xi=10**53.14, ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	'''niondot describes the number density of Lymann continuum photonsper second
       capable of reionizing intergalactic hydrogen,
       xi is the (photons/s) per (M_Sun/yr)
       and f is a fiducial value for the fraction of photons escaping from galaxies
       capable of affecting the IGM.'''
	return f_esc * xi * rho(z, ap=ap, bp=bp, cp=cp, dp=dp)/MPC3 #converting rhop to M_Sun yr^-1 cm^-3

def trec(z, C_HII=3, Y_p=0.25, X_p=0.75, T_4=2): #1./ don't need __future__,
	'''The average recombination time in the IGM as a function of redshift.
	   C_HII is the clumping factor, Yp is the primordial helium mass fraction,
	   X_p is the primordial hydrogen mass fraction,
	   and T_4 is a scaled temperature divided by 10^4(K) Osterbrock & Ferland 2006''' 
	return 1/(C_HII * ALPHA_B * T_4**-0.845 * (1 + Y_p/(4*X_p)) * NH * (1 + z)**3) 

def Qdot(z, Q, ap=0.01376, bp=3.26, cp=2.59, dp=5.68): 
	'''Is a differential equation describing the time derivative of Q_HII(z)
	   which is a dimensionless volume filling faction of ionized hydrogen'''
	return niondot(z, ap=ap, bp=bp, cp=cp, dp=dp)/NH - Q/trec(z)

def calc_Q(N=2001,zlow=0,zhigh=20, ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	z = np.linspace(zhigh,zlow,N)
	t_Gyr = np.array(cosmo.age(z)) # In units of Gyr
	t = t_Gyr*GYR_S # Time t in seconds
	Q = np.zeros(N)
	dt = np.zeros(N)
	Qprev = np.zeros(N)
	Qdotprev = np.zeros(N)
	'''Evolving the differential equation Qdot to solve for Q(z) using Euler's method'''
	for i in xrange(1, N):
		Qprev[i] = Q[i-1]
		Qdotprev[i] = Qdot(z[i-1], Qprev[i], ap=ap, bp=bp, cp=cp, dp=dp)
		dt[i] = t[i] - t[i-1]	
		Q[i] = Qprev[i] + dt[i]*Qdotprev[i]
		if Q[i] > 1.0:
			Q[i] = 1.0
	return Q, z

def get_Q(redshift,N=2001,zlow=0,zhigh=20):
	Q, z = calc_Q(N=N,zlow=zlow,zhigh=zhigh)
	index = np.where(z ==  redshift)
	return Q[index]

def f_e(z):
	#f_e is not f_esc, it is free electrons per hydrogen nucleus, 1+ y/2x for z <= 4, and 1+ y/4x else.
	low_z = z <= 4
	electron_frac = np.ones(len(z))*1.083
	electron_frac[low_z] = 1.167
	return electron_frac

def dtau(z, Q):
	'''taud(z) is the ThomsonOptical Depth differential that must be integrated over z.
	   The factor of'''
	return (c * NH * SIGMA_T * f_e(z) * Q * (1 + z)**2)/(cosmo.H(z)/KM_PER_MPC)

# tau is the result of integrating dtau over redshift z
def calc_tau_Q_rho(N=2001,zhigh=20,zlow=0,ap=0.01376, bp=3.26, cp=2.59, dp=5.68):

	# Calculate Q (we get rho for free)
	Q, z = calc_Q(N=2001,zhigh=20,zlow=0,ap=ap,bp=bp,cp=cp,dp=dp)

	# Integrate Q to get Tau
	tau = integrate.cumtrapz(dtau(z, Q)[::-1], z[::-1])[::-1]

	# Match Q
	Q_adrian = Q[700:1410][::10]

	# Calc & Match Robertson_z's Rho
	rho_uv = rho(np.array(uv_z),ap=ap,bp=bp,cp=cp,dp=dp)
	rho_ir = rho(np.array(ir_z),ap=ap,bp=bp,cp=cp,dp=dp)

	return tau, Q, z, Q_adrian, rho_uv, rho_ir

def calc_x_HI(ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	Q, z = calc_Q(N=2001,zhigh=20,zlow=0,ap=ap,bp=bp,cp=cp,dp=dp)
	return np.ones(len(Q)) - Q

tau, Q, z, Q_adrian, rho_uv, rho_ir = calc_tau_Q_rho()

electron_frac = f_e(z)
