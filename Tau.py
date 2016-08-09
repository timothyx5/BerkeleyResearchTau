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

#UV data from Brant Robertson recomputed from MD14
uv_z	= [5.000000e-02, 3.000000e-01, 5.000000e-01, 7.000000e-01, 1.000000e+00, 4.000000e-02, 1.300000e-01,
		   3.000000e-01, 5.000000e-01, 7.000000e-01, 9.000000e-01, 1.100000e+00, 1.450000e+00, 2.100000e+00,
		   3.000000e+00, 4.000000e+00, 1.100000e+00, 1.750000e+00, 2.230000e+00, 2.300000e+00, 3.050000e+00,
		   3.800000e+00, 4.900000e+00, 5.900000e+00, 7.000000e+00, 7.070000e+00, 7.900000e+00, 7.050000e+00,
		   7.950000e+00, 9.000000e+00, 1.000000e+01, 1.040000e+01 ]  #Averaged for values in a range
uv_data = [10**-1.797000e+00,10**-1.392000e+00,10**-1.282000e+00,10**-1.092000e+00,10**-1.142000e+00,10**-1.743000e+00,
		   10**-1.735000e+00,10**-1.526000e+00,10**-1.423500e+00,10**-1.231500e+00,10**-9.830200e-01,10**-9.311500e-01,
		   10**-9.322000e-01,10**-7.112000e-01,10**-9.624000e-01,10**-1.530000e+00,10**-9.474000e-01,10**-6.774000e-01,
		   10**-7.974000e-01,10**-5.900000e-01,10**-8.100000e-01,10**-1.138000e+00,10**-1.345000e+00,10**-1.576000e+00,
		   10**-1.600000e+00,10**-1.648000e+00,10**-1.909000e+00,10**-1.782000e+00,10**-1.915000e+00,10**-1.988000e+00,
		   10**-2.324000e+00,10**-2.406000e+00]
uv_err_up = [ uv_data[0]*10**4.000000e-02,uv_data[1]*10**1.500000e-01,uv_data[2]*10**1.500000e-01,uv_data[3]*10**2.300000e-01,
			  uv_data[4]*10**2.300000e-01,uv_data[5]*10**1.000000e-01,uv_data[6]*10**1.900000e-01,uv_data[7]*10**1.300000e-01,
			  uv_data[8]*10**1.100000e-01,uv_data[9]*10**1.100000e-01,uv_data[10]*10**9.100000e-02,uv_data[11]*10**9.300000e-02, 
			  uv_data[12]*10**1.000000e-01,uv_data[13]*10**1.000000e-01,uv_data[14]*10**1.700000e-01,uv_data[15]*10**3.600000e-01,
			  uv_data[16]*10**1.500000e-01,uv_data[17]*10**1.900000e-01,uv_data[18]*10**1.600000e-01,uv_data[19]*10**1.400000e-01,
			  uv_data[20]*10**1.900000e-01,uv_data[21]*10**7.200000e-02,uv_data[22]*10**8.500000e-02,uv_data[23]*10**1.300000e-01, 
			  uv_data[24]*10**2.100000e-01,uv_data[25]*10**1.500000e-01,uv_data[26]*10**4.200000e-01,uv_data[27]*10**2.500000e-01,
			  uv_data[28]*10**2.700000e-01,uv_data[29]*10**3.900000e-01,uv_data[30]*10**6.300000e-01,uv_data[31]*10**7.400000e-01]
uv_err_low = [ uv_data[0]*10**-4.000000e-02,uv_data[1]*10**-1.500000e-01,uv_data[2]*10**-1.500000e-01,uv_data[3]*10**-2.300000e-01,
			  uv_data[4]*10**-2.300000e-01,uv_data[5]*10**-1.000000e-01,uv_data[6]*10**-1.900000e-01,uv_data[7]*10**-1.300000e-01,
			  uv_data[8]*10**-1.100000e-01,uv_data[9]*10**-1.100000e-01,uv_data[10]*10**-9.100000e-02,uv_data[11]*10**-9.300000e-02, 
			  uv_data[12]*10**-1.000000e-01,uv_data[13]*10**-1.000000e-01,uv_data[14]*10**-1.700000e-01,uv_data[15]*10**-3.600000e-01,
			  uv_data[16]*10**-1.500000e-01,uv_data[17]*10**-1.900000e-01,uv_data[18]*10**-1.600000e-01,uv_data[19]*10**-1.400000e-01,
			  uv_data[20]*10**-1.900000e-01,uv_data[21]*10**-7.200000e-02,uv_data[22]*10**-8.500000e-02,uv_data[23]*10**-1.300000e-01, 
			  uv_data[24]*10**-2.100000e-01,uv_data[25]*10**-1.500000e-01,uv_data[26]*10**-4.200000e-01,uv_data[27]*10**-2.500000e-01,
			  uv_data[28]*10**-2.700000e-01,uv_data[29]*10**-3.900000e-01,uv_data[30]*10**-6.300000e-01,uv_data[31]*10**-7.400000e-01]
 		 
#IR data from Brant Robertson recomputed from MD14
ir_z	= [3.000000e-02, 3.000000e-02, 5.500000e-01, 8.500000e-01, 1.150000e+00, 1.550000e+00, 2.050000e+00, 5.500000e-01,
		   8.500000e-01, 1.150000e+00, 1.550000e+00, 2.050000e+00, 1.500000e-01, 3.800000e-01, 5.300000e-01, 7.000000e-01,
		   9.000000e-01, 1.100000e+00, 1.450000e+00, 1.850000e+00, 2.250000e+00, 2.750000e+00, 3.600000e+00] #Averaged for values in a range
ir_data = [10**-1.652500e+00,10**-1.882500e+00,10**-1.272500e+00,10**-8.925000e-01,10**-8.225000e-01,10**-8.425000e-01,10**-8.225000e-01,
           10**-1.152500e+00,10**-1.032500e+00,
		   10**-8.925000e-01,10**-8.725000e-01,10**-7.325000e-01,10**-1.634650e+00,10**-1.412410e+00,10**-1.312410e+00,10**-1.132410e+00,
		   10**-9.324100e-01,10**-8.024100e-01,
		   10**-8.324100e-01,10**-8.524100e-01,10**-9.024100e-01,10**-8.524100e-01,10**-1.352410e+00]
ir_err_up = [ ir_data[0]*10**1.000000e-01,ir_data[1]*10**2.030000e-01,ir_data[2]*10**1.700000e-01,ir_data[3]*10**2.500000e-01,
			  ir_data[4]*10**2.700000e-01,ir_data[5]*10**2.700000e-01,ir_data[6]*10**3.100000e-01,ir_data[7]*10**1.700000e-01,
			  ir_data[8]*10**1.900000e-01,ir_data[9]*10**2.600000e-01,ir_data[10]*10**2.400000e-01,ir_data[11]*10**2.100000e-01,
			  ir_data[12]*10**1.153500e-01,ir_data[13]*10**4.759000e-02,ir_data[14]*10**5.759000e-02,ir_data[15]*10**6.759000e-02,
			  ir_data[16]*10**5.759000e-02,ir_data[17]*10**5.759000e-02,ir_data[18]*10**4.759000e-02,ir_data[19]*10**3.759000e-02,
			  ir_data[20]*10**1.275000e-01,ir_data[21]*10**2.375000e-01,ir_data[22]*10**5.076000e-01 ]
ir_err_low = [ ir_data[0]*10**-1.000000e-01,ir_data[1]*10**-2.030000e-01,ir_data[2]*10**-1.700000e-01,ir_data[3]*10**-2.500000e-01,
			  ir_data[4]*10**-2.700000e-01,ir_data[5]*10**-2.700000e-01,ir_data[6]*10**-3.100000e-01,ir_data[7]*10**-1.700000e-01,
			  ir_data[8]*10**-1.900000e-01,ir_data[9]*10**-2.600000e-01,ir_data[10]*10**-2.400000e-01,ir_data[11]*10**-2.100000e-01,
			  ir_data[12]*10**-1.153500e-01,ir_data[13]*10**-4.759000e-02,ir_data[14]*10**-5.759000e-02,ir_data[15]*10**-6.759000e-02,
			  ir_data[16]*10**-5.759000e-02,ir_data[17]*10**-5.759000e-02,ir_data[18]*10**-4.759000e-02,ir_data[19]*10**-3.759000e-02,
			  ir_data[20]*10**-1.275000e-01,ir_data[21]*10**-2.375000e-01,ir_data[22]*10**-5.076000e-01 ]

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

def trec(z, C_HII=3, Y_p=0.2453, X_p=0.75, T_4=2): #1./ don't need __future__,
	'''The average recombination time in the IGM as a function of redshift.
	   C_HII is the clumping factor, Yp is the primordial helium mass fraction,
	   X_p is the primordial hydrogen mass fraction,
	   and T_4 is a scaled temperature divided by 10^4(K) Osterbrock & Ferland 2006''' 
	return 1/(C_HII * ALPHA_B * T_4**-0.845 * (1 + Y_p/(4*X_p)) * NH * (1 + z)**3) 

def Qdot(z, Q, ap=0.01376, bp=3.26, cp=2.59, dp=5.68): 
	'''Is a differential equation describing the time derivative of Q_HII(z)
	   which is a dimensionless volume filling faction of ionized hydrogen'''
	return niondot(z, ap=ap, bp=bp, cp=cp, dp=dp)/NH - Q/trec(z)

zbuf, tbuf = None, None

def calc_Q(N=2001,zlow=0,zhigh=20, ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	global zbuf, tbuf
	z = np.linspace(zhigh,zlow,N)
	if zbuf is None or np.any(zbuf != z):
		t_Gyr = np.array(cosmo.age(z)) # In units of Gyr
		t = t_Gyr*GYR_S # Time t in seconds
		zbuf, tbuf = z, t
	else:
		z, t = zbuf, tbuf
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
	Q = calc_Q(z)
	index = np.where(z ==  redshift)
	return Q[index]

def f_e(z):
	#f_e is not f_esc, it is free electrons per hydrogen nucleus, 1+ y/2x for z <= 4, and 1+ y/4x else.
	low_z = np.where(z <= 4)
	electron_frac = np.ones(z.size)*1.083
	electron_frac[low_z] = 1.167
	return electron_frac

def dtau(z, Q):
	'''taud(z) is the ThomsonOptical Depth differential that must be integrated over z.
	   The factor of'''
	return (c * NH * SIGMA_T * f_e(z) * Q * (1 + z)**2)/(cosmo.H(z)/KM_PER_MPC)

# tau is the result of integrating dtau over redshift z
def calc_tau_Q_rho(N=2001,zhigh=20,zlow=0, ap=0.01376, bp=3.26, cp=2.59, dp=5.68):

	# Calculate Q (we get rho for free)
	Q, z = calc_Q(ap=ap,bp=bp,cp=cp,dp=dp)

	# Integrate Q to get Tau
	tau = integrate.cumtrapz(dtau(z, Q)[::-1], z[::-1])[::-1]

	# Match Q
	Q_adrian = Q[700:1410][::10]

	# Calc & Match Robertson_z's Rho
	rho_uv = rho(np.array(uv_z),ap=ap,bp=bp,cp=cp,dp=dp)
	rho_ir = rho(np.array(ir_z),ap=ap,bp=bp,cp=cp,dp=dp)

	return tau, Q, z, Q_adrian, rho_uv, rho_ir

def calc_x_HI(ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	Q, z = calc_Q(ap=ap,bp=bp,cp=cp,dp=dp)
	return np.ones(len(Q)) - Q

tau, Q, z, Q_adrian, rho_uv, rho_ir = calc_tau_Q_rho()

electron_frac = f_e(z)
