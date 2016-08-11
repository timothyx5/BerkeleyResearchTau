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
uv_z = np.array( [ 0.05, 0.30, 0.50, 0.70, 1.00, 0.04, 0.13, 0.30,
                   0.50, 0.70, 0.90, 1.10, 1.45, 2.10, 3.00, 4.00,
                   1.10, 1.75, 2.23, 2.30, 3.05, 3.80, 4.90, 5.90,
                   7.00, 7.07, 7.90, 7.05, 7.95, 9.00, 10.0, 10.4 ] )  #Averaged for values in a range
uv_data = np.array( [ -1.7970, -1.3920, -1.2820, -1.0920, -1.14200, -1.74300,
                      -1.7350, -1.5260, -1.4235, -1.2315, -0.98302, -0.93115,
                      -0.9322, -0.7112, -0.9624, -1.5300, -0.94740, -0.67740,
                      -0.7974, -0.5900, -0.8100, -1.1380, -1.34500, -1.57600,
                      -1.6000, -1.6480, -1.9090, -1.7820, -1.91500, -1.98800,
                      -2.3240, -2.4060 ] )
uv_data = 10**uv_data
uv_err = np.array( [ 0.04, 0.15, 0.150, 0.230, 0.23, 0.100, 0.190, 0.13,
                     0.11, 0.11, 0.091, 0.093, 0.10, 0.100, 0.170, 0.36,
                     0.15, 0.19, 0.160, 0.140, 0.19, 0.072, 0.085, 0.13,
                     0.21, 0.15, 0.420, 0.250, 0.27, 0.390, 0.630, 0.74 ] )
uv_err_up = uv_data * 10**uv_err - uv_data
uv_err_low = -uv_data * 10**-uv_err + uv_data
          
#IR data from Brant Robertson recomputed from MD14
ir_z    = np.array( [ 0.03, 0.03, 0.55, 0.85, 1.15, 1.55, 2.05, 0.55, 0.85,
                      1.15, 1.55, 2.05, 0.15, 0.38, 0.53, 0.70, 0.90, 1.10,
                      1.45, 1.85, 2.25, 2.75, 3.60] ) #Averaged for values in a range
ir_data = np.array( [ -1.65250, -1.88250, -1.27250, -0.89250, -0.82250, -0.84250,
                      -0.82250, -1.15250, -1.03250, -0.89250, -0.87250, -0.73250,
                      -1.63465, -1.41241, -1.31241, -1.13241, -0.93241, -0.80241,
                      -0.83241, -0.85241, -0.90241, -0.85241, -1.35241] )
ir_data = 10**ir_data
ir_err = np.array( [ 0.10000, 0.20300, 0.17000, 0.25000, 0.27000, 0.27000,
                     0.31000, 0.17000, 0.19000, 0.26000, 0.24000, 0.21000,
                     0.11535, 0.04759, 0.05759, 0.06759, 0.05759, 0.05759,
                     0.04759, 0.03759, 0.12750, 0.2375 , 0.5076 ] )
ir_err_up = ir_data * 10**ir_err - ir_data
ir_err_low = -ir_data * 10**-ir_err + ir_data

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
