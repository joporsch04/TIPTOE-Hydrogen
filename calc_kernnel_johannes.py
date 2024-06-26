import numexpr as ne
import os
import numpy as np
from numpy import interp as interp1d
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
#import matplotlib.pyplot as plt


pi=np.pi
CPU=os.cpu_count()
ne.set_num_threads(CPU)
from numba import jit
import cmath, math
import os
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class AU:
	meter = 5.2917720859e-11 # atomic unit of length in meters
	nm = 5.2917721e-2 # atomic unit of length in nanometres
	second = 2.418884328e-17 # atomic unit of time in seconds
	fs = 2.418884328e-2 # atomic unit of time in femtoseconds
	Joule = 4.359743935e-18 # atomic unit of energy in Joules
	eV = 27.21138383 # atomic unit of energy in electronvolts
	Volts_per_meter = 5.142206313e+11 # atomic unit of electric field in V/m
	Volts_per_Angstrom = 51.42206313 # atomic unit of electric field in V/Angström
	speed_of_light = 137.035999 # vacuum speed of light in atomic units
	Coulomb = 1.60217646e-19 # atomic unit of electric charge in Coulombs
	PW_per_cm2_au = 0.02849451308 # PW/cm^2 in atomic units
AtomicUnits=AU

def integrate_oscillating_function_numexpr(X, f, phi, phase_step_threshold=1e-3):
	r""" The function evaluates \int dx f(x) exp[i phi(x)] using an algorithm
	suitable for integrating quickly oscillating functions.

	Parameters
	----------
	X: a vector of sorted x-values;
	f: either a vector or a matrix where each column contains
		the values of a function f(x);
	phi: either a vector or a matrix where each column contains
		the values of a real-valued phase phi(x);
	phase_step_threshold (float): a small positive number; the formula that
		approximates the integration of an oscillating function over a
		small interval contains phi(x+dx)-phi(x) in the denominator;
		this parameter prevents divisions by very small numbers.

	Returns
	-------
	result: a row vector where elements correspond to the columns in
	  'f' and 'phi'.
	"""

	# check that the input data is OK
	assert(X.shape[0] == f.shape[0])
	assert(X.shape[0] == phi.shape[0])
	assert(np.all(np.imag(phi)) == 0)
	# evaluate the integral(s)
	dx = X[1:] - X[:-1]
	f1 = f[:-1, ...]
	f2 = f[1:, ...]
	phi1=phi[1:, ...]
	phi2=phi[:-1, ...]
	df = ne.evaluate("f2 - f1")
	f_sum=ne.evaluate("f2+f1")
	dphi = ne.evaluate("phi1-phi2")
	#phi_sum= ne.evaluate("0.5*1j*(phi1 + phi2)")
	s = np.ones((f.ndim), dtype=int)
	s[0] = dx.size
	Z = dx.reshape(s)
	Z=ne.evaluate("Z*exp(0.5*1j*(phi1 + phi2))")
	#del phi1, phi2
	Z=ne.evaluate("where(abs(dphi).real < phase_step_threshold, Z * (0.5 * f_sum + 0.125*1j * dphi * df), Z / dphi**2 * (exp(0.5*1j * dphi) * (df - 1j*f2*dphi)-(df - 1j*f1*dphi) / exp(0.5*1j * dphi)))")
	return ne.evaluate("sum(Z, axis=0)")

IOF=integrate_oscillating_function_numexpr

@jit(nopython=True, parallel=True,fastmath = True, nogil=True, cache=True)
def Electric_Field(t,lam0, I, cep, FWHM, pi=np.pi, N=8): 
    """" return E(t) for given Full with half maximum and phase with a cos8 envelope  """
    #print('Electric_Field is deprecated, use Vector_potential instead')
    w0=2*pi*137.035999/(lam0/5.2917721e-2)
    A0=np.sqrt(I/1e15*0.02849451308/w0**2)
    tau_injection=pi*FWHM/(4*np.arccos(2**(-1/(2*N))))
    t=np.asarray(t)
    Field= np.where(np.abs(t)<=tau_injection,np.cos(pi*t/tau_injection/2)**N*np.cos(w0*t-cep)*w0-(N/2/tau_injection*pi*np.cos(pi*t/tau_injection/2)**(N-1)*np.sin(pi*t/tau_injection/2)*np.sin(w0*t-cep)),0)
    return -1*Field*A0


@jit(nopython=True, parallel=True,fastmath = True, nogil=True, cache=True)
def interpFuncCumA(x,lam0, I, cep, FWHM, pi=np.pi, N=8):
	""" return the cumulative vector potential for a given pulse """
	w0=2*pi*137.035999/(lam0/5.2917721e-2)
	tau_injection=int(pi*FWHM/(4*np.arccos(2**(-1/(2*N)))))+2
	T_grid_field_sample=np.arange(-tau_injection,tau_injection, 0.0078125)
	return interp1d(x,T_grid_field_sample,np.cumsum(Vector_potential(T_grid_field_sample, lam0, I, cep, FWHM)*np.diff(T_grid_field_sample)[0]))

@jit(nopython=True, parallel=True,fastmath = True, nogil=True, cache=True)
def interpFuncCumA2(x,lam0, I, cep, FWHM, pi=np.pi, N=8) :
	""" return the cumulative square of vector potential for a given pulse """
	w0=2*pi*137.035999/(lam0/5.2917721e-2)
	tau_injection=int(pi*FWHM/(4*np.arccos(2**(-1/(2*N)))))+2
	T_grid_field_sample=np.arange(-tau_injection,tau_injection, 0.0078125)
	return interp1d(x,T_grid_field_sample,np.cumsum((Vector_potential(T_grid_field_sample, lam0, I, cep, FWHM))**2*np.diff(T_grid_field_sample)[0]))

@jit(nopython=True)
def find_zero_crossings(X, Y):
	""" Find all the zero crossings: y(x) = 0

Parameters
----------
X : a 1D float array of x-values sorted in ascending order;
	the array may not have identical elements;
Y : a float array of the same shape as X;

Returns
-------
out : an array of x-values where the linearly interpolated function y(x)
has zero values (an empty list if there are no zero crossings).
"""
	Z = Y[:-1] * Y[1:]
	out = []
	for i in np.nonzero(Z <= 0)[0]:
		if Z[i] == 0:
			if Y[i] == 0:
				out.append(X[i])
		else:
			# there is a zero crossing between X[i] and X[i+1]
			out.append((X[i]*Y[i+1] - X[i+1]*Y[i]) / (Y[i+1] - Y[i]))
	return np.array(out)

@jit(nopython=True)
def find_extrema_positions(X, Y):
	""" Find all the extrema of the given function

	Parameters
	----------
	X : a 1D float array of x-values sorted in ascending order;
		the array may not have identical elements;
	Y : a float array of the same shape as X;

	Returns
	-------
	out : an array of x-values where the linearly interpolated y'(x)
	has zero values (an empty list if there are no such x-values).
	"""
	dY_dX = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
	return find_zero_crossings(0.5 * (X[1:] + X[:-1]), dY_dX)


@jit(nopython=True, parallel=True) #fastmath=True causing it to look like E²
def Kernel_jit(t, T, EF, c1, e1, e2, e2n, e2d,a1, a2, b1, b2, b3, b4, p1, dTi, Ip):
	""" return the kernel(t_grid,T_grid) for the a define pulse computed with provided parameters """
	dt=min(np.diff(t))
	dT=min(np.diff(T))
	tau_injection=int(max(np.abs(t)))
	assert dt%dT==0
	n=int(dt//dT)
	N=int(tau_injection//dT)
	nmax=int(t[-1]//dT)
	nmin=int(t[0]//dT)
	tAr=np.arange(-N*dT-2*dT,N*dT+dT, dT, dtype=np.float32)
	OptCycPump=850/5.2917721e-2/137.035999
	# EF=Electric_Field(tAr, lam0=850, I=1.25e14, cep=0, FWHM=OptCycPump)
	VP=-np.cumsum(EF)*dT
	A=np.cumsum(VP)*dT
	A2=np.cumsum(VP**2)*dT
	tAr=tAr[2:]
	EF=EF[2:]
	VP=VP[1:]
	Ti_ar= np.floor_divide(T,dT)#, casting='unsafe', dtype=np.int32)
	f = np.zeros((T.size, t.size), dtype=np.csingle)
	phi = np.zeros((T.size, t.size), dtype=np.float32)
	nskip=N+nmin
	for i in range(T.size):
		for j in range(t.size):
			Ti=Ti_ar[i]
			tj=nskip+j*n
			tp=int(tj+Ti)
			tm=int(tj-Ti)
			if tp>=0 and tm>=0 and tp<EF.size and tm<EF.size:
				f0=EF[tp]*EF[tm]*c1
				if f0!=0:
					VPt = VP[tj]
					xi1 = VP[tp]
					xi2 = VP[tm]
					xi1 = xi1 - xi2
					xi2 = xi1 + 2 * xi2 - 2 * VPt
					CVdiff = A[tp] - A[tm]
					CV2diff = A2[tp] - A2[tm]
					Ti= Ti*dT
					phi[i, j] = (2 * Ti * VPt**2 + CV2diff - 2 * VPt * CVdiff + 4 * Ti * Ip) / 2
					f[i, j] = f0*np.exp(-e1 * xi1 ** 2 - 1j * e2 * xi2 ** 2 * Ti* (Ti- e2n * 1j) / (Ti+ e2d * 1j )) * (a1 - 2 * 1j * Ti* b1 + (-a2 + 2 * 1j * Ti* b2 + Ti**2 * b3) * xi1**2 + Ti**2 * b4 * xi2**2) / (dTi - 1j * Ti) ** p1
					



	return f, phi


def IonRate(t_grid, t_field, input_field, c1, e1, e2, e2n, e2d,a1, a2, b1, b2, b3, b4, p1, dTi, Ip, dT=0.0078125):
	""" return the ionization rate for a define pulse computed with provided parameters """
	
	dt=min(np.diff(t_grid))
	tau_injection=int(max(np.abs(t_field)))
	
	assert dt%dT==0
	n=int(dt//dT)
	N=int(tau_injection//dT)
	nmax=int(t_grid[-1]//dT)
	nmin=int(t_grid[0]//dT)
	
	T_grid=np.unique(np.concatenate((np.arange(0, N//8*dT, dT, dtype=np.float32),np.arange(N//8*dT, N//4*dT, dT*4, dtype=np.float32), np.arange(N//4*dT, N//2*dT, dT*16, dtype=np.float32), np.arange(N//2*dT, N*dT+dT, dt, dtype=np.float32))))
	#T_grid=np.arange(0, N*dT, dT, dtype=np.float32)
	np.testing.assert_array_almost_equal(t_grid,np.arange(nmin*dT,nmax*dT+n*dT, n*dT))
	tAr=np.arange(-N*dT-2*dT,N*dT+dT, dT, dtype=np.float32)
	EF=UnivariateSpline(t_field, input_field, k=5, s=0, ext=1 )(tAr)
	f, phase=Kernel_jit(t_grid, T_grid, EF, c1, e1, e2, e2n, e2d,a1, a2, b1, b2, b3, b4, p1, dTi, Ip)
	return IOF(T_grid, f, phase)


def IonProb(t_field, input_field, c1, e1, e2, e2n, e2d,a1, a2, b1, b2, b3, b4, p1, dTi, Ip, dt=1, PulseN=8):
	""" return the ionization probability for the defined pulse computed with provided parameters """
	ElecField=lambda t: interp1d(t, t_field, input_field)
	t_grid=np.arange(t_field[0],t_field[-1]+1,1) 
	# extr=find_extrema_positions(t_grid, ElecField(t_grid))
	# Fextr=ElecField(extr)
	# if Fextr.size > 0:
	# 	extr=extr[np.abs(Fextr)>max(np.abs(Fextr))/100]
	# else:
	# 	print("Fextr is empty")
	# #extr=extr[np.abs(Fextr)>max(np.abs(Fextr))/100]
	# if extr[0]>0 and extr[-1]<0:
	# 	extr=extr[::-1]
	# zeroCr=find_zero_crossings(t_grid, ElecField(t_grid))
	# if zeroCr[0]>0 and zeroCr[-1]<0:
	# 	zeroCr=zeroCr[::-1]
	# t_grid=np.arange(np.floor(zeroCr[zeroCr<extr[0]][-1]), np.ceil(zeroCr[zeroCr>extr[-1]][0])+dt, dt, dtype=np.float32)
	rate=IonRate(t_grid, t_field, input_field, c1, e1, e2, e2n, e2d,a1, a2, b1, b2, b3, b4, p1, dTi, Ip)
	prob=simpson(y=rate, x=t_grid, axis=-1, even='simpson')
	#prob=simpson(x=t_grid, y=rate)
	return np.double(prob)

def remove_irrelevant_data(time, field, x):
	relevant_start = None
	relevant_end = None
	for i in range(len(field)):
		if field[i] > x:
			relevant_start = i
			break
	for i in range(len(field)-1, -1, -1):
		if field[i] > x:
			if relevant_end is None:
				relevant_end = i
	return time[relevant_start:relevant_end + 1], field[relevant_start:relevant_end + 1]

# gamma = 5.664805631417528
# alpha = 0.7
# time = np.loadtxt("output_interpol.csv", delimiter=',', skiprows=1)[:, 1]
# field = np.loadtxt("output_interpol.csv", delimiter=',', skiprows=1)[:, 2]
# print(time, field)
# p=IonProb(time, field, c1=100, e1=gamma/2*1, e2=1.7064681211148507/36, e2n=30*alpha, e2d=2*alpha, a1=4*gamma*1, a2=4*gamma**2*1, b1=4.977965258900104, b2=2*gamma*1*4.977965258900104, b3=4.941414470108175, b4=-16/9*4.941414470108175, p1=3.36598832010105, dTi=2*gamma*1, Ip=0.50484955022526852)
# print(p)
#IonProb(np.arange(-200,201,1), np.sin(np.linspace(-np.pi, np.pi, 401)), c1=2658.86, e1=2.5, e2=1/36, e2n=21.2353, e2d=1.41569, a1=20., a2=40.0001, b1=1, b2=10., b3=3., b4=-5.33333, p1=4.04918, dTi=10., Ip=1)

#def main():
	#time = np.loadtxt("output_interpol.csv", delimiter=',', skiprows=1)[:, 1]
	#for i in range(2, 35):
		#field = np.loadtxt("output_interpol.csv", delimiter=',', skiprows=1)[:, i]
		#Ionprob = IonProb(time, field, c1=100, e1=gamma/2*1, e2=1.7064681211148507/36, e2n=30*alpha, e2d=2*alpha, a1=4*gamma*1, a2=4*gamma**2*1, b1=4.977965258900104, b2=2*gamma*1*4.977965258900104, b3=4.941414470108175, b4=-16/9*4.941414470108175, p1=3.36598832010105, dTi=2*gamma*1, Ip=0.50484955022526852)
		#print(Ionprob)
# 	IonProb(np.arange(-100,100,1), np.arange(0,100,0.0078125/4), wavel=800, intens=0.1, cep=0, fwhmau=200, c1=1, e1=1, e2=1, e2n=1, e2d=1,a1=1, a2=1, b1=1, b2=1, b3=1, b4=1, p1=1, dTi=1, Ip=1)
#if __name__ == '__main__':
# 	import cProfile, pstats
# 	profiler = cProfile.Profile()
# 	profiler.enable()
	#main()
# 	profiler.disable()
# 	stats = pstats.Stats(profiler).sort_stats('cumtime')
	#stats.print_stats()



	### warum funktioniert remove_irr.. erst ab 1e-9? liegt daran das extr gleich bleibt und dann bei zb 1e-5 
	### in zeroCr kein element kleiner ist als extr[0] 