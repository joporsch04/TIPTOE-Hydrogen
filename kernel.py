# Copyright (c) 2024 Manoram Agarwal
#
# -*- coding:utf-8 -*-
# @Script: kernels.py
# @Author: Manoram Agarwal
# @Email: manoram.agarwal@mpq.mpg.de
# @Create At: 2024-07-19 15:00:20
# @Last Modified By: Manoram Agarwal
# @Last Modified At: 2024-07-21 20:49:56
# @Description: This python file contains the function that provides three different methods to compute the kernel, rates and probabilities


from numba import njit, cuda
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import numpy as np
from field_functions import find_zero_crossings, find_extrema_positions
from field_functions import integrate_oscillating_function_jit as IOF
from field_functions import LaserField 






@cuda.jit(fastmath = True, cache=True)
def Kernel_cuda_helper(f0, phase0, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, EF, EF2,VP, A, A2,dT, n,nskip, Ti_ar):
    """ return the kernel(t_grid,T_grid) for the a define pulse computed with provided parameters 
    Args:
        f0 (np.ndarray, shape=(T_grid.size, t_grid.size)): 2d grid to store pre-exponential
        phase0 (np.ndarray, shape=(T_grid.size, t_grid.size)):2d grid storing complex agument of the exponential function
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift
        EF (np.ndarray): Electric field
        EF2 (np.ndarray): Cummulative Electric field squared
        VP (np.ndarray): Vector potential
        A (np.ndarray): cummulative of the vector potential
        A2 (np.ndarray): cummulative squared of the vector potential
        dT (float64): step size of dense arrays used for EF, EF2 etc
        n (int64): number of steps of dT needed to reach next t[i+1]
        nskip (int64): number of steps to skip initally to get to the first provided t
        Ti_ar (np.ndarray): indices of time array T for which the values of kernel will actually be stored. 

    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel"""
    i, j = cuda.grid(2)
    if i < f0.shape[0] and j < f0.shape[1]:
        Ti=Ti_ar[i]
        tj=nskip+j*n
        tp=tj+Ti
        tm=tj-Ti
        if tp>=0 and tp<EF.size and tm>=0 and tm<EF.size:
            #if EF[tp]!=0 and EF[tm]!=0:
            #f0[i,j]=EF[tp]*EF[tm]*Multiplier
            VPt = VP[tj]
            xi1 = (VP[tp] - VP[tm])**2
            CVdiff = A[tp] - A[tm]
            CV2diff = A2[tp] - A2[tm]
            E2diff=EF2[tp]-EF2[tm]
            Ti= Ti*dT
            xi2 = (VP[tp] + VP[tm] - 2 * VPt)**2 #EFd[tj]*Ti**2 #
            f0[i, j] = EF[tp]*EF[tm]*Multiplier * (a0 - 2 * 1j * Ti* a1 + (-b0 + 2 * 1j * Ti* b1 + Ti**2 * b2) * xi1 + Ti**2 * c2 * xi2) / (1  - d1*1j * Ti )**p1 
            phase0[i, j] =  1j*(2 * Ti * VPt**2 + CV2diff - 2 * VPt * CVdiff + 4 * Ti * Ip + αPol * E2diff) / 2  -e0 * xi1 
                        
def Kernel_cuda(t_grid, T_grid, LaserPulse, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol):
    """ return the kernel(t_grid,T_grid) for the a define pulse computed with provided parameters using a cuda implementation
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        T_grid (np.ndarray): array time T before and after time t that affects that moment of ionization 
        wavel (float64): central wavelength
        I (float64): intensity W/cm2
        cep (float64): cep in radians
        fwhmau (float64): full width half maximum in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift

    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel"""
    t=t_grid
    T=T_grid
    dT=0.0078125/4
    tau_injection=np.ceil(LaserPulse.tau_injection)+4
    N=int(tau_injection//dT)+1
    
    tAr=np.arange(-N*dT-dT,N*dT+dT, dT, dtype=np.float64)
    stream = cuda.stream()
    VP=LaserPulse.Vector_potential(tAr)
    EF=LaserPulse.Electric_Field(tAr)
    A=cuda.to_device(np.cumsum(VP*dT, dtype=np.float64),stream=stream)
    A2=cuda.to_device(np.cumsum(VP**2*dT, dtype=np.float64),stream=stream)
    EF2=cuda.to_device(np.cumsum(EF**2*dT, dtype=np.float64),stream=stream)
    tAr=tAr[1:]
    EF= cuda.to_device(EF[1:],stream=stream)
    VP=cuda.to_device(VP[1:],stream=stream)
    Ti_ar=cuda.to_device(np.array(T//dT, dtype=np.int64),stream=stream)
    
    #if T.size*t_grid.size*16*3<cuda.get_current_device().total_memory:
    t=t_grid
    dt=min(np.diff(t))
    assert dt%dT==0
    n=int(dt//dT)
    nmax=int(t[-1]//dT)
    nmin=int(t[0]//dT)
    f0_device = cuda.to_device(np.zeros((T.size, t.size), dtype=np.cdouble),stream=stream)
    phase0_device = cuda.to_device(np.zeros((T.size, t.size), dtype=np.cdouble),stream=stream)
    threadsperblock = (16, 16)
    blockspergrid_y = (t.size + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_x = (T.size + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    Kernel_cuda_helper[blockspergrid, threadsperblock](f0_device, phase0_device, Multiplier, e0,a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, EF, EF2,VP, A, A2, dT, n,N+nmin,Ti_ar)
    return f0_device.copy_to_host(), phase0_device.copy_to_host()


def Kernel_jit(t_grid, T_grid, LaserPulse, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol):
    """ return the kernel(t_grid,T_grid) for the a define pulse computed with provided parameters using a jit optimized implementation
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        T_grid (np.ndarray): array time T before and after time t that affects that moment of ionization 
        wavel (float64): central wavelength
        I (float64): intensity W/cm2
        cep (float64): cep in radians
        fwhmau (float64): full width half maximum in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift

    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel"""
    t=t_grid
    T=T_grid
    dt=min(np.diff(t))
    dT=min(np.diff(T))
    tau_injection=max(T)
    assert dt%dT==0
    n=int(dt//dT)
    N=int(tau_injection//dT)+1
    nmax=int(t[-1]//dT)
    nmin=int(t[0]//dT)
    tAr=np.arange(-N,N+1, 1)*dT
    VP=LaserPulse.Vector_potential(tAr)
    EF=LaserPulse.Electric_Field(tAr)
    A=LaserPulse.int_A(tAr)
    A2=LaserPulse.int_A2(tAr)
    EF2=LaserPulse.int_E2(tAr)
    Ti_ar=(T//dT).astype(np.int64)
    return Kernel_jit_helper(N, T, t, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, EF, EF2,VP, A, A2,dT, n, nmin, Ti_ar)

@njit(parallel=True, cache=True)
def Kernel_jit_helper(N, T, t, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, EF, EF2,VP, A, A2,dT, n, nmin, Ti_ar):
    f0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    phase0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    for i in range(T.size):
        for j in range(t.size):
            Ti=Ti_ar[i]
            tj=N+nmin+j*n
            tp=tj+Ti
            tm=tj-Ti
            if tp>=0 and tp<EF.size and tm>=0 and tm<EF.size:
                VPt = VP[tj]
                xi1 = (VP[tp] - VP[tm])**2
                xi2 = (VP[tp] + VP[tm] - 2 * VPt )**2
                CVdiff = A[tp] - A[tm]
                CV2diff = A2[tp] - A2[tm]
                E2diff=EF2[tp]-EF2[tm]
                Ti= Ti*dT
                f0[i, j] = EF[tp]*EF[tm]*Multiplier * (a0 - 2 * 1j * Ti* a1 + (-b0 + 2 * 1j * Ti* b1 + Ti**2 * b2) * xi1 + Ti**2 * c2 * xi2) / (1  - d1*1j * Ti )**p1 
                phase0[i, j] =  1j*(2 * Ti * VPt**2 + CV2diff - 2 * VPt * CVdiff + 4 * Ti * Ip + αPol * E2diff) / 2  -e0 * xi1
    return f0, phase0


def Kernel(t_grid, T_grid, LaserPulse, Multiplier, e0 ,a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol):
    """ return the kernel(t_grid,T_grid) for the a define pulse computed with provided parameters
    This method is available for consistancy checks in case something seems amiss with the efficient indexing based method

    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        T_grid (np.ndarray): array time T before and after time t that affects that moment of ionization 
        wavel (float64): central wavelength
        I (float64): intensity W/cm2
        cep (float64): cep in radians
        fwhmau (float64): full width half maximum in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift

    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel
    """
    
    import numexpr as ne
    from os import cpu_count
    from field_functions import int_A, int_A2, interpFuncCumE2
    intens=LaserPulse.I
    ne.set_num_threads(cpu_count())
    print("old method activated")
    VecPot = lambda t: LaserPulse.Vector_potential(t)
    ElecField=lambda t: LaserPulse.Electric_Field(t)
    t_grid=np.asarray(t_grid)
    T_grid=np.asarray(T_grid)
    T,t=np.meshgrid(T_grid, t_grid, indexing="ij",sparse=True, copy=False)
    tp=ne.evaluate("t+T")
    tm=ne.evaluate("t-T")
    f0=ElecField(tp)*ElecField(tm)*Multiplier
    VPt=VecPot(t)
    xi1=VecPot(tp)
    xi2=VecPot(tm)
    xi1=ne.evaluate("xi1-xi2")
    xi2=ne.evaluate("xi1+2*xi2-2*VPt")
    CVdiff=int_A(tp,wavel, intens, cep, fwhmau)-int_A(tm,wavel, intens, cep, fwhmau)
    CV2diff=int_A2(tp,wavel, intens, cep, fwhmau)-int_A2(tm,wavel, intens, cep, fwhmau)
    E2diff=interpFuncCumE2(tp,wavel, intens, cep, fwhmau)-interpFuncCumE2(tm,wavel, intens, cep, fwhmau)
    phase0=ne.evaluate("1j*(2*T*VPt**2+CV2diff-2*VPt*CVdiff+4*T*Ip+αPol*E2diff)/2-e0*xi1**2")
    f0=ne.evaluate("f0*(a0 - 2*1j*T*a1 + (-b0 + 2*1j*T*b1 + T**2*b2)*xi1**2 + T**2*c2*xi2**2)/(1 - d1*1j*T)**p1")
    #phase0=0#np.unwrap(np.angle(f1))
    #f1=abs(f1)
    return f0, phase0 #f0,f1,phase0,phase0


def IonRate(LaserPulse, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, dT, t_grid = None, PulseN=8):
    """ return the ionization rate for a define pulse computed with provided parameters 
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        wavel (float64): central wavelength
        I (float64): intensity W/cm2
        cep (float64): cep in radians
        fwhmau (float64): full width half maximum in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift
        dT (float64): time step make it smaller if computations not converged

    Returns:
        np.ndarray, shape=(t_grid.size): the ionization rates for given time grid"""
    if np.all(t_grid) != None:
        dt=min(np.diff(t_grid))
    else:
        t_grid = LaserPulse.get_time_array()
        dt=min(np.diff(t_grid))
    #tau_injection=np.ceil(np.pi*fwhmau/(4*np.arccos(2**(-1/(2*PulseN)))))+4
    #tau_injection=LaserPulse.tau_injection
    T_min, T_max = LaserPulse.get_time_interval()
    T_grid=np.arange(0, max(abs(T_min), abs(T_max))+dT, dT, dtype=np.float64)
    if cuda.is_available() and T_grid.size*t_grid.size>10000000 and T_grid.size*t_grid.size*16*3<2048*10**6: # use cuda kernel if available and grid is large enough to benifit from it
        f, phase=Kernel_cuda(t_grid, T_grid, LaserPulse, Multiplier, e0,a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol)
    else:
        f, phase=Kernel_jit(t_grid, T_grid, LaserPulse, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol)
    rate=2*np.real(IOF(T_grid, f, phase))
    return rate 




#@njit(parallel=True,fastmath = False, cache=True)
def analyticalRate(LaserPulse, t_grid, Multiplier, e0, a0, a1, b0, b1, b2, p1, d1, Ip, αPol):
    """ return the ionization rate for a define pulse computed with provided parameters 
     
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        wavel (float64): central wavelength
        I (float64): intensity W/cm2
        cep (float64): cep in radians
        fwhmau (float64): full width half maximum in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift
        dT (float64): time step make it smaller if computations not converged

    Returns:
        np.ndarray, shape=(t_grid.size): the quasi-static ionization rates for given time grid """    
    field = np.abs(LaserPulse.Electric_Field(t_grid))
    Stark=αPol/2
    field += 1e-50
    tmp = (1 / (2**(1/4) * field**(5/2) * (8 * e0**2 * field**2 + Ip + field**2 * Stark)**(1/4)) * Multiplier * 
           np.exp(-((4 * (-4 * e0 * field + np.sqrt(2) * np.sqrt(Ip + field**2 * (8 * e0**2 + Stark))) * 
                      (Ip + field**2 * (4 * e0**2 + Stark) - np.sqrt(2) * e0 * field * np.sqrt(Ip + field**2 * (8 * e0**2 + Stark)))) / (3 * field))) *
           np.sqrt(np.pi) * (d1 - 4 * e0 + (np.sqrt(2) * np.sqrt(8 * e0**2 * field**2 + Ip + field**2 * Stark)) / field)**-p1 * 
           (a0 * field**4 + 2 * np.sqrt(2) * field**3 * (a1 - 16 * e0 * (b0 + 4 * e0 * (-3 * b1 + 8 * b2 * e0)) * field**2) * 
            np.sqrt(8 * e0**2 * field**2 + Ip + field**2 * Stark) - 16 * np.sqrt(2) * (-b1 + 8 * b2 * e0) * field**3 * 
            (8 * e0**2 * field**2 + Ip + field**2 * Stark)**(3/2) + 
            8 * (-a1 * e0 * field**4 + b0 * field**4 * (16 * e0**2 * field**2 + Ip + field**2 * Stark) - 
                 8 * b1 * e0 * field**4 * (32 * e0**2 * field**2 + 3 * (Ip + field**2 * Stark)) + 
                 2 * b2 * field**2 * (512 * e0**4 * field**4 + 64 * e0**2 * field**2 * (Ip + field**2 * Stark) + (Ip + field**2 * Stark)**2))))
    return tmp






@njit(parallel=True,fastmath = False, cache=True)
def QSRate(LaserPulse, t_grid, Multiplier, e0, a0, a1, b0, b1, b2, p1, d1, Ip, αPol):
    """ return the ionization rate for a define pulse computed with provided parameters  
    Args:
        field (float64): Electric field strength in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift

    Returns:
        np.ndarray, shape=(t_grid.size): the ionization rates for given time grid"""
    Stark=αPol/2
    field=LaserPulse.Electric_Field(t_grid)
    tmp = (1 / (2**(1/4) * field**(5/2) * (8 * e0**2 * field**2 + Ip + field**2 * Stark)**(1/4)) * Multiplier * 
        np.exp(-((4 * (-4 * e0 * field + np.sqrt(2) * np.sqrt(Ip + field**2 * (8 * e0**2 + Stark))) * 
                    (Ip + field**2 * (4 * e0**2 + Stark) - np.sqrt(2) * e0 * field * np.sqrt(Ip + field**2 * (8 * e0**2 + Stark)))) / (3 * field))) *
        np.sqrt(np.pi) * (d1 - 4 * e0 + (np.sqrt(2) * np.sqrt(8 * e0**2 * field**2 + Ip + field**2 * Stark)) / field)**-p1 * 
        (a0 * field**4 + 2 * np.sqrt(2) * field**3 * (a1 - 16 * e0 * (b0 + 4 * e0 * (-3 * b1 + 8 * b2 * e0)) * field**2) * 
        np.sqrt(8 * e0**2 * field**2 + Ip + field**2 * Stark) - 16 * np.sqrt(2) * (-b1 + 8 * b2 * e0) * field**3 * 
        (8 * e0**2 * field**2 + Ip + field**2 * Stark)**(3/2) + 
        8 * (-a1 * e0 * field**4 + b0 * field**4 * (16 * e0**2 * field**2 + Ip + field**2 * Stark) - 
                8 * b1 * e0 * field**4 * (32 * e0**2 * field**2 + 3 * (Ip + field**2 * Stark)) + 
                2 * b2 * field**2 * (512 * e0**4 * field**4 + 64 * e0**2 * field**2 * (Ip + field**2 * Stark) + (Ip + field**2 * Stark)**2))))
    return tmp



def IonProb(LaserPulse, Multiplier, e0,a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, dt=2, dT=0.25, PulseN=8, filterTrehold=0.01):
    """ return the ionization probability for the defined pulse computed with provided parameters 
    Note: by default this function filters out the t_grid such that |E(t_grid)|>=1% of max(E(t_grid)) 
    Args:
        wavel (float64): central wavelength
        I (float64): intensity W/cm2
        cep (float64): cep in radians
        fwhmau (float64): full width half maximum in atomic units
        Multiplier (float64): overall normalization factor
        e0 (float64): exponential factor responsible for decay wrt xi1**2
        a0 (float64): zeroth order coefficient wrt T
        a1 (float64): first order coefficient wrt T
        Note: the b variables are all multiplied by xi1**2
        b0 (float64): zeroth order coefficient wrt T
        b1 (float64): first order coefficient wrt T
        b2 (float64): second order coefficient wrt T
        c2 (float64): secondf order coefficient wrt T that is multiplied by xi2**2
        p1 (float64): 1/(1-d1*1j*T)**p1  decay
        d1 (float64): 1/(1-d1*1j*T)**p1  decay
        Ip (float64): Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift
        dT (float64): time step make it smaller if computations not converged

    Returns:
        float64: the ionization probability for given pulse"""
        
    #tau_injection=np.ceil(np.pi*fwhmau/(4*np.arccos(2**(-1/(2*PulseN)))))
    #t_grid=np.arange(-tau_injection,tau_injection+dt, dt)

    t_grid = LaserPulse.get_time_array(dt)
    
    ### filter out the t_grid such that |E(t_grid)|>=1% of max(E(t_grid)) ###
    ElecField = LaserPulse.Electric_Field

    extr=find_extrema_positions(t_grid, ElecField(t_grid))
    Fextr=ElecField(extr)
    extr=extr[np.abs(Fextr)>=max(np.abs(Fextr))*filterTrehold]
    if extr[0]>0 and extr[-1]<0:
        extr=extr[::-1]
    ### smartly take the t_grid uptill the correspondin zero crossing rather than abruptly ending a a sub-cycle peak
    zeroCr=find_zero_crossings(t_grid, ElecField(t_grid))
    if zeroCr[0]>0 and zeroCr[-1]<0:
        zeroCr=zeroCr[::-1]
    t_grid=np.arange(np.floor(zeroCr[zeroCr<extr[0]][-1]), np.ceil(zeroCr[zeroCr>extr[-1]][0])+dt, dt, dtype=np.float64)
    rate=IonRate(LaserPulse, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol, dT, t_grid=t_grid)
    return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson')))











        

### if called directly, this file can be used for profiling

if __name__ == '__main__':
    ## the first function call simply ensures that the function is called at least once and numba compilation is done
    IonProb(wavel=100, intens=1e04, cep=0, fwhmau=20, Multiplier=2658.86, e0=2.5, a0=20., b0=40.0001, a1=1, b1=10., b2=3., c2=-5.33333, p1=4.04918, d1=10., Ip=0.5, αPol=0)
    print('Profiling...')
    print("user is adviced to change the dummy parameters (wavel, intens, cep, fwhmau, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol) to see the profiling results")
    
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    IonProb(wavel=400, intens=1e04, cep=0, fwhmau=50, Multiplier=2658.86, e0=2.5, a0=20., b0=40.0001, a1=1, b1=10., b2=3., c2=-5.33333, p1=4.04918, d1=10., Ip=0.5, αPol=0)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()