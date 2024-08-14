# Copyright (c) 2024 Manoram Agarwal
#
# -*- coding:utf-8 -*-
# @Script: kernels.py
# @Author: Manoram Agarwal
# @Email: manoram.agarwal@mpq.mpg.de
# @Create At: 2024-07-19 15:00:20
# @Last Modified By: Manoram Agarwal
# @Last Modified At: 2024-08-08 11:48:47
# @Description: This python file contains the function that provides three different methods to compute the kernel, rates and probabilities


from numba import njit, cuda
from scipy.integrate import simpson
import numpy as np
from field_functions import LaserField, find_zero_crossings, find_extrema_positions
from field_functions import integrate_oscillating_function_jit as IOF

@cuda.jit(fastmath = True, cache=True)
def Kernel_cuda_helper(f0, phase0, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, EF, EF2,VP, A, A2,dT, n,nskip, Ti_ar):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters 
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
        E_g (float64): band gap
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
            phase0[i, j] =  1j*(2 * Ti * VPt**2 + CV2diff - 2 * VPt * CVdiff + 4 * Ti * E_g + αPol * E2diff) / 2  -e0 * xi1 
                        
def Kernel_cuda(t_grid, T_grid, laser_field, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters using a cuda implementation
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        T_grid (np.ndarray): array time T before and after time t that affects that moment of ionization
        laser_field: an object of LaserField
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift


    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel"""
    t=t_grid
    T=T_grid
    dT=0.0078125/4
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=max(abs(t_min), abs(t_max))
    N=int(tau_injection//dT)+1
    
    tAr=np.arange(-N*dT-dT,N*dT+dT, dT, dtype=np.float64)
    stream = cuda.stream()
    VP=laser_field.Vector_potential(tAr)
    EF=laser_field.Electric_Field(tAr)
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
    Kernel_cuda_helper[blockspergrid, threadsperblock](f0_device, phase0_device, Multiplier, e0,a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, EF, EF2,VP, A, A2, dT, n,N+nmin,Ti_ar)
    return f0_device.copy_to_host(), phase0_device.copy_to_host()

########################

@njit(parallel=True, fastmath=False, cache=True)
def Kernel_jit_helper(t, T, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, EF, EF2, VP, A, A2, dT, N, n, nmin, Ti_ar):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters using a jit optimized implementation
    Args:
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift
        EF (np.ndarray): Electric field
        EF2 (np.ndarray): Cummulative Electric field squared
        VP (np.ndarray): Vector potential
        A (np.ndarray): cummulative of the vector potential
        A2 (np.ndarray): cummulative squared of the vector potential
        dT (float64): step size of dense arrays used for EF, EF2 etc
        n (int64): number of steps of dT needed to reach next t[i+1]
        Ti_ar (np.ndarray): indices of time array T for which the values of kernel will actually be stored. 

    Returns:
        f0 (np.ndarray, shape=(T_grid.size, t_grid.size)): 2d grid to store pre-exponential
        phase0 (np.ndarray, shape=(T_grid.size, t_grid.size)):2d grid storing complex agument of the exponential function
    """
    
    f0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    phase0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    for i in range(T.size):
    # for i in range(f0.shape[0]):
        # for j in range(f0.shape[1]):
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
                phase0[i, j] =  1j*(2 * Ti * VPt**2 + CV2diff - 2 * VPt * CVdiff + 4 * Ti * E_g + αPol * E2diff) / 2  -e0 * xi1
    return f0, phase0


def Kernel_jit(t_grid, T_grid, laser_field, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters using a jit optimized implementation
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        T_grid (np.ndarray): array time T before and after time t that affects that moment of ionization 
        laser_field: an object of LaserField
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift
 
    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel"""
    t=t_grid
    T=T_grid
    dt=min(np.diff(t))
    dT=min(np.diff(T))#/2
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=max(abs(t_min), abs(t_max))
    assert dt%dT==0
    n=int(dt//dT)
    N=int(tau_injection//dT)+1
    # nmax=int(t[-1]//dT)
    nmin=int(t[0]//dT)
    tAr=np.arange(-N, N+1, 1) * dT
    VP=laser_field.Vector_potential(tAr)
    EF=laser_field.Electric_Field(tAr)
    A=laser_field.int_A(tAr) # np.cumsum(VP*dT)
    A2=laser_field.int_A2(tAr) # np.cumsum(VP**2*dT)
    EF2=laser_field.int_E2(tAr) # np.cumsum(EF**2*dT)
    Ti_ar=(T//dT).astype(np.int64)
    # f0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    # phase0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    f0, phase0= Kernel_jit_helper(t, T, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, EF, EF2, VP, A, A2, dT, N, n, nmin, Ti_ar)
    return f0, phase0


def IonRate(t_grid, laser_field, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, dT):
    """ return the ionization rate for a define pulse computed with provided parameters 
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        laser_field: an object of LaserField
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift
        dT (float64): time step make it smaller if computations not converged

    Returns:
        np.ndarray, shape=(t_grid.size): the ionization rates for given time grid"""
    dt=min(np.diff(t_grid))
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=max(abs(t_min), abs(t_max))
    T_grid=np.arange(0, tau_injection+dT, dT, dtype=np.float64)
    if cuda.is_available() and T_grid.size*t_grid.size>10000000 and T_grid.size*t_grid.size*16*3<2048*10**6: # use cuda kernel if available and grid is large enough to benifit from it
        f, phase=Kernel_cuda(t_grid, T_grid, laser_field, Multiplier, e0,a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol)
    else:
        f, phase=Kernel_jit(t_grid, T_grid, laser_field, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol)
    rate=2*np.real(IOF(T_grid, f, phase))
    return rate  


#@njit(parallel=True,fastmath = False, cache=True)
def analyticalRate(t_grid, laser_field, Multiplier, e0, a0, a1, b0, b1, b2, p1, d1, E_g, αPol):
    """ return the ionization rate for a define pulse computed with provided parameters 
     
    Args:
        t_grid (np.ndarray): the grid of moment of ionization
        laser_field: an object of LaserField
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift
        dT (float64): time step make it smaller if computations not converged

    Returns:
        np.ndarray, shape=(t_grid.size): the quasi-static ionization rates for given time grid """
    field = np.abs(laser_field.Electric_Field(t_grid))
    Stark=αPol/2
    tmp = (1 / (2**(1/4) * field**(5/2) * (8 * e0**2 * field**2 + E_g + field**2 * Stark)**(1/4)) * Multiplier * 
           np.exp(-((4 * (-4 * e0 * field + np.sqrt(2) * np.sqrt(E_g + field**2 * (8 * e0**2 + Stark))) * 
                      (E_g + field**2 * (4 * e0**2 + Stark) - np.sqrt(2) * e0 * field * np.sqrt(E_g + field**2 * (8 * e0**2 + Stark)))) / (3 * field))) *
           np.sqrt(np.pi) * (1- 4 *d1 * e0 + (np.sqrt(2) *d1 * np.sqrt(8 * e0**2 * field**2 + E_g + field**2 * Stark)) / field)**-p1 * 
           (a0 * field**4 + 2 * np.sqrt(2) * field**3 * (a1 - 16 * e0 * (b0 + 4 * e0 * (-3 * b1 + 8 * b2 * e0)) * field**2) * 
            np.sqrt(8 * e0**2 * field**2 + E_g + field**2 * Stark) - 16 * np.sqrt(2) * (-b1 + 8 * b2 * e0) * field**3 * 
            (8 * e0**2 * field**2 + E_g + field**2 * Stark)**(3/2) + 
            8 * (-a1 * e0 * field**4 + b0 * field**4 * (16 * e0**2 * field**2 + E_g + field**2 * Stark) - 
                 8 * b1 * e0 * field**4 * (32 * e0**2 * field**2 + 3 * (E_g + field**2 * Stark)) + 
                 2 * b2 * field**2 * (512 * e0**4 * field**4 + 64 * e0**2 * field**2 * (E_g + field**2 * Stark) + (E_g + field**2 * Stark)**2))))
    return np.nan_to_num(tmp)


#@njit(parallel=True,fastmath = False, cache=True)
def QSRate(field, Multiplier, e0, a0, a1, b0, b1, b2, p1, d1, E_g, αPol):
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift

    Returns:
        np.ndarray, shape=(t_grid.size): the ionization rates for given time grid"""
    Stark=αPol/2
    tmp = (1 / (2**(1/4) * field**(5/2) * (8 * e0**2 * field**2 + E_g + field**2 * Stark)**(1/4)) * Multiplier * 
           np.exp(-((4 * (-4 * e0 * field + np.sqrt(2) * np.sqrt(E_g + field**2 * (8 * e0**2 + Stark))) * 
                      (E_g + field**2 * (4 * e0**2 + Stark) - np.sqrt(2) * e0 * field * np.sqrt(E_g + field**2 * (8 * e0**2 + Stark)))) / (3 * field))) *
           np.sqrt(np.pi) * (1- 4 *d1 * e0 + (np.sqrt(2) *d1 * np.sqrt(8 * e0**2 * field**2 + E_g + field**2 * Stark)) / field)**-p1 * 
           (a0 * field**4 + 2 * np.sqrt(2) * field**3 * (a1 - 16 * e0 * (b0 + 4 * e0 * (-3 * b1 + 8 * b2 * e0)) * field**2) * 
            np.sqrt(8 * e0**2 * field**2 + E_g + field**2 * Stark) - 16 * np.sqrt(2) * (-b1 + 8 * b2 * e0) * field**3 * 
            (8 * e0**2 * field**2 + E_g + field**2 * Stark)**(3/2) + 
            8 * (-a1 * e0 * field**4 + b0 * field**4 * (16 * e0**2 * field**2 + E_g + field**2 * Stark) - 
                 8 * b1 * e0 * field**4 * (32 * e0**2 * field**2 + 3 * (E_g + field**2 * Stark)) + 
                 2 * b2 * field**2 * (512 * e0**4 * field**4 + 64 * e0**2 * field**2 * (E_g + field**2 * Stark) + (E_g + field**2 * Stark)**2))))
    return tmp


def IonProb(laser_field, Multiplier, e0,a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, dt=2, dT=0.25, filterTreshold=0.0):
    """ return the ionization probability for the defined pulse computed with provided parameters 
    Note: by default this function filters out the t_grid such that |E(t_grid)|>=1% of max(E(t_grid)) 
    Args:
        laser_field: an object of LaserField
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
        E_g (float64): band gap
        αPol (float64): Polarization computed analytically to account for stark shift
        dT (float64): time step make it smaller if computations not converged

    Returns:
        float64: the ionization probability for given pulse"""
        
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=max(abs(t_min), abs(t_max))
    t_grid=np.arange(-tau_injection,tau_injection+dt, dt)
    
    if filterTreshold > 0:
        ### filter out the t_grid such that |E(t_grid)|>=1% of max(E(t_grid)) ###
        ElecField=lambda t: laser_field.Electric_Field(t)
        extr=find_extrema_positions(t_grid, ElecField(t_grid))
        Fextr=ElecField(extr)
        extr=extr[np.abs(Fextr)>=max(np.abs(Fextr))*filterTreshold]
        if extr[0]>0 and extr[-1]<0:
            extr=extr[::-1]
        ### smartly take the t_grid uptill the correspondin zero crossing rather than abruptly ending a a sub-cycle peak
        zeroCr=find_zero_crossings(t_grid, ElecField(t_grid))
        if zeroCr[0]>0 and zeroCr[-1]<0:
            zeroCr=zeroCr[::-1]
        t_grid=np.arange(np.floor(zeroCr[zeroCr<extr[0]][-1]), np.ceil(zeroCr[zeroCr>extr[-1]][0])+dt, dt, dtype=np.float64)
    rate=IonRate(t_grid, laser_field, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol, dT)
    # return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson')))
    return np.double(simpson(rate, x=t_grid, axis=-1, even='simpson'))


### if called directly, this file can be used for profiling

if __name__ == '__main__':
    ## the first function call simply ensures that the function is called at least once and numba compilation is done
    laser_field = LaserField()
    laser_field.add_pulse(central_wavelength=100, peak_intensity=1e04, CEP=0, FWHM=20.)
    IonProb(laser_field, Multiplier=2658.86, e0=2.5, a0=20., b0=40.0001, a1=1, b1=10., b2=3., c2=-5.33333, p1=4.04918, d1=10., E_g=0.5, αPol=0)
    print('Profiling...')
    print("user is adviced to change the dummy parameters (laser_field, Multiplier, e0, a0, a1, b0, b1, b2, c2, p1, d1, E_g, αPol) to see the profiling results")
    
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    laser_field.reset()
    laser_field.add_pulse(central_wavelength=400, peak_intensity=1e04, CEP=0, FWHM=50.)
    IonProb(laser_field, Multiplier=2658.86, e0=2.5, a0=20., b0=40.0001, a1=1, b1=10., b2=3., c2=-5.33333, p1=4.04918, d1=10., E_g=0.5, αPol=0)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()