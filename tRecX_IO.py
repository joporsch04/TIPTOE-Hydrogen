#!/usr/bin/python3.8.8

import os
from posixpath import basename
import sys
import pandas as pd
import numpy as np
from scipy import constants as const
import ast
from __init__ import AtomicUnits as AU


pd.set_option('display.float_format', lambda x: "%.2E" % x )
c_v=const.c

def tRecX_write_inp(csv_file, Objectives=[], single_shot=False):
    mainDir=os.getcwd()
    tRecX_df=pd.read_csv(os.path.join(mainDir,csv_file))
    newline ='\n'
    phase={'cos':'0', 'sin':'pi/2'}
    for i in range(tRecX_df.shape[0]):
        data=tRecX_df.loc[i]
        min_sample_points=np.ceil((2.3*max(data["omega_injection_max"],data["omega_drive_max"])/2/np.pi)*(2*data['tau_drive']))
        if  data['N_drive']<=min_sample_points:
            print("Nyquist condition not fullfilled. \n minimum sampling points = ",min_sample_points, f"current sampling points: {data['N_drive']}")
            #data['N_drive']=int(min_sample_points//2*2+1)
        else:
            print("Nyquist condition fullfilled. \n",data['N_drive'],">",min_sample_points,"=min samp points")
        
        d1 =data['tau_drive'] +1.2*data['tau_injection'] /data['N_injection']
        d2 =data['tau_drive'] +1.2*data['tau_injection']
        print(d1,'d1')
        d3=d2#+3*data['tau_injection'] /data['N_injection']
        if data['N_injection']==0:
                delay_array=np.unique(np.round(np.concatenate((np.linspace(-data['tau_drive'], data['tau_drive'], data['N_drive']),[0]))/AU.second,2))
        else:
            delay_array=np.unique(np.round(np.concatenate((np.linspace(-d2, -d1, data['N_injection']),np.linspace(-data['tau_drive'], data['tau_drive'], data['N_drive']),[0], np.linspace(d1, d3, data['N_injection'])))/AU.second,2))
        #delay_array=np.linspace(-6,6,121)*1e-15/AU.second
        #delay_array=np.linspace(-3*data['drive_FWHM_relative'], 3*data['drive_FWHM_relative'], data['N_drive'])
        if data['convergence_test_bool']==True:
            print('duration of the sampling pulse: ', data['tau_injection'])
            delay_array=np.linspace(-data['drive_FWHM'], data['drive_FWHM'], data['N_drive']//8*2+1)#np.array([-0.4,0.4])
            if single_shot:
                delay_array=np.array([0])
            print(data['File_dir'], ': points the current is sampled at: ', delay_array, 'OptCyc')
            inp_name='/inp'
        elif data['convergence_test_bool']=="one_shot":
            delay_array=np.array([0])
            inp_name='.inp'
        else:
            inp_name='/inp'
            print('#injection field samples:', data['N_injection'], ' #drive samples outside this regime: ',data['N_drive'])
            print("delay Aray (every 5th elem) : ", delay_array[::5])
        for j in range(len(delay_array)):
                delay=delay_array[j]
                del_cycle=delay 
                input_file = f'''Title: 
Photoconductive Sampling - 
Setup using-                    {os.path.join(mainDir,csv_file)}- row-{i}
code initialisation-            tRecX {data['File_dir']}/{j}/inpc       
{''.join(f'{newline}            [Objective]  {Objective}' for Objective in Objectives)}

#define fs e-15 s

Operator: hamiltonian='1/2<<Laplacian>>-<1><1><trunc[{data['lower_trunc']},{data['upper_trunk']}]/Q>'
Operator: interaction='iLaserAz[t]<<D/DZ>>+iLaserAx[t]<<D/DX>>+iLaserAy[t]<<D/DY>>'
{f'Operator: expectationValue={data["expecVal_inside"]}' if data['expecVal_inside']!="none" else ''}

Spectrum:      radialPoints={data['R_density_Spectrum']}, {f"plot={data['Plot_type']}":>10}
Spectrum: expectationValues='{data['expecVals_outside']}'
Spectrum: iSurff=true
Spectrum: maxEnergy={data['spec_cut']} eV
Spectrum: minEnergy={data['spec_cut_min']} eV



Axis:   {'name':>10},{'nCoefficients':>13},{'lower end':>13},{'upper end':>10},{'functions':>21},{'order':>10}
        {'Phi':>10},{data['phi_coeff']:>13}
        {'Eta':>10},{data['eta_coeff']:>13},{'-1':>13},{'1':>10},{'assocLegendre{Phi}':>21}
        {'Rn':>10},{int(data['boundary_au']/data['grid_spacing']*data['order']):>13},{'0':>13},{data['boundary_au']:>10},{'polynomial':>21},{data['order']:>10}
        {'Rn':>10},{data['Rn_coeff_outer']:>13},{data['boundary_au']:>13},{'Infty':>10},{f'polExp[{data["alpha"]}]':>21}

Absorption:              {'kind':>7},{' axis':>13},{' theta':>10},{' upper':>10}
                        {'ECS':>7},{'Rn':>13},{data['ecs_coeff']:>10},{data['boundary_au']:>10}
        
Laser:  {'shape':>10},{'I(W/cm2)':>13},{'FWHM':>13},{'lambda(nm)':>10},{'polarAngle':>12},{'azimuthAngle':>13},{'phiCEO':>10},{'peak':>24}
        {data['injection_shape']:>10},{f"{data['injection_Intensity_Wcm2']:.2e}":>13},{f"{data['injection_FWHM']:.2f} OptCyc":>13},{data['injection_wavelength_nm']:>10},{f"{data['injection_polar_angle']:02d}":>12},{f"{data['injection_azimuthal_angle']:02d}":>13},{phase[data['injection_CEPhase']]:>10},{f"{data['injection_peak']:.2f} OptCyc":>24}
        {data['drive_shape']:>10},{f"{data['drive_Intensity_Wcm2']:.2e}":>13},{f"{data['drive_FWHM']:.2f}{data['drive_unit']}":>13},{data['drive_wavelength_nm']:>10},{data['drive_polar_angle']:>12},{data['drive_azimuthal_angle']:>13},{phase[data['drive_CEPhase']]:>10},{f"{delay:.2f} au":>24}
        
TimePropagation:  {'cutEnergy':>20},{'accuracy':>10}, {'fixStep':>10}
                  {data['cutEnergy']:>20},{data['accuracy']:>10}, {data['fixStep']:>10}

# cutEnergy ...impose a spectral cut on the Hamiltonian
                    this essential to speed up up the time-propagation
# store ...we save the surface values and derivatives at a fine time grid (au)'''
                if data['convergence_test_bool']=="one_shot":
                    base=os.path.basename(data['File_dir'])
                    dire=os.path.dirname(data['File_dir'])
                    os.makedirs(dire, exist_ok=True)
                    filename=os.path.join(dire,base+'.inp')
                else:
                    inp_name='inpc'
                    os.makedirs(os.path.join(data['File_dir'],str(j)), exist_ok=True)
                    filename=os.path.join(data['File_dir'],str(j),inp_name)
                with open(filename, "w") as f:
                    f.write(input_file)
        input_file = f'''Title: 
Background - effects of drive field independent of the injection pulse
Setup using-                    {os.path.join(mainDir,csv_file)}- row-{i}
code initialisation-            tRecX {data['File_dir']}/{j}/inpc       
{''.join(f'{newline}            [Objective]  {Objective}' for Objective in Objectives)}

#define fs e-15 s

Operator: hamiltonian='1/2<<Laplacian>>-<1><1><trunc[{data['lower_trunc']},{data['upper_trunk']}]/Q>'
Operator: interaction='iLaserAz[t]<<D/DZ>>+iLaserAx[t]<<D/DX>>+iLaserAy[t]<<D/DY>>'
{f'Operator: expectationValue={data["expecVal_inside"]}' if data['expecVal_inside']!="none" else ''}

Spectrum:      radialPoints={data['R_density_Spectrum']}, {f"plot={data['Plot_type']}":>10}
Spectrum: expectationValues='{data['expecVals_outside']}'
Spectrum: iSurff=true
Spectrum: maxEnergy={data['spec_cut']} eV


Axis:   {'name':>10},{'nCoefficients':>13},{'lower end':>13},{'upper end':>10},{'functions':>21},{'order':>10}
        {'Phi':>10},{data['phi_coeff']:>13}
        {'Eta':>10},{data['eta_coeff']:>13},{'-1':>13},{'1':>10},{'assocLegendre{Phi}':>21}
        {'Rn':>10},{int(data['boundary_au']/data['grid_spacing']*data['order']):>13},{'0':>13},{data['boundary_au']:>10},{'polynomial':>21},{data['order']:>10}
        {'Rn':>10},{data['Rn_coeff_outer']:>13},{data['boundary_au']:>13},{'Infty':>10},{f'polExp[{data["alpha"]}]':>21}

Absorption:              {'kind':>7},{' axis':>13},{' theta':>10},{' upper':>10}
                        {'ECS':>7},{'Rn':>13},{data['ecs_coeff']:>10},{data['boundary_au']:>10}
        
Laser:  {'shape':>10},{'I(W/cm2)':>13},{'FWHM':>13},{'lambda(nm)':>10},{'polarAngle':>10}, {'azimuthAngle':>10},{'phiCEO':>10},{'peak':>24}
        {data['drive_shape']:>10},{f"{data['drive_Intensity_Wcm2']:.2e}":>13},{f"{data['drive_FWHM']:.2e}{data['drive_unit']}":>13},{data['drive_wavelength_nm']:>10},{data['drive_polar_angle']:>10},{data['drive_polar_angle']:>10},{phase[data['drive_CEPhase']]:>10},{f"{delay:.2f} au":>24}
        
TimePropagation:  {'end':>15},{'print':>10},{'store':>10},{'cutEnergy':>20},{'accuracy':>10}, {'fixStep':>10}
                  {f"{data['end_sim']} OptCyc":>10},{'1 OptCyc':>10},{f"{data['store']}{data['store']}":>10},{data['cutEnergy']:>20},{data['accuracy']:>10}, {data['fixStep']:>10}

# NOTE: This run is to be used to check for current when no injection field along Z direction is present'''
        input_file_injection = f'''Title: 
Effects of injection pulse
Setup using-                    {os.path.join(mainDir,csv_file)}- row-{i}
code initialisation-            tRecX {data['File_dir']}/{j}/inpc       
{''.join(f'{newline}            [Objective]  {Objective}' for Objective in Objectives)}

Operator: hamiltonian='1/2<<Laplacian>>-<1><1><trunc[{data['lower_trunc']},{data['upper_trunk']}]/Q>'
Operator: interaction='iLaserAz[t]<<D/DZ>>+iLaserAx[t]<<D/DX>>+iLaserAy[t]<<D/DY>>'
{f'Operator: expectationValue={data["expecVal_inside"]}' if data['expecVal_inside']!="none" else ''}

Spectrum:      radialPoints={data['R_density_Spectrum']}, {f"plot={data['Plot_type']}":>10}
Spectrum: expectationValues='{data['expecVals_outside']}'
Spectrum: iSurff=true
Spectrum: maxEnergy={data['spec_cut']} eV


Axis:   {'name':>10},{'nCoefficients':>13},{'lower end':>13},{'upper end':>10},{'functions':>21},{'order':>10}
        {'Phi':>10},{data['phi_coeff']:>13}
        {'Eta':>10},{data['eta_coeff']:>13},{'-1':>13},{'1':>10},{'assocLegendre{Phi}':>21}
        {'Rn':>10},{int(data['boundary_au']/data['grid_spacing']*data['order']):>13},{'0':>13},{data['boundary_au']:>10},{'polynomial':>21},{data['order']:>10}
        {'Rn':>10},{data['Rn_coeff_outer']:>13},{data['boundary_au']:>13},{'Infty':>10},{f'polExp[{data["alpha"]}]':>21}

Absorption:              {'kind':>7},{' axis':>13},{' theta':>10},{' upper':>10}
                        {'ECS':>7},{'Rn':>13},{data['ecs_coeff']:>10},{data['boundary_au']:>10}
        
Laser:  {'shape':>10},{'I(W/cm2)':>13},{'FWHM':>13},{'lambda(nm)':>10},{'polarAngle':>10}, {'azimuthAngle':>10},{'phiCEO':>10},{'peak':>24}
        {data['injection_shape']:>10},{f"{data['injection_Intensity_Wcm2']:.2e}":>13},{f"{data['injection_FWHM']:.2f} OptCyc":>13},{data['injection_wavelength_nm']:>10},{f"{data['injection_polar_angle']:02d}":>10},{f"{data['injection_azimuthal_angle']:02d}":>10},{phase[data['injection_CEPhase']]:>10},{f"{data['injection_peak']} OptCyc":>24}
        
        
TimePropagation:  {'cutEnergy':>20},{'accuracy':>10}, {'fixStep':>10}
                  {data['cutEnergy']:>20},{data['accuracy']:>10}, {data['fixStep']:>10}

# NOTE: This run is to be used to check for current when no injection field along Z direction is present'''

        if data['convergence_test_bool']!=True:
            inp_name='inpc'
            os.makedirs(os.path.join(data['File_dir'],str(j+1)), exist_ok=True)
            filename=os.path.join(data['File_dir'] ,str(j+1),inp_name)
            with open(filename, "w") as f:
                f.write(input_file_injection)    

def Script_GNU_parallel_convergence(data, useTime):
    mainDir=os.getcwd()
    tRecX_exec=os.path.join(os.environ['HOME'],'tRecX-develop','build','tRecX')
    base=data['File_dir']
    node=1
    useNodes=""
    dependency=""
    newline ='\n'
    name=os.path.basename(base)
    #job_name=name.split('=')
    #job_name=job_name[0][:3]+job_name[1]
    array_max=len(next(os.walk(base))[1])
    #task=min(40,array_max)#data['phi_coeff']
    core_per_task=40//array_max
    ntasks=core_per_task*array_max
    if core_per_task==0:
        core_per_task=1
    useTim=useTime
    useTim=f'{useTim:02d}'
    batch=f'''#!/bin/bash                                \
        \n#SBATCH -o {base}_log                          \
        \n#SBATCH -e {base}_err                          \
        \n#SBATCH -D {mainDir}                           \
        \n#SBATCH -J {name}                          \
        \n#SBATCH --get-user-env                         \
        \n#SBATCH --mail-type=all                        \
        \n#SBATCH --nodes={node}                         \
        \n#SBATCH --ntasks-per-node={ntasks}             \
        \n#SBATCH --ntasks-per-core=1                    \
        \n#SBATCH --time={useTim}:30:00                  \
        \n#SBATCH --mem-per-cpu=2000                     \
        \n#SBATCH --exclusive                            \
        {f"{newline }# designate compute nodes {newline} #SBATCH --nodelist={useNodes}" if useNodes!="" else ""}     

        \nmodule load parallel            \
        \nsrun="srun -n{core_per_task} -N1" \
        \nparallel="parallel -N 1 --delay .2 -j {array_max} --joblog {base}_parallel_joblog"\
        \n$parallel "$srun {tRecX_exec} {base}/{{1}}/inpc" ::: {{0..{array_max-1}}}
        \n
        ''' 
    return batch    

def Script_GNU_parallel(data, useTime,memory):
    mainDir=os.getcwd()
    tRecX_exec=os.path.join(os.environ['HOME'],'tRecX-develop','build-single','tRecX')
    base=data['File_dir']
    useNodes=""
    dependency=""
    newline ='\n'
    name=os.path.basename(base)
    array_max=len(next(os.walk(base))[1])
    task=min(40,array_max)#data['phi_coeff']
    node=1#array_max//task
    assert array_max%task==0
    useTim=useTime
    mem=task*2000
    useTim=f'{useTim:02d}'
    batch=f'''#!/bin/bash                                \
        \n#SBATCH -o {base}_log                          \
        \n#SBATCH -e {base}_err                          \
        \n#SBATCH -D {mainDir}                           \
        \n#SBATCH -J {name}                              \
        \n#SBATCH --get-user-env                         \
        \n#SBATCH --mail-type=all                        \
        \n#SBATCH --nodes={node}                         \
        \n#SBATCH --ntasks-per-node={task}               \
        \n#SBATCH --time={useTim}:30:00                  \
        \n#SBATCH --mem-per-cpu={int(memory/40)}         \
        \n#SBATCH --cpus-per-task=1                      \
        \n#SBATCH --exclusive                            \
        {f"{newline }# designate compute nodes {newline} #SBATCH --nodelist={useNodes}" if useNodes!="" else ""}     

        \nmodule load parallel            \
        \nsrun="srun  -n1 -N1 -c1 --exclusive" \
        \nparallel="parallel -N 1 --delay .2 -j $SLURM_NTASKS --joblog {base}_parallel_joblog"\
        \n$parallel "$srun {tRecX_exec} {base}/{{1}}/inpc" ::: {{0..{array_max-1}}}
        \n
        ''' 
    return batch    

def Script_array(data, useTime):
    mainDir=os.getcwd()
    tRecX_exec=os.path.join(os.environ['HOME'],'tRecX-develop','build','tRecX')
    base=data['File_dir']
    node=1
    useNodes=""
    dependency=""
    newline ='\n'
    name=os.path.basename(base)
    array_max=len(next(os.walk(base))[1])
    task=4
    useTim=useTime
    mem=task*1000
    useTim=f'{useTim:02d}'
    batch=f'''#!/bin/bash                        \
    \n#SBATCH -o {base}/%a/log                       \
    \n#SBATCH -e {base}/%a_err                       \
    \n#SBATCH -D {mainDir}                           \
    \n#SBATCH -J {job_name}                          \
    \n#SBATCH --get-user-env                         \
    \n#SBATCH --mail-type=all                        \
    \n#SBATCH --mem={mem}                            \
    \n#SBATCH --nodes={node}                         \
    \n#SBATCH --ntasks-per-node={task}               \
    \n#SBATCH --time={useTim}:20:00                  \
    \n#SBATCH --array=0-{array_max-1}                \
    \n{dependency}\
    \n{f"{newline }# designate compute nodes {newline} #SBATCH --nodelist={useNodes}" if useNodes!="" else ""}                 \
    \nsrun {tRecX_exec} {base}/$SLURM_ARRAY_TASK_ID/inpc    \
            ''' 
    return batch

def row_wise_submit(csv_file, submit_max_from_input=False, use_gnu_par=False, single_shot=False, maxTime=2, memory=85000):
    mainDir=os.getcwd()
    tRecX_df=pd.read_csv(os.path.join(mainDir,csv_file))

    if not submit_max_from_input: iter_max=tRecX_df.shape[0]
    else: iter_max=submit_max_from_input
    for i in range(iter_max):
        data=tRecX_df.loc[i]
        base=data['File_dir']
        base=data['File_dir']
        name=os.path.basename(base)
        array_max=len(next(os.walk(base))[1])
        if use_gnu_par==True and not data['convergence_test_bool']==True:
            BATCH_SCRIPT=Script_GNU_parallel(data, useTime=maxTime,memory=memory)
            with open(base+'_script_GNU', "w") as f:
                f.write(BATCH_SCRIPT)
            print(os.system('sbatch '+base+'_script_GNU'))
        elif use_gnu_par==True and data['convergence_test_bool']==True:
            BATCH_SCRIPT=Script_GNU_parallel_convergence(data, useTime=maxTime)
            with open(base+'_script_GNU_convergence', "w") as f:
                f.write(BATCH_SCRIPT)
            print(os.system('sbatch '+base+'_script_GNU_convergence'))
        elif 'HOST' not in list(os.environ.keys()):
            tRecX_exec=os.path.join(os.environ['HOME'],'build','tRecX-develop','build','tRecX')
            array_max=len(next(os.walk(base))[1])
            for j in range(array_max):
                print(os.system(f'{tRecX_exec} {base}/{str(j)}/inpc > {base}/{str(j)}/log &'))
        else: 
            with open(base+'_script', "w") as f:
                BATCH_SCRIPT=Script_array(data, useTime=maxTime)
                f.write(BATCH_SCRIPT)
            print(os.system('sbatch '+base+'_script'))
        #if type()message=message.split(' ')[-1]
        #if (i+1)%2==0:
        #    dependency=f"sbatch --dependency=afterok:{message}"      


def row_wise_submit_local(csv_file, submit_max_from_input=False, ncores=4):
    mainDir=os.getcwd()
    tRecX_df=pd.read_csv(os.path.join(mainDir,csv_file))
    tRecX_exec=os.path.join(os.environ['HOME'],'tRecX','tRecX')
    if not submit_max_from_input: iter_max=tRecX_df.shape[0]
    else: iter_max=submit_max_from_input
    for i in range(iter_max):
        data=tRecX_df.loc[i]
        base=data['File_dir']
        if data['convergence_test_bool']=='one_shot':
            print('Modify submit_script in the main directory to submit the input files')
        elif 'HOST' not in list(os.environ.keys()):
            array_max=len(next(os.walk(base))[1])
            for j in range(array_max):
                print(os.system(f"mpirun -n {ncores} {tRecX_exec} {base}/{j}/inpc"))

        #if type()message=message.split(' ')[-1]
        #if (i+1)%2==0:
        #    dependency=f"sbatch --dependency=afterok:{message}"      
    return 0


class tRecX_sim:
    def __init__(self, name= 'test'):
        self.name = name
        self.Hamiltonian='1/2<<Laplacian>>-<1><1><trunc[10,20]/Q>'
        self.Interaction='iLaserAz[t]<<D/DZ>>+iLaserAx[t]<<D/DX>>+iLaserAy[t]<<D/DY>>'
        self.expecVal_inside="none"
        self.expecVals_outside='<1><1><GridWeight>, <1><Q><GridWeight*Q>, <cos(Q)><sqrt(1-Q*Q)><GridWeight*Q>, <sin(Q)><sqrt(1-Q*Q)><GridWeight*Q>'
        self.order=13
        self.numProcs=40
        #self.Rn_coeff=48
        self.grid_spacing=10
        self.phi_coeff=5
        self.eta_coeff=29
        self.injection_shape='cos8'
        self.injection_Intensity_Wcm2=2e14
        self.injection_FWHM=1#2.2e-15
        self.injection_wavelength_nm=750
        self.injection_polar_angle=0
        self.injection_azimuthal_angle=0
        self.injection_CEPhase='cos'
        self.injection_peak=0
        self.drive_shape='cos8'
        self.drive_Intensity_Wcm2=1e10
        self.drive_FWHM_OptCyc=0.7
        self.drive_FWHM_fs='none'
        self.drive_wavelength_nm=150
        self.drive_peak=0.
        self.drive_polar_angle=0
        self.drive_azimuthal_angle=0
        self.drive_CEPhase='sin'
        self.R_density_Spectrum=100
        self.Plot_type='total'
        self.ellipticity_injection=0
        self.accuracy= 1e-10
        self.cutEnergy=100
        self.convergence_test_bool=True
        self.Rn_coeff_outer=30
        self.fixStep=0.025
        self.lower_trunc=10
        self.upper_trunk=20
        self.boundary_au=20
        self.store=0.01
        self.spec_cut=128
        self.spec_cut_min=0
        self.ecs_coeff=0.3
        self.number_orbitals=0
        self.alpha=0.5

    def calc(self):
        #self.boundary_au=self.upper_trunk
        FWHM_inj=self.injection_FWHM*(np.array(self.injection_wavelength_nm)*1e-9)/const.c
        if self.drive_FWHM_fs!='none':
            self.drive_FWHM=FWHM_drive=self.drive_FWHM_fs
            self.drive_unit='fs'
        else: 
            FWHM_drive=self.drive_FWHM_OptCyc*(np.array(self.drive_wavelength_nm)*1e-9)/const.c
            self.drive_FWHM=self.drive_FWHM_fs=np.round(FWHM_drive*1e15,2)
            self.drive_unit='fs'
        self.Rn_coeff=np.round(np.array(self.upper_trunk)*np.array(self.order)/np.array(self.grid_spacing),0)
        self.tau_injection = np.round(np.pi / (4 * np.arccos(2**-0.0625)) * np.array(FWHM_inj),21)
        self.tau_drive = np.round(np.pi / (4 * np.arccos(2**-0.0625)) * np.array(FWHM_drive),21)
        
        # decide on delays
        self.omega_injection_max= 2*np.pi*c_v/(np.array(self.injection_wavelength_nm)*1e-9)+2*np.pi*.7/(FWHM_inj)
        self.omega_drive_max=2*np.pi*c_v/(np.array(self.drive_wavelength_nm)*1e-9)+2*np.pi/FWHM_drive 
        #omega_drive_max*=1.5
        #omega_drive_max=np.ceil(omega_drive_max) 
        self.N_drive = 1 + 2 *(np.ceil(self.tau_drive / np.pi * self.omega_drive_max)).astype(int)
        self.N_injection = (np.ceil(self.tau_injection / np.pi* self.omega_injection_max)).astype(int)
        # rem=(self.N_drive+2*self.N_injection+1)%self.numProcs
        # for index, remainder in enumerate(rem): 
        #     self.N_drive[index]+=40-remainder
        # self.d1 = self.tau_drive + self.tau_injection / self.N_injection
        # self.d2 = self.tau_drive + self.tau_injection
        # define apropriate simulation begining and end timings 
        #self.begin_sim=-(self.d2+self.tau_injection)*1.1
        self.OptCyc= np.round(np.array(self.injection_wavelength_nm)*1e-9/c_v ,21)
        self.end_sim=np.ceil((self.tau_drive+2*self.tau_injection)/self.OptCyc)
        #### the following parameters are calculated while writing the input file ###
        #fs=1e15
        #self.delay_start1=-d2*fs
        #self.delay_stop1=-d1*fs
        #self.delay_Nstep1=N_injection
        #self.delay_start2=-self.tau_drive*fs
        #self.delay_stop2=self.tau_drive*fs
        #self.delay_Nstep2=N_drive
        #self.delay_start3=d1*fs
        #self.delay_stop3=d2*fs
        #self.delay_Nstep3=N_injection
        #self.injection_peak_fs_arr= np.concatenate((np.linspace({-d2*fs}, {-d1*fs}, {N_injection}),np.linspace({-self.tau_drive*fs}, {self.tau_drive*fs}, {N_drive}), np.linspace({d1*fs}, {d2*fs}, {N_injection}))
        
        for index, key in enumerate(vars(self)):
            break
            if key=="grid_spacing":
                print("the following valriables are calculated based on user input :")
            print(f'{index:>3}. {key:<25}=   {f"{vars(self)[key]}":<40}')#, end=f'{" ":>5}')
    def csv_populate_self(self):
        name=self.name
        mainDir=os.getcwd()
        df=pd.DataFrame.from_dict(vars(self))
        nunique= df.nunique()
        columns = nunique[nunique!=1].index
        diff = df.filter(columns)
        filesubdir=[]
        for i in range(diff.shape[0]):
            data=diff.loc[i]
            name_sub=''
            # j=0
            for key in diff:
                if isinstance(data[key], int) and data[key]<10000: val=int(data[key])
                elif isinstance(data[key], float): 
                    if int(data[key])==data[key] and data[key]<10000: val=int(data[key])
                    elif data[key]<10000 and data[key]>0.001: val=f'{data[key]}'
                    else: val=f'{data[key]:.2e}'
                elif isinstance(data[key], str): val="diff"#data[key][0:6].replace("\'","")
                else: val=data[key]       
                temp_name=""
                for key_sub in key.split('_'): temp_name+=key_sub[:3]
                name_sub+=f"{temp_name}{val}_"
                # if j>=1:
                #     break
                # j=j+1
            filesubdir.append(os.path.join(os.path.join(mainDir,name),name_sub))
        df['File_dir']= filesubdir
        os.makedirs(os.path.join(mainDir,name), exist_ok = True)
        file=os.path.join(mainDir,name,'list.csv')
        if os.path.isfile(file):
            df2=pd.read_csv(file)
            #print(df2['File_dir'])
            if not df.equals(df2):
                #tmp=df.append(df2, ignore_index=True)
                tmp=pd.concat([df, df2], ignore_index=True)
                print(tmp['File_dir'])
                nunique= tmp.drop(columns='File_dir').nunique()
                columns = nunique[nunique!=1].index
                diff = tmp.filter(columns)
                print(diff)
                filesubdir=[]
                for i in range(diff.shape[0]):
                    data=diff.loc[i]
                    name_sub=''
                    # j=0
                    for key in diff:                        
                        if isinstance(data[key], int) and data[key]<10000: val=int(data[key])
                        elif isinstance(data[key], float): 
                            if int(data[key])==data[key] and data[key]<10000: val=int(data[key])
                            elif data[key]<10000 and data[key]>0.001: val=f'{data[key]}'
                            else: val=f'{data[key]:.2e}'      
                        elif isinstance(data[key], str): val=data[key][0:6].replace("\'","")
                        else: val=data[key]  
                        temp_name=""
                        for key_sub in key.split('_'): temp_name+=key_sub[:3]
                        name_sub+=f"{temp_name}{val}_"
                        # if j>=1:
                        #     break
                        # j=j+1
                    filesubdir.append(os.path.join(os.path.join(mainDir,name),name_sub))
                df['File_dir']=filesubdir[:df.shape[0]]
                df=pd.concat([df, df2], ignore_index=True)
                df=df.drop_duplicates()  # drop rows that are exactly identical 
            else: 
                print("the dataframes are exactly identical! check if you are not running the same command twice")
                return 0.
        df.to_csv(file, index=False)
        return df
                        
    def csv_populate_from_dict(self,name, user_dict):
        has_no_list=True
        self.name=name
        names=[]
        for key in user_dict:
            vars(self)[key]=user_dict[key]
            if isinstance(user_dict[key], list): has_no_list=False
        if has_no_list:
            vars(self)[key]=[vars(self)[key]]
        mainDir=os.getcwd()
        self.calc()
        df=pd.DataFrame.from_dict(vars(self))
        filesubdir=[]
        for i in range(df.shape[0]):
            data=df.loc[i]
            name_sub=''
            for key in user_dict:
                #[print(sym[:3]) for sym in key.split('_')]
                name_sub+=f"{key}={data[key]}_"
            filesubdir.append(os.path.join(os.path.join(mainDir,name),name_sub))
        df['File_dir']= filesubdir
        os.makedirs(os.path.join(mainDir,name), exist_ok = True)
        file=os.path.join(mainDir,name,name+'.csv')
        if os.path.isfile(file):
            df2=pd.read_csv(file)
            if not df.equals(df2): 
                df=pd.concat([df, df2], ignore_index=True)
                df=df.drop_duplicates()  # drop rows that are exactly identical 
            else: 
                print("the dataframes are exactly identical! check if you are not running the same command twice")
                return 0.
        df.to_csv(file, index=False)
        return df
    


if __name__ == "__main__":
    user=sys.argv
    noArg=len(user)
    if noArg < 2:
        user.append(input("the script needs an appropriate name: "))
        tRecX_sim().calc()
        user.append(input("Using the above mentioned keys,\n input a dictionary to modify any defaults with new (array) values : "))
    name=user[1]
    user_d= ast.literal_eval(user[2])
    tRecX_sim().csv_populate_from_dict(name,user_d)
    tRecX_write_inp(os.path.join(os.getcwd(),name,name+'.csv'))
    user=os.environ['USER']
    if user=='mano':
        row_wise_submit_local(os.path.join(os.getcwd(),name,name+'.csv'))    
    elif user=='gaf': 
        row_wise_submit(os.path.join(os.getcwd(),name,name+'.csv'))