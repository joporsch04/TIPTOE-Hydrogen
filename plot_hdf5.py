#!/usr/bin/env python3.8.8

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import trapz
plt.rcParams['axes.grid'] = True
import pandas as pd
import os   
#import h5py|
from matplotlib.backends.backend_pdf import PdfPages
from __init__ import Fourier_filter, FourierTransform as FT, soft_window
from __init__ import InverseFourierTransform as IFT
from __init__ import AtomicUnits as AU
#from __init__ import get_complex_envelope
from scipy.interpolate import griddata, interp1d
from scipy import constants
#from progressbar import progressbar
import math
import glob

plt.rc('legend',fontsize=6)

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2,(y1-y2)/2,v2)
    adjust_yaxis(ax1,(y2-y1)/2,v1)

def adjust_yaxis(ax,ydif,v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny>maxy or (-miny==maxy and dy > 0):
        nminy = miny
        nmaxy = miny*(maxy+dy)/(miny+dy)
    else:
        nmaxy = maxy
        nminy = maxy*(miny+dy)/(maxy+dy)
    ax.set_ylim(nminy+v, nmaxy+v)

def normalize(x):
    '''Normalize the array x to the maximum value of x and return the normalized array and the maximum value of x'''
    x=np.asarray(x)
    return x/np.max(abs(x)), np.max(abs(x))
def normalize_only(x):
    '''Normalize the array x to the maximum value of x and return the normalized array and the maximum value of x'''
    x=np.asarray(x)
    return x/np.max(abs(x))#, np.max(abs(x))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def plot_laser(laser, type, file_name):
    group= pd.concat(laser)
    pdf=PdfPages(f'png/{file_name}/pulse_{type}.pdf', 'a')
    val=[type+'[0]', type+'[1]']
    for group_name, grp in group.groupby(["type", "delay(injection-test)"]):
        plot_table = grp.pivot(columns='name', values=val)    
        fig, axes = plt.subplots(nrows=4, figsize=(20,10))
        for col in plot_table.columns:
            if '0' not in col[0]:
                time=(plot_table.index.to_numpy())*AU.second
                amp=plot_table[col].to_numpy()
                spec, omega=FT(time, amp, t0=0)
                omega=omega.flatten()

                laser_FT=pd.DataFrame()
                laser_FT[f'omega_{col}']=omega
                laser_FT[col]=spec
                maximum=abs(laser_FT[col]).max()
                #max_ind=laser_FT[col].idxmax()
                cut_frac=0.001
                laser_FT=laser_FT[abs(laser_FT[col])>cut_frac*maximum]

                X=laser_FT[f'omega_{col}']/2/np.pi/1e15
                Yl=laser_FT[col]

                axes[0].plot(X, np.real(Yl), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                axes[0].set(ylabel=f'Re({type})')
                axes[0].legend(loc=1, prop={'size': 8})
                axes[1].plot(X, np.imag(Yl), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                axes[1].set(ylabel=f'Im({type})')
                axes[1].legend(loc=1, prop={'size': 8})
                axes[2].plot(X, np.abs(Yl), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                axes[2].set(ylabel=f'|{type}|')    
                axes[2].legend(loc=1, prop={'size': 8})
                plt.setp(axes[:3], xlabel=r'$\nu$ (PHz)', xlim=(0,2.5))
        fig.suptitle(f'{group_name} FFT of the pulse')    
        #axes.legend(loc=1, prop={'size': 6})

        plot_table['time(s)']=plot_table.index.values*AU.second
        fig.suptitle(f'{group_name[0]} Pulse with Injection field')    
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.xticks(plot_table['time(s)'][::200])
        plot_table.plot(x='time(s)',ax=axes[3], ylabel=val)
        axes[3].legend(loc=1, prop={'size': 5})
        plot_table=plot_table.drop(columns='time(s)')
        fig.tight_layout()
        plt.grid()
        pdf.savefig()
        #plt.show()
        plt.close()
    pdf.close()

from PIL import Image, ImageDraw
def plot_expec_video(laser, expec, Injection_param, Drive_param, file_name, expec_injection=None):
    first_lev="name"
    expec_vals= pd.concat(expec)
    if expec_injection!=None: 
        expec_injection_vals=pd.concat(expec_injection)
        background_val=expec_injection_vals.groupby([first_lev])
    laser_val= pd.concat(laser)
    laser_gif=laser_val.groupby([first_lev])
    grouped=expec_vals.groupby([first_lev])
    val=['Field[0]', 'Apot[1]']
    for group_name_main, grp_main in grouped:
        gif_image_ar=[]
        if expec_injection!=None: 
            tmp_grp=background_val.get_group(group_name_main)
            expec_val_array_inj=tmp_grp.columns[1:tmp_grp.columns.to_list().index("type")]
            background_expec_table=tmp_grp.pivot(columns='name', values=expec_val_array_inj)
        laser_top=laser_gif.get_group(group_name_main).groupby(["delay(injection-test)"])
        for group_name, grp in grp_main.groupby(["delay(injection-test)"]):
            laser_sub=laser_top.get_group(group_name)
            laser_table = laser_sub.pivot(columns='name', values=val)
            fig, axes = plt.subplots(nrows=2, figsize=(20,10), sharex=True)
            plt.setp(axes[:], xlim=(-250,250), xlabel="time (au)")
            expec_val_array=grp.columns[1:grp.columns.to_list().index("delay(injection-test)")]
            plot_table=grp.pivot(columns='name', values=expec_val_array)
            for col in plot_table.columns:
                time=plot_table.index.to_numpy()
                #time_inj=background_expec_table.index.to_numpy()
                Expec_value=plot_table[col].to_numpy()                
                if "<Exp0>" in col: 
                    Expec_value=1-Expec_value
                    col=["1-eDens[0,5]", col[1]]
                elif "<P1[0]>" in col:
                    Expec_value=1-Expec_value
                    col=["1-ground", col[1]]
                elif "<Ovr(Phi.Eta.Rn.Rn)>" in col: 
                    Expec_value=1-Expec_value
                    col=["1-Bound", col[1]]
                elif "<H0>" in col: 
                    continue
                elif "<Exp1>" in col: 
                    col=["eDens[5,10]", col[1]]
                elif "<Exp2>" in col: 
                    col=["eDens[10,15]", col[1]]
                elif "<Exp3>" in col: 
                    col=["eDens[15,50]", col[1]]
                axes[0].plot(time, Expec_value, label=col[0])
            #axes[0].set_yscale('log')
            axes[0].legend(fontsize=9)
            axes[0].set_ylabel("Electron Density")
            axes[1].plot(laser_table.index.to_numpy(), laser_table['Field[0]']/laser_table['Field[0]'].max(), label=f"Injection Field/{laser_table['Field[0]'].to_numpy().max()}")
            axes[1].plot(laser_table.index.to_numpy(), laser_table['Apot[1]']/laser_table['Apot[1]'].max(), label=f"Drive Vector Potential/{laser_table['Apot[1]'].to_numpy().max()}")
            axes[1].legend(fontsize=9, loc="right")
            axes[1].set_xlabel("Normalised Field and vector potential")
            plt.grid(True)
            plt.savefig(f"png/{file_name}/tmp.png")
            plt.close()
            gif_image_ar.append(Image.open(f"png/{file_name}/tmp.png"))
        gif_image_ar[0].save(f'png/{file_name}/expec_vals_{group_name_main}.gif',save_all=True, append_images=gif_image_ar[1:], optimize=True, duration=80, loop=0)

def plot_current(scan, Injection_param, Drive_param, file_name, val='Kx'):
    group=pd.concat(scan)
    pdf=PdfPages(f'png/{file_name}/current.pdf', 'a')
    inj_g = pd.concat(Injection_param).groupby("type")
    for group_name, grp in group.groupby("type"):
        Inj=inj_g.get_group(group_name)
        Inj_table=Inj.pivot(columns='name', values='wavelength')
        plot_table = grp.pivot(columns='name', values=val)
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
        plt.suptitle(group_name)
        #plt.tick_params(axis='x', which='major', labelsize=3)
        #plt.xticks(plot_table.index)
        #plot_table.plot.scatter(ax=axes[0],figsize=(10,5), ylabel=val)
        for col in plot_table.columns:
            scan_tmp=plot_table.dropna(subset=[col])
            tau=scan_tmp[col].index.to_numpy()
            if np.all(tau<1e-5): # if everything was stored in seconds
                    tau=tau*1e15
            else: #it probably used Optical Cycle units
                tau=(scan_tmp[col].index.to_numpy())*Inj_table[col].to_numpy()*1e-9/constants.c*1e15
                print(f"Wavelength={Inj_table[col].to_numpy()} nm, 1 OptCyc={Inj_table[col].to_numpy()*1e-9/constants.c*1e15} fs")
            #tau=scan_tmp[col].index.to_numpy()
            amp_s=scan_tmp[col].to_numpy()
            S_interpolated=interp1d(tau, amp_s, kind='cubic', fill_value='extrapolate')
            tau_interp=np.linspace(-3,3,1000)
            axes[0].plot(tau_interp,S_interpolated(tau_interp)/np.max(np.abs(S_interpolated(tau_interp))), linewidth=0.25, label=rf'$({np.max(np.abs(S_interpolated(tau_interp)))})^{{-1}} \times (S(\tau))$'+f' fold: {col}', alpha=0.75)
            axes[0].scatter(tau,amp_s/np.max(np.abs(S_interpolated(tau_interp))), s=0.25, alpha=0.5)
            axes[0].legend(fontsize = 6)
        axes[0].grid(True)
        maximum=plot_table.abs().max()
        maximum=maximum[1:].append(maximum.head(1))
        ind=-1
        #diff=plot_table.T.diff(periods=-1).T
        #diff=abs(diff/maximum.values*100)
        change=plot_table.T.pct_change(periods=-1).T
        diff=change
        diff.plot(ax=axes[1], ylabel=r"$\frac{\Delta(E2-E1)}{max(abs(E2))}$%")
        plt.ylim((-5,5))
        axes[1].grid(True)
        fig.tight_layout()
        plt.legend(fontsize = 6)
        pdf.savefig()
        #plt.show()
        plt.close()
    for group_name, grp in group.groupby("name"):
        plot_table = grp.pivot(columns='type', values=val)
        fig, axes = plt.subplots(nrows=2, sharex=True)
        plt.suptitle(group_name)
        plt.tick_params(axis='x', which='major', labelsize=3)
        plt.xticks(plot_table.index)
        plot_table.plot(ax=axes[0],figsize=(10,5), ylabel=val)
        axes[0].grid(True)
        maximum=plot_table.abs().max()
        maximum=maximum[1:].append(maximum.head(1))
        ind=-1
        #diff=plot_table.T.diff(periods=-1).T
        #diff=abs(diff/maximum.values*100)
        change=plot_table.T.pct_change(periods=-1).T
        diff=change
        diff.plot(ax=axes[1], ylabel=r"$\frac{\Delta(E2-E1)}{max(abs(E2))}$%")
        plt.ylim((-5,5))
        axes[1].grid(True)
        fig.tight_layout()
        plt.legend(fontsize = 2)
        pdf.savefig()
        #plt.show()
        plt.close()
    pdf.close()
    return 0

def plot_FT(scan,laser, laser_injec ,expec, S_Rn, inj_param, test_param, file_name,column_group='type',main_group='name'):
    # main_group='type'
    # column_group='name'    
    check_recollision=False
    nl='\n'
    nrows=4
    ncols=4
    scan_g = pd.concat(scan).groupby(main_group)
    inj_g = pd.concat(inj_param).groupby(main_group)
    test_g = pd.concat(test_param).groupby(main_group)
    if len(laser_injec)!=0:
        laser_injec_g=pd.concat(laser_injec).groupby(main_group)
    laser_g = pd.concat(laser).groupby([main_group,"delay(injection-test)"])
    expec_g = pd.concat(expec).groupby([main_group,"delay(injection-test)"])
    S_Rn_g = pd.concat(S_Rn).groupby([main_group,"delay(injection-test)"])
    val='Kz'
    ionization_P='e_density'
    val2_ar=['Apot[1]']
    for val2 in val2_ar:
        for type_name, sg in scan_g:
            print(type_name)
            drive_wave=type_name.split("_")[0].split('nm')[0]
            print(drive_wave)
            phase=type_name.split("_")[1]
            print(phase)
            fig, axes= plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,15))
            try:
                pdf=PdfPages(f'png/{file_name}/FFT_scan_{type_name.split("=")[-1]}_{type_name.split("=")[1]}.pdf')
            except:
                pdf=PdfPages(f'png/{file_name}/FFT_scan_{type_name.replace("/","_")}.pdf')
            if len(val2_ar)>1:
                fig.suptitle(type_name+val2)
            else:
                fig.suptitle(type_name)
            plt.tick_params(axis='x', which='major', labelsize=8)
            plt.tick_params(axis='y', which='major', labelsize=8)
            Inj=inj_g.get_group(type_name)
            test=test_g.get_group(type_name)
            scan_table = sg.pivot(columns=column_group, values=val)
            P_ion=sg.pivot(columns=column_group, values=ionization_P)
            Inj_table=Inj.pivot(columns=column_group, values='wavelength')
            for col in scan_table.columns:
                scan_FT=pd.DataFrame()
                tmp_FT=pd.DataFrame()
                scan_IFT=pd.DataFrame()
                delV_IFT=pd.DataFrame()
                laser_FT=pd.DataFrame()            
                scan_tmp=scan_table.dropna(subset=[col])
                P_ion_tmp=P_ion.dropna(subset=[col])
                tau=(scan_tmp[col].index.to_numpy())
                center=find_nearest(tau, 0)
                lg=laser_g.get_group((type_name,center))
                if len(laser_injec)!=0:
                    l_inj=laser_injec_g.get_group((type_name))
                    laser_injec_table= l_inj.pivot(columns=column_group, values=val2).dropna(subset=[col])
                laser_table = lg.pivot(columns=column_group, values=val2).dropna(subset=[col])
                laser_table_field=lg.pivot(columns=column_group, values='Field[1]').dropna(subset=[col])
                laser_table_inj = lg.pivot(columns=column_group, values='Field[0]').dropna(subset=[col])
                laser_table_inj_pot = lg.pivot(columns=column_group, values='Apot[0]').dropna(subset=[col])
                if np.all(tau<1e-5): # if everything was stored in seconds
                    tau=tau*1e15
                    tau_au=tau/AU.fs
                else: #it probably used Optical Cycle units
                    tau=(scan_tmp[col].index.to_numpy())*Inj_table[col].to_numpy()*1e-9/constants.c*1e15
                    tau_au=tau/AU.fs
                    print(f"Wavelength={Inj_table[col].to_numpy()} nm, 1 OptCyc={Inj_table[col].to_numpy()*1e-9/constants.c*1e15} fs")
                    #tau=(scan_tmp[col].index.to_numpy())*750*1e-9/constants.c*1e15
                time_au=(laser_table[col].index.to_numpy()+center)
                time=time_au*AU.fs
                if len(laser_injec)!=0:
                    amp_l=-1*laser_table[col].to_numpy()+laser_injec_table[col].to_numpy()
                else:
                    amp_l=-1*laser_table[col].to_numpy()
                amp_l_field=laser_table_field[col].to_numpy()
                time_inj=laser_table_inj[col].index.to_numpy()*AU.fs
                amp_inj_field=laser_table_inj[col].to_numpy()
                amp_inj_pot=-1*laser_table_inj_pot[col].to_numpy()
                ###### soft window the delay scan #######
                amp_s=-1*scan_tmp[col].to_numpy()
                P_ion_s=P_ion_tmp[col].to_numpy()
                current_shift=""
                if abs(amp_s[-1])>max(abs(amp_s))/100:
                    current_shift=f"-{amp_s[-1]}"
                    amp_s=amp_s-amp_s[-1]
                tau=tau
                tmp=pd.DataFrame({"tau(fs)": tau, "tau(au)": tau_au, "S(tau)": amp_s, "P_ion(tau)": P_ion_s, "V_x(tau)": amp_s/P_ion_s})
                tmp.to_csv(f'png/{file_name}/Current_{type_name.replace("/","_")}.csv')
                tmp=pd.DataFrame({"t(fs)": time, "t(au)": time_au, "Field(t)": amp_l_field, "Apot(t)": amp_l})
                tmp.to_csv(f'png/{file_name}/Probe_{type_name.replace("/","_")}.csv')
                tmp=pd.DataFrame({"t(fs)": time_inj, "t(au)": time_au, "Field(t)": amp_inj_field, "Apot(t)": amp_inj_pot})
                tmp.to_csv(f'png/{file_name}/Pump_{type_name.replace("/","_")}.csv')
                if not tau[3]>-2.5:
                    amp_s=amp_s*(soft_window(tau, -2.0, tau[1])*soft_window(tau, 2.0, tau[-2]))#*soft_window(tau, -.8, tau[8])*soft_window(tau, .8, tau[-8])
                S_interpolated=interp1d(tau, amp_s, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)
                tau_interp=np.linspace(tau[0],tau[-1],1000)
                SMax=max(np.abs(S_interpolated(tau_interp)))
                axes[0,2].plot(tau_interp,S_interpolated(tau_interp)/max(np.abs(S_interpolated(tau_interp))), linewidth=0.5, 
                               label=rf'$S(\tau){current_shift}, S_{{max}}={SMax:.2e}$'+f' {nl} folder: {col}')
                axes[0,2].scatter(tau,amp_s/max(np.abs(S_interpolated(tau_interp))), s=1)
                AxS=axes[0,2].twinx()
                AxS.plot(time,amp_l, label=r'$A_{probe}(t)$'+col, alpha=0.5, color='grey')
                axes[0,2].set_ylim(-1.1,1.1)
                #AxS.set_ylim(axes[0,2].get_ylim()[0]*AxS.get_ylim()[0],AxS.get_ylim()[1]/axes[0,2].get_ylim()[0])
                AxS.set_ylim(-1.1*max(abs(amp_l)),1.1*max(abs(amp_l)))
                AxS.grid(False)
                AxS.set(ylabel=r'$A_{probe}(t)$')
                axes[0,2].set(ylabel=r'$S(\tau)$ normalized', xlabel='delay (fs)')
                axes[0,2].legend(loc='upper left')
                V_interpolated=interp1d(tau, -1*amp_s/P_ion_s, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)
                axes[0,3].plot(tau_interp,V_interpolated(tau_interp), linewidth=0.5, label=rf'$V(\tau){current_shift}$'+f' {nl} folder: {col}')
                axes[0,3].scatter(tau,-1*amp_s/P_ion_s, s=1)
                AxV=axes[0,3]#.twinx()
                AxV.plot(time,-1*amp_l, label=r'$-A_{probe}(t)$'+col, color='grey')
                AxV.grid(False)
                #axes[0,3].set_ylim(-0.9*max(abs(amp_s/P_ion_s)),1.1*max(abs(amp_s/P_ion_s)))
                #AxV.set_ylim(-0.9*max(abs(amp_s/P_ion_s)),1.1*max(abs(amp_s/P_ion_s)))
                axes[0,3].set(ylabel=r'$v(\tau)$ (au)', xlabel='time (fs)', xlim=(-5,5))
                #align_yaxis(AxV, 0, axes[0,3], 0)
                axes[0,3].legend()

                 #axes[0,2].set_xticks(tau)
                #axes[0,2].xaxis.set_ticklabels([])
                #axes[0,2].set_xticks()
                ###### get the frequencies for which the function shall return the FT '######### 
                
                spec_tmp, omega =FT(time, amp_l, t0=0)
                spec_tmp2=spec_tmp#FT(time, amp_l_field, omega=omega)
                spec_tmp=np.abs(spec_tmp).flatten()
                omega_mid=(omega[abs(spec_tmp)>np.max(spec_tmp)*0.10])[-1]
                omega=omega[abs(spec_tmp)>np.max(spec_tmp)*0.01]
                omega=omega[omega>0]
                omega_max=omega[-1]
                omega_min=omega[0]
                x=False
                if x:
                    tmp,omega=FT(tau, amp_s)
                    omega=omega[omega>omega_min]
                    omega=omega[omega<omega_max]
                else:
                    omega=np.linspace(omega_min, omega_max,10000)
                

                ####### FFT of the delay scan with the frequencies obtained ###########
                spec= FT(tau, amp_s, omega=omega)
                scan_FT[f'omega_{col}']=omega
                scan_FT[col]=spec

                ####### FFT of the probe vector potential at the calculated cirecular frequncies #########
                spec=FT(time, amp_l, t0=0 , omega=omega)  
                laser_FT[f'omega_{col}']=omega
                laser_FT[col]=spec
                omega=laser_FT[f'omega_{col}'].to_numpy()

                ####### FFT of the Probe Field #########
                amp_Eprobe=1*laser_table_field[col].to_numpy()
                spec_Eprobe=FT(time, amp_Eprobe, omega=omega, t0=0)

                # ####### FFT of the Injection vector potential #########
                # time_inj=(laser_table_inj[col].index.to_numpy())*AU.second*1e15
                # amp_l_inj=laser_table_inj[col].to_numpy()
                # spec_inj, omega_inj=FT(time_inj, amp_l_inj, t0=0) 
                # spec_inj=spec_inj.flatten()
                # omega_inj=omega_inj[spec_inj>np.max(spec_inj)*0.01]
                # omega_inj=omega_inj[omega_inj<6*2*np.pi]
                # omega_inj=omega_inj[omega_inj>=0]
                # omega_inj_max=omega_inj[-1]
                # omega_inj_min=omega_inj[0]
                # omega_inj=np.linspace(omega_inj_min, omega_inj_max,5000)
                # spec_inj=FT(time_inj, amp_l_inj, t0=0, omega=omega_inj)

                ####### plotting real imaginary and absolute values
                X=laser_FT[f'omega_{col}']/2/np.pi
                Ys=scan_FT[col]
                #plt.xticks(X[:-1:4])
                axes[0,1].plot(X, normalize_only(np.unwrap(np.angle(Ys))), label=f'tRecX, max={max(abs(np.unwrap(np.angle(Ys)))):.2e} {nl} folder: {col}')
                axes[0,1].set(ylabel=r'$\Phi(S,\nu)$')
                axes[0,0].plot(X, normalize_only(np.abs(Ys)), label=f'tRecX, max={max(abs(abs(Ys))):.2e} {nl} folder: {col}')
                axes[0,0].set(ylabel=r'$|S(\nu)|$')
                #plt.show()
                #axes[0,0].legend()
                #axes[0,1].legend()
                axes[0,0].legend()
                Yl=laser_FT[col].to_numpy()
                ax_tmp=axes[0,1].twinx()
                ax_tmp.plot(X, (np.unwrap(np.angle(Yl))), label=f'A_probe, {nl} folder: {col}', color='grey')
                ax_tmp.set(ylabel=r'$\Phi(A_{probe}, \nu)$')
                #ax_tmp.set_ylim(axes[0,1].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[0,1].get_ylim()[1], ax_tmp.get_ylim()[1])
                ax_tmp.grid(False)
                ax_tmp=axes[0,0].twinx()
                ax_tmp.plot(X, (np.abs(Yl)), label=f'A_probe, {nl} folder: {col}', color='grey')
                ax_tmp.set(ylabel=r'|$A_{probe}$|')  
                ax_tmp.set_ylim(axes[0,0].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[0,0].get_ylim()[1], ax_tmp.get_ylim()[1])
                
                ax_tmp.grid(False)


                # Yl_inj=spec_inj
                # Yl_inj=Yl_inj/max(abs(Yl_inj))*max(abs(Yl))
                # X_inj=omega_inj/2/np.pi
                # axes[1,0].plot(X_inj, np.real(Yl_inj), label=f'injection {col} normalised')
                # axes[1,1].plot(X_inj, np.imag(Yl_inj), label=f'injection {col} normalised')
                # axes[1,0].plot(X_inj, np.abs(Yl_inj), label=f'injection {col} normalised')
                # axes[1,0].legend()
                # axes[1,1].legend()
                # axes[1,0].legend()              

                # the conjugatuion due to the definition of the Fourier transform for defined delay time
                Y=np.conjugate(np.divide(Ys,Yl))#*soft_window(X.to_numpy(), 1., 2.0)
                # tmp=pd.DataFrame({"v(PHz)": X, "G(v)": Y})
                # tmp.to_csv(f'png/{file_name}/G_v_{type_name.replace("/","_")}.csv')
                #axes[1,1].plot(X, np.angle(Y), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                axes[1,1].plot(X, normalize_only(np.unwrap(np.angle(Y))), label=f'tRecX, max={max(abs(np.unwrap(np.angle(Y)))):.2e} {nl} folder: {col}')
                axes[1,1].set(ylabel=r'$\Phi$($G(\nu)$)')
                axes[1,0].plot(X, normalize_only(np.abs(Y)), label=f'tRecX, max={max(abs(np.abs(Y))):.2e} {nl} folder: {col}')
                axes[1,0].set(ylabel=r'|$G(\nu)$|')
                axes[1,0].grid()
                axes[1,1].grid()
                axes[1,0].grid()
                nu0=299.79/float(Inj_table[col].to_numpy())#0.399
                max_nu_int=int(min(X.to_numpy()[-1],3.29)/nu0)+1
                labels=[]
                for i in  np.arange(0,max_nu_int,1):
                    labels.append(rf"{i}$\nu_0$")
                plt.setp(axes[:,:2], xticks=nu0*np.arange(0,max_nu_int,1),xticklabels=labels, xlabel=rf'$\nu_0$= {nu0:.2f} PHz', xlim=(-1e-3,nu0*max_nu_int))
                #Y=Y*soft_window(X.to_numpy(), omega_mid/2/np.pi, omega_max/2/np.pi)
                X, Y = X.to_numpy(), Y.to_numpy()    
                #print(Y[0])
                #Y=Y[X>0.1]#*soft_window(X, 3.5, 4.5)#*soft_window(X, 1., 1.5)#*soft_window(X, .5, .4)#*soft_window(X,3, 3.2)#*soft_window(X, 2.9,3.1)#*soft_window(X, 1.3, 1.1)  
                #X=X[X>0.1]#Y=Y*soft_window(X, .8,.25)
                #### worked for low bandwidth pulse soft_window(X, 1., 1.5)###
                Y_imag=np.imag(Y)#*soft_window(X, 1., 2.0)
                Y_real=np.real(Y)#*soft_window(X, 1., 2.0)   
                #Y_imag=np.insert(np.imag(Y)*soft_window(X, 0.25, .005), 0, 0)
                #f_real_extrap=interp1d(X, np.real(Y), fill_value='extrapolate')
                #Y_real=np.insert(np.real(Y),0, f_real_extrap(0))
                X_omega=X*2*np.pi
                #X=np.insert(X,0,0)

                # freq_time_analysis=False
                # if freq_time_analysis:
                #     Y0=Y*soft_window(X, .5, 2.2)*soft_window(X, .2, 0)
                #     del_amp, time =IFT(X,Y0, omega0=.3)#, t_array=time)
                #     scan_IFT[f'time_{col}']=time
                #     scan_IFT[col]=del_amp
                #     axes[2,0].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                #     axes[2,0].set(ylabel=r'G(t) in low frequncy regime', xlabel='time (fs)')

                #     Y1=Y*soft_window(X, 1.3, 2.2)*soft_window(X, .5, 0)
                #     del_amp, time =IFT(X,Y1, omega0=.85)#, t_array=time)
                #     scan_IFT[f'time_{col}']=time
                #     scan_IFT[col]=del_amp
                #     axes[2,1].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                #     axes[2,1].set(ylabel=r'G(t) in middle frequncy regime', xlabel='time (fs)')

                #     Y2=Y*soft_window(X, 1.9, 2.2)*soft_window(X, 1.5, 0)
                #     del_amp, time =IFT(X,Y2, omega0=1.625)#, t_array=time)
                #     scan_IFT[f'time_{col}']=time
                #     scan_IFT[col]=del_amp
                #     axes[2,0].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                #     axes[2,0].set(ylabel=r'G(t) in high frequncy regime', xlabel='time (fs)')#, ylim=(0,.5e-17))


                Y3=Y_real+1j*Y_imag   
                
                tau2=np.linspace(tau[0], tau[-1], 50*len(tau))
                #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                del_amp =IFT(X_omega,Y3, t_array=tau2)
                scan_IFT[f'time_{col}']=tau2
                scan_IFT[col]=del_amp
                axes[1,2].plot(scan_IFT[f'time_{col}'], normalize_only(np.real(scan_IFT[col])), label=f'tRecX, max={max(abs(np.real(scan_IFT[col]))):.2e} {nl} folder: {col}', linewidth=0.5) 
                
            
            # del_amp =IFT(X,Ys.to_numpy(), t_array=tau2)*max(abs(del_amp))/max(abs(amp_s))
            # scan_IFT[col]=del_amp
            # axes[1,2].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'S(tau) reconstructed and normalised {col}')
                axes[1,2].set(ylabel=r'G(tau)', xlabel='time (fs)')#, ylim=(0,.5e-17))         
                axes[1,2].legend(fontsize=6)
                scn=scan_table
                #scn.index=scn.index*1e15
                #scn.plot(ax=axes[0,2], xlabel='delay (fs)', ylabel=f'current in direction : {val[-1]}', legend=False)
                #axes[0,2].set_xticks(scn.index[::3])
                #axes[0,2].grid()
                #cen=0#-6.807653999999999e-15
                lg=laser_g.get_group((type_name,center))
                laser_table=lg.pivot(columns=column_group, values=["Apot[1]", "Field[0]", "Field[1]"])
                if len(laser_injec)!=0:
                    laser_injec_table= l_inj.pivot(columns=column_group, values=["Apot[1]", "Field[0]"])
                    laser_table["Apot[1]"]=laser_table["Apot[1]"]#-laser_injec_table["Apot[1]"]
                
                # laser_table["Field[0]"]=laser_table["Field[0]"]/laser_table["Field[0]"].abs().max()
                # laser_table["Apot[1]"]=laser_table["Apot[1]"]/laser_table["Apot[1]"].abs().max()
                lsr=laser_table
                lsr['time(fs)']=lsr.index.values*AU.second*1e15
                ax_tmp=axes[1,2].twinx()
                lsr.plot(x='time(fs)', y=["Field[0]"],ax=ax_tmp,ylabel=rf'$E_{{pump}}(t)$ $\tau$={center:.1f}', legend=False, color='grey')
                ax_tmp.grid(False)
                ax_tmp.set_ylim(axes[1,2].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[1,2].get_ylim()[1], ax_tmp.get_ylim()[1])
                
                
                if(check_recollision):

                    t=51.82#850/AU.nm/AU.speed_of_light/2.3


                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm1.25e+14_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()

                    #%%
                    E_inj1=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm7.35e+13_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()
                    E_inj2=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    # plt.plot(time_inj+t, -1*E_inj_exact)
                    # plt.plot(time_inj-t, -1*E_inj_exact)

                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm1.41e+13_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()
                    E_inj3=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    # plt.plot(time_inj+2*t, 1*E_inj_exact)
                    # plt.plot(time_inj-2*t, 1*E_inj_exact)

                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm7.45e+11_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()
                    E_inj4=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    # plt.plot(time_inj+3*t, -1*E_inj_exact)
                    # plt.plot(time_inj-3*t, -1*E_inj_exact)
                    #%%
                    time_inj=np.linspace(time_inj[0], 6*t, 100000)

                    ax_tmp.plot(time_inj*AU.fs, E_inj1(time_inj)-E_inj2(time_inj+t)-E_inj2(time_inj-t)
                                +E_inj3(time_inj+2*t)+E_inj3(time_inj-2*t)-E_inj4(time_inj+3*t)
                                -E_inj4(time_inj-3*t), label='reconstructed', color='grey')
                    ax_tmp.grid(False)
                    # %%

                    Ion_prob=0
                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm1.25e+14_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current1=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)

                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm7.35e+13_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+2*np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current2=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)#

                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm1.41e+13_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+2*np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current3=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)

                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm7.45e+11_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+2*np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current4=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)

                    t_tRecX=np.linspace(-3*t, 3*t, 1000)
                    #plt.plot(t_tRecX,current1(t_tRecX), label='only central half cycle', linestyle='dotted')
                    current_recon=current1(t_tRecX)-current2(t_tRecX+t)-current2(t_tRecX-t)+current3(t_tRecX+2*t)+current3(t_tRecX-2*t)-current4(t_tRecX+3*t)-current4(t_tRecX-3*t)
                    print(Ion_prob)
                    print(np.mean(P_ion_s))
                    Ion_prob=np.mean(P_ion_s)
                    axes[0,2].plot(t_tRecX*AU.fs, normalize_only(current_recon), label=f'recon. half-cycles, max={max(abs(current_recon)):.2e}')
                    V_reconstruct=-1/Ion_prob*(current1(t_tRecX)-current2(t_tRecX+t)-current2(t_tRecX-t)+current3(t_tRecX+2*t)+current3(t_tRecX-2*t)-current4(t_tRecX+3*t)-current4(t_tRecX-3*t))
                    axes[0,3].plot(t_tRecX*AU.fs, V_reconstruct, label='recon. half-cycles')
                    Del_V_recollision=V_interpolated(t_tRecX*AU.fs)-V_reconstruct
                    axes[2,3].plot(t_tRecX*AU.fs, Del_V_recollision, label='Recollision')
                    ax_tmp=axes[2,3].twinx()
                    ax_tmp.plot(time, normalize_only(amp_Eprobe), label=r'E_{probe}(t)', color='grey', alpha=0.7)
                    ax_tmp.plot(time, normalize_only(amp_inj_field), label=r'E_{inj}(t)', color='grey', alpha=0.3)
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)
                    ax_tmp=axes[2,3]
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)
                    ax_tmp.set_xlim(-3,3)
                    axes[2,3].set_xlabel(r'\tau (fs)')
                    axes[2,3].set_ylabel(r'$\Delta V^{recol.}$ (a.u.)')
                    spec_Del_V_recollision=FT(t_tRecX*AU.fs, Del_V_recollision, omega=omega)
                    axes[2,0].plot(X, abs(spec_Del_V_recollision), label='Recollision')
                    #axes[2,0].set_ylabel(r'$|\Delta V|$ (a.u.)')
                    axes[2,1].plot(X, np.unwrap(np.angle(spec_Del_V_recollision)), label='Recollision')
                    #axes[2,1].set_ylabel(r'$\phi(\Delta V)$ (rad)')


                    spec_f=np.conjugate(np.divide(spec_Del_V_recollision,spec_Eprobe))
                    axes[3,0].plot(X, np.abs(spec_f), label=f'Recollision{nl} folder: {col}')
                    axes[3,0].set(ylabel=r'|f^{recol.}|')
                    axes[3,1].plot(X, np.unwrap(np.angle(spec_f)), label=f'Recollision{nl} folder: {col}')
                    axes[3,1].set(ylabel=r'$\Phi(f^{recol.})$')
                    tau2=np.linspace(tau[0], tau[-1], 500*len(tau))
                    #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                    del_amp =IFT(omega,spec_f, t_array=tau2)
                    delV_IFT[f'time_{col}']=tau2
                    delV_IFT[col]=del_amp
                    axes[3,3].plot(delV_IFT[f'time_{col}'], np.real(delV_IFT[col]), label=f'Recollision {nl} folder: {col}', linewidth=0.5)
                    axes[3,3].set(ylabel=r'$f^{recol.}(t)$')
                    axes[3,3].set(xlabel=r'$time (fs)$')

                #plt.setp(axes[:,2], xticks=np.arange(-6, 7, 1), xlim=(-6.5, 6.5))
                # ep=expec_g.get_group((type_name,center))
                # SRn=S_Rn_g.get_group((type_name,center))
                # SRn_table=SRn.drop_duplicates().pivot(columns=column_group, values=["<Overlap>"])
                # # try:
                # #     ep_table=ep.pivot(columns=column_group, values=["<Ovr(Phi.Eta.Rn.Rn)>"])
                # # except:
                # #     ep_table=ep.pivot(columns=column_group, values=["<Ovr(Orbital&Phi.Phi.Eta.Rn.Rn)>"])
                # # #ep_table=ep_table.apply(lambda x: 1-x)
                # # ep_table['time(fs)']=ep_table.index.values*AU.second*1e15
                # SRn_table['time(fs)']=SRn_table.index.values*AU.second*1e15
                # SRn_table.plot(x='time(fs)', ax=axes[1,3], ylabel='density of free electrons', xlim=(tau[0],tau[-1]), legend=False, color='grey')#, logy=True)
                if os.path.exists(f'png/{file_name}/SFA'):
                    df_tmp=pd.read_csv(f'png/{file_name}/SFA', index_col=0)
                    Gamma=(df_tmp['G_gamma']+df_tmp['G_T_2']+df_tmp['G_T_3']+df_tmp['G_dip']*0).to_numpy()
                    S_SFA=[]
                    for delay in tau_au:
                        ApotX=np.interp(df_tmp.index.to_numpy()+delay, time_au, amp_l)
                        S_SFA.append(trapz(Gamma*ApotX, df_tmp.index.to_numpy()))
                    
                    del_v=-1*(amp_s/P_ion_s-S_SFA/np.sum(Gamma))


                    spec_v=FT(tau, del_v, omega=omega)
                    axes[2,1].plot(X, np.unwrap(np.angle(spec_v)), label=f'tRecX-SFA, {nl} folder: {col}')
                    axes[2,1].set(ylabel=r'$ \Phi(\Delta v, \nu)$')
                    axes[2,0].plot(X, np.abs(spec_v), label=f'tRecX-SFA,  {nl} folder: {col}')
                    axes[2,0].set(ylabel=r'$|\Delta v(\nu)|$')
                    del_v=interp1d(tau, del_v, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)(tau_interp)
                    axes[2,2].plot(tau_interp, del_v, label=f'tRecX-SFA, {nl} folder: {col}')
                    axes[2,2].set(ylabel=r'$\Delta v(\tau)$')
                    axes[2,2].set(xlabel=r'$\tau (fs)$')

                    spec_f=np.conjugate(np.divide(spec_v,spec_Eprobe))
                    axes[3,0].plot(X, np.abs(spec_f), label=f'{nl} folder: {col}')
                    axes[3,0].set(ylabel=r'|f|')
                    axes[3,1].plot(X, np.unwrap(np.angle(spec_f)), label=f'{nl} folder: {col}')
                    axes[3,1].set(ylabel=r'$\Phi(f)$')
                    tau2=np.linspace(tau[0], tau[-1], 500*len(tau))
                    #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                    del_amp =IFT(omega,spec_f, t_array=tau2)
                    delV_IFT[f'time_{col}']=tau2
                    delV_IFT[col]=del_amp
                    axes[3,2].plot(delV_IFT[f'time_{col}'], np.real(delV_IFT[col]), label=f'{nl} folder: {col}', linewidth=0.5)
                    axes[3,2].set(ylabel=r'$f(t)$')
                    axes[3,2].set(xlabel=r'$time (fs)$')

                    if(check_recollision):
                        del_v=-1*(interp1d(t_tRecX, V_reconstruct)(tau_au)-S_SFA/np.sum(Gamma))
                        ax22=axes[0,3]#.twinx()
                        spec_v=FT(tau, del_v, omega=omega)
                        axes[2,1].plot(X, np.unwrap(np.angle(spec_v)), label=f'half-SFA')
                        axes[2,1].set(ylabel=r'$ \Phi(\Delta v, \nu)$')
                        axes[2,0].plot(X, np.abs(spec_v), label=f'half-SFA,')
                        axes[2,0].set(ylabel=r'$|\Delta v(\nu)|$')
                        del_v=interp1d(tau, del_v, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)(tau_interp)
                        axes[2,2].plot(tau_interp, del_v, label=f'half-SFA')
                        axes[2,2].set(ylabel=r'$\Delta v(\tau)$')
                        axes[2,2].set(xlabel=r'$\tau (fs)$')


                        spec_f=np.conjugate(np.divide(spec_v,spec_Eprobe))
                        axes[3,0].plot(X, np.abs(spec_f), label=f'half')
                        axes[3,0].set(ylabel=r'|f|')
                        axes[3,1].plot(X, np.unwrap(np.angle(spec_f)), label=f'half')
                        axes[3,1].set(ylabel=r'$\Phi(f)$')
                        tau2=np.linspace(tau[0], tau[-1], 500*len(tau))
                        #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                        del_amp =IFT(omega,spec_f, t_array=tau2)
                        delV_IFT[f'time_{col}']=tau2
                        delV_IFT[col]=del_amp
                        axes[3,2].plot(delV_IFT[f'time_{col}'], np.real(delV_IFT[col]), label=f'half', linewidth=0.5)
                        axes[3,2].set(ylabel=r'$f(t)$')
                        axes[3,2].set(xlabel=r'$time (fs)$')


                    ax22=axes[0,3]#.twinx()
                    S_SFA=interp1d(tau, S_SFA, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)(tau_interp)
                    ax22.plot(tau_interp,-1*S_SFA/np.sum(Gamma), label=f'SFA', color='red', linestyle=':', alpha=0.5)

                    


                    axes[1,3].plot(df_tmp.index.to_numpy()*AU.fs, np.cumsum(Gamma), label='SFA Ionization', color='red')
                    G_spec= FT(df_tmp.index.to_numpy()*AU.fs, Gamma, omega=omega, t0=0)
                    ang= np.unwrap(np.angle(G_spec))
                    G_om=abs(G_spec)



                    S_SFA, S_SFA_max=normalize(S_SFA)

                    ax22=axes[0,2]#.twinx()
                    ax22.plot(tau_interp,S_SFA, label=f'SFA, max={S_SFA_max:.2e}', color='red', linestyle=':', alpha=0.5)
                    Gamma, Gamma_max=normalize(Gamma)
                    G_om, G_om_max=normalize(G_om)
                    ang, ang_max=normalize(ang)
                    ax22=axes[1,2]#.twinx()
                    ax22.plot(df_tmp.index.to_numpy()*AU.fs, Gamma, label=f'SFA, G_max={Gamma_max:.2e}', color='red')
                    #ax22.grid(False)
                    ax22=axes[1,1]#.twinx()
                    ax22.plot(omega/2/np.pi,ang, label=f'SFA, max={ang_max:.2e}', color='red')
                    #ax22.grid(False)
                    ax22=axes[1,0]#.twinx()
                    ax22.plot(omega/2/np.pi,G_om, label=f'SFA, max={G_om_max:.2e}', color='red')
                    Current_SFA=G_spec*Yl
                    C_abs, C_abs_max=normalize(abs(Current_SFA))
                    C_angle, C_angle_max=normalize(np.unwrap(np.angle(Current_SFA)))
                    ax22=axes[0,0]#.twinx()
                    ax22.plot(omega/2/np.pi,C_abs, label=f'SFA, max={C_abs_max:.2e}', color='red')
                    ax22=axes[0,1]#.twinx()
                    ax22.plot(omega/2/np.pi,C_angle, label=f'SFA, max={C_angle_max:.2e}', color='red')


                    ax_tmp=axes[2,2].twinx()
                    ax_tmp.plot(time, -amp_Eprobe, color='grey')
                    ax_tmp.set_ylabel(rf'-$E_{{probe}}(t)$')
                    #ax_tmp.set_ylim(axes[2,2].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[2,2].get_ylim()[1], ax_tmp.get_ylim()[1])
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)
                    ax_tmp.grid(False)
                    ax_tmp=axes[2,2]
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)


                    ax_tmp=axes[2,0].twinx()
                    ax_tmp.plot(X, abs(spec_Eprobe), color='grey')
                    ax_tmp.set_ylabel(rf'$E_{{probe}}(\nu)$')
                    ax_tmp.set_ylim(axes[2,0].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[2,0].get_ylim()[1], ax_tmp.get_ylim()[1])
                    ax_tmp.grid(False)


                    ax_tmp=axes[2,1].twinx()
                    ax_tmp.plot(X, np.unwrap(np.angle(spec_Eprobe)), color='grey')
                    ax_tmp.set_ylabel(rf'$\phi (E_{{probe}},\nu)$')
                    #ax_tmp.set_ylim(axes[2,1].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[2,1].get_ylim()[1], ax_tmp.get_ylim()[1])
                    ax_tmp.grid(False)

                axes[1,2].get_shared_x_axes().join(axes[0,2], axes[1,2])
                axes[1,2].get_shared_x_axes().join(axes[0,2], axes[1,2])
                axes[1,2].get_shared_x_axes().join(axes[0,2], axes[1,2])
                axes[1,2].set_xlim(tau[0]-.5,tau[-1]+.5)
                axes[1,3].get_shared_x_axes().join(axes[0,3], axes[1,3])
                axes[1,3].get_shared_x_axes().join(axes[0,3], axes[1,3])
                axes[1,3].get_shared_x_axes().join(axes[0,3], axes[1,3])
                axes[1,3].set_xlim(tau[0]-.5,tau[-1]+.5)
                axes[1,1].grid(True)
                #axes[1,3].grid(True)
                handles, labels = AxV.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                AxV.legend(by_label.values(), by_label.keys(), loc='lower right')
                plt.setp(axes[:,2], xlim=(tau[0]-.5,tau[-1]+.5))
                # handles, labels = AxS.get_legend_handles_labels()
                # by_label = dict(zip(labels, handles))
                # AxS.legend(by_label.values(), by_label.keys(), loc='lower right')
                # AxS.set_ylabel(r'$A_{probe}$')
                #AxV.set_ylabel(r'$A_{probe}$')
                for i in range(nrows):
                    for j in range(ncols):
                        axes[i,j].legend()
                plt.tight_layout()
                # align_yaxis(AxS, 0, axes[0,2], 0)
                # align_yaxis(AxV, 0, axes[0,3], 0)
            pdf.savefig()
            #plt.show()
            plt.close()
            pdf.close()    
    return 0


def plot_FT_tiptoe(scan,laser, laser_injec,expec_injec ,expec, S_Rn, inj_param, test_param, file_name,column_group='type',main_group='name'):
    # main_group='type'
    # column_group='name'    
    check_recollision=False
    nl='\n'
    nrows=4
    ncols=4
    scan_g = pd.concat(scan).groupby(main_group)
    inj_g = pd.concat(inj_param).groupby(main_group)
    test_g = pd.concat(test_param).groupby(main_group)
    #if len(laser_injec)!=0:
    laser_injec_g=pd.concat(laser_injec).groupby(main_group)
    expec_injec_g=pd.concat(expec_injec).groupby(main_group)
    laser_g = pd.concat(laser).groupby([main_group,"delay(injection-test)"])
    expec_g = pd.concat(expec).groupby([main_group,"delay(injection-test)"])
    S_Rn_g = pd.concat(S_Rn).groupby([main_group,"delay(injection-test)"])
    val='e_density'
    ionization_P='e_density'
    val2_ar=['Apot[0]']
    for val2 in val2_ar:
        for type_name, sg in scan_g:
            print(type_name)
            drive_wave=type_name.split("_")[0].split('nm')[0]
            print(drive_wave)
            phase=type_name.split("_")[1]
            print(phase)
            fig, axes= plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,15))
            try:
                pdf=PdfPages(f'png/{file_name}/FFT_scan_{type_name.split("=")[-1]}_{type_name.split("=")[1]}.pdf')
            except:
                pdf=PdfPages(f'png/{file_name}/FFT_scan_{type_name.replace("/","_")}.pdf')
            if len(val2_ar)>1:
                fig.suptitle(type_name+val2)
            else:
                fig.suptitle(type_name)
            plt.tick_params(axis='x', which='major', labelsize=8)
            plt.tick_params(axis='y', which='major', labelsize=8)
            Inj=inj_g.get_group(type_name)
            test=test_g.get_group(type_name)
            scan_table = sg.pivot(columns=column_group, values=val)
            P_ion=sg.pivot(columns=column_group, values=ionization_P)
            Inj_table=Inj.pivot(columns=column_group, values='wavelength')
            for col in scan_table.columns:
                scan_FT=pd.DataFrame()
                tmp_FT=pd.DataFrame()
                scan_IFT=pd.DataFrame()
                delV_IFT=pd.DataFrame()
                laser_FT=pd.DataFrame()            
                scan_tmp=scan_table.dropna(subset=[col])
                P_ion_tmp=P_ion.dropna(subset=[col])
                tau=(scan_tmp[col].index.to_numpy())
                center=find_nearest(tau, 0)
                lg=laser_g.get_group((type_name,center))
                if len(laser_injec)!=0:
                    l_inj=laser_injec_g.get_group((type_name))
                    laser_injec_table= l_inj.pivot(columns=column_group, values=val2).dropna(subset=[col])
                
                    laser_table_inj = l_inj.pivot(columns=column_group, values='Field[0]').dropna(subset=[col])
                    laser_table_inj_pot = l_inj.pivot(columns=column_group, values='Apot[0]').dropna(subset=[col])

                laser_table = lg.pivot(columns=column_group, values=val2).dropna(subset=[col])
                laser_table=laser_table-laser_table_inj_pot
                laser_table_field=lg.pivot(columns=column_group, values='Field[0]').dropna(subset=[col])
                laser_table_field=laser_table_field-laser_table_inj # important
                # laser_table_field=lg.pivot(columns=column_group, values='Field[0]').dropna(subset=[col])
                if np.all(tau<1e-5): # if everything was stored in seconds
                    tau=tau*1e15
                    tau_au=tau/AU.fs 
                elif np.all(tau<20): #it probably used Optical Cycle units
                    tau=(scan_tmp[col].index.to_numpy())*Inj_table[col].to_numpy()*1e-9/constants.c*1e15
                    tau_au=tau/AU.fs
                    print(f"Wavelength={Inj_table[col].to_numpy()} nm, 1 OptCyc={Inj_table[col].to_numpy()*1e-9/constants.c*1e15} fs")
                    #tau=(scan_tmp[col].index.to_numpy())*750*1e-9/constants.c*1e15
                else: # likely stored in atomic units
                    tau_au=tau
                    tau=tau*AU.fs
                
                time_au=(laser_table[col].index.to_numpy()+center)
                time=time_au*AU.fs
                if len(laser_injec)!=0:
                    amp_l=-1*laser_table[col].to_numpy()+laser_injec_table[col].to_numpy()
                else:
                    amp_l=-1*laser_table[col].to_numpy()
                amp_l_field=laser_table_field[col].to_numpy()
                time_inj=laser_table_inj[col].index.to_numpy()*AU.fs
                amp_inj_field=laser_table_inj[col].to_numpy()
                amp_inj_pot=-1*laser_table_inj_pot[col].to_numpy()
                ###### soft window the delay scan #######
                amp_s=-1*scan_tmp[col].to_numpy()
                P_ion_s=P_ion_tmp[col].to_numpy()
                current_shift=""
                if abs(amp_s[-1])>max(abs(amp_s))/100:
                    current_shift=f"-{amp_s[-1]}"
                    amp_s=amp_s-amp_s[-1]
                tau=tau
                tmp=pd.DataFrame({"tau(fs)": tau, "tau(au)": tau_au, "S(tau)": amp_s, "P_ion(tau)": P_ion_s, "V_x(tau)": amp_s/P_ion_s})
                tmp.to_csv(f'png/{file_name}/Current_{type_name.replace("/","_")}.csv')
                P_background=0
                tmp=pd.DataFrame({"tau(fs)": tau, "tau(au)": tau_au, "P_ion(tau)": P_ion_s, "P_ion_perturb":P_ion_s-P_background})
                tmp.to_csv(f'png/{file_name}/TIPTOE_{type_name.replace("/","_")}.csv')#, index=None)
                break
                tmp=pd.DataFrame({"t(fs)": time, "t(au)": time_au, "Field(t)": amp_l_field, "Apot(t)": amp_l})
                tmp.to_csv(f'png/{file_name}/Probe_{type_name.replace("/","_")}.csv')
                tmp=pd.DataFrame({"t(fs)": time_inj, "t(au)": time_au, "Field(t)": amp_inj_field, "Apot(t)": amp_inj_pot})
                tmp.to_csv(f'png/{file_name}/Pump_{type_name.replace("/","_")}.csv')
                if not tau[3]>-2.5:
                    amp_s=amp_s*(soft_window(tau, -2.0, tau[1])*soft_window(tau, 2.0, tau[-2]))#*soft_window(tau, -.8, tau[8])*soft_window(tau, .8, tau[-8])
                S_interpolated=interp1d(tau, amp_s, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)
                tau_interp=np.linspace(tau[0],tau[-1],1000)
                SMax=max(np.abs(S_interpolated(tau_interp)))
                axes[0,2].plot(tau_interp,S_interpolated(tau_interp)/max(np.abs(S_interpolated(tau_interp))), linewidth=0.5, 
                               label=rf'$S(\tau){current_shift}, S_{{max}}={SMax:.2e}$'+f' {nl} folder: {col}')
                axes[0,2].scatter(tau,amp_s/max(np.abs(S_interpolated(tau_interp))), s=1)
                AxS=axes[0,2].twinx()
                AxS.plot(time,amp_l, label=r'$A_{probe}(t)$'+col, alpha=0.5, color='grey')
                axes[0,2].set_ylim(-1.1,1.1)
                #AxS.set_ylim(axes[0,2].get_ylim()[0]*AxS.get_ylim()[0],AxS.get_ylim()[1]/axes[0,2].get_ylim()[0])
                AxS.set_ylim(-1.1*max(abs(amp_l)),1.1*max(abs(amp_l)))
                AxS.grid(False)
                AxS.set(ylabel=r'$A_{probe}(t)$')
                axes[0,2].set(ylabel=r'$S(\tau)$ normalized', xlabel='delay (fs)')
                axes[0,2].legend(loc='upper left')
                V_interpolated=interp1d(tau, -1*amp_s/P_ion_s, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)
                axes[0,3].plot(tau_interp,V_interpolated(tau_interp), linewidth=0.5, label=rf'$V(\tau){current_shift}$'+f' {nl} folder: {col}')
                axes[0,3].scatter(tau,-1*amp_s/P_ion_s, s=1)
                AxV=axes[0,3]#.twinx()
                AxV.plot(time,-1*amp_l, label=r'$-A_{probe}(t)$'+col, color='grey')
                AxV.grid(False)
                #axes[0,3].set_ylim(-0.9*max(abs(amp_s/P_ion_s)),1.1*max(abs(amp_s/P_ion_s)))
                #AxV.set_ylim(-0.9*max(abs(amp_s/P_ion_s)),1.1*max(abs(amp_s/P_ion_s)))
                axes[0,3].set(ylabel=r'$v(\tau)$ (au)', xlabel='time (fs)', xlim=(-5,5))
                #align_yaxis(AxV, 0, axes[0,3], 0)
                axes[0,3].legend()

                 #axes[0,2].set_xticks(tau)
                #axes[0,2].xaxis.set_ticklabels([])
                #axes[0,2].set_xticks()
                ###### get the frequencies for which the function shall return the FT '######### 
                
                spec_tmp, omega =FT(time, amp_l, t0=0)
                spec_tmp2=spec_tmp#FT(time, amp_l_field, omega=omega)
                spec_tmp=np.abs(spec_tmp).flatten()
                omega_mid=(omega[abs(spec_tmp)>np.max(spec_tmp)*0.10])[-1]
                omega=omega[abs(spec_tmp)>np.max(spec_tmp)*0.01]
                omega=omega[omega>0]
                omega_max=omega[-1]
                omega_min=omega[0]
                x=False
                if x:
                    tmp,omega=FT(tau, amp_s)
                    omega=omega[omega>omega_min]
                    omega=omega[omega<omega_max]
                else:
                    omega=np.linspace(omega_min, omega_max,10000)
                

                ####### FFT of the delay scan with the frequencies obtained ###########
                spec= FT(tau, amp_s, omega=omega)
                scan_FT[f'omega_{col}']=omega
                scan_FT[col]=spec

                ####### FFT of the probe vector potential at the calculated cirecular frequncies #########
                spec=FT(time, amp_l, t0=0 , omega=omega)  
                laser_FT[f'omega_{col}']=omega
                laser_FT[col]=spec
                omega=laser_FT[f'omega_{col}'].to_numpy()

                ####### FFT of the Probe Field #########
                amp_Eprobe=1*laser_table_field[col].to_numpy()
                spec_Eprobe=FT(time, amp_Eprobe, omega=omega, t0=0)

                # ####### FFT of the Injection vector potential #########
                # time_inj=(laser_table_inj[col].index.to_numpy())*AU.second*1e15
                # amp_l_inj=laser_table_inj[col].to_numpy()
                # spec_inj, omega_inj=FT(time_inj, amp_l_inj, t0=0) 
                # spec_inj=spec_inj.flatten()
                # omega_inj=omega_inj[spec_inj>np.max(spec_inj)*0.01]
                # omega_inj=omega_inj[omega_inj<6*2*np.pi]
                # omega_inj=omega_inj[omega_inj>=0]
                # omega_inj_max=omega_inj[-1]
                # omega_inj_min=omega_inj[0]
                # omega_inj=np.linspace(omega_inj_min, omega_inj_max,5000)
                # spec_inj=FT(time_inj, amp_l_inj, t0=0, omega=omega_inj)

                ####### plotting real imaginary and absolute values
                X=np.asarray(laser_FT[f'omega_{col}'])/2/np.pi
                Ys=np.asarray(scan_FT[col])
                #plt.xticks(X[:-1:4])
                axes[0,1].plot(X, normalize_only(np.unwrap(np.angle(Ys))), label=f'tRecX, max={max(abs(np.unwrap(np.angle(Ys)))):.2e} {nl} folder: {col}')
                axes[0,1].set(ylabel=r'$\Phi(S,\nu)$')
                axes[0,0].plot(X, normalize_only(np.abs(Ys)), label=f'tRecX, max={max(abs(abs(Ys))):.2e} {nl} folder: {col}')
                axes[0,0].set(ylabel=r'$|S(\nu)|$')
                #plt.show()
                #axes[0,0].legend()
                #axes[0,1].legend()
                axes[0,0].legend()
                Yl=laser_FT[col].to_numpy()
                ax_tmp=axes[0,1].twinx()
                ax_tmp.plot(X, (np.unwrap(np.angle(Yl))), label=f'A_probe, {nl} folder: {col}', color='grey')
                ax_tmp.set(ylabel=r'$\Phi(A_{probe}, \nu)$')
                #ax_tmp.set_ylim(axes[0,1].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[0,1].get_ylim()[1], ax_tmp.get_ylim()[1])
                ax_tmp.grid(False)
                ax_tmp=axes[0,0].twinx()
                ax_tmp.plot(X, (np.abs(Yl)), label=f'A_probe, {nl} folder: {col}', color='grey')
                ax_tmp.set(ylabel=r'|$A_{probe}$|')  
                ax_tmp.set_ylim(axes[0,0].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[0,0].get_ylim()[1], ax_tmp.get_ylim()[1])
                
                ax_tmp.grid(False)


                # Yl_inj=spec_inj
                # Yl_inj=Yl_inj/max(abs(Yl_inj))*max(abs(Yl))
                # X_inj=omega_inj/2/np.pi
                # axes[1,0].plot(X_inj, np.real(Yl_inj), label=f'injection {col} normalised')
                # axes[1,1].plot(X_inj, np.imag(Yl_inj), label=f'injection {col} normalised')
                # axes[1,0].plot(X_inj, np.abs(Yl_inj), label=f'injection {col} normalised')
                # axes[1,0].legend()
                # axes[1,1].legend()
                # axes[1,0].legend()              

                # the conjugatuion due to the definition of the Fourier transform for defined delay time
                Y=np.conjugate(np.divide(Ys,Yl))#*soft_window(X.to_numpy(), 1., 2.0)
                # tmp=pd.DataFrame({"v(PHz)": X, "G(v)": Y})
                # tmp.to_csv(f'png/{file_name}/G_v_{type_name.replace("/","_")}.csv')
                #axes[1,1].plot(X, np.angle(Y), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                axes[1,1].plot(X, normalize_only(np.unwrap(np.angle(Y))), label=f'tRecX, max={max(abs(np.unwrap(np.angle(Y)))):.2e} {nl} folder: {col}')
                axes[1,1].set(ylabel=r'$\Phi$($G(\nu)$)')
                axes[1,0].plot(X, normalize_only(np.abs(Y)), label=f'tRecX, max={max(abs(np.abs(Y))):.2e} {nl} folder: {col}')
                axes[1,0].set(ylabel=r'|$G(\nu)$|')
                axes[1,0].grid()
                axes[1,1].grid()
                axes[1,0].grid()
                nu0=299.79/float(Inj_table[col].to_numpy())#0.399
                max_nu_int=int(min(X[-1],3.29)/nu0)+1
                labels=[]
                for i in  np.arange(0,max_nu_int,1):
                    labels.append(rf"{i}$\nu_0$")
                plt.setp(axes[:,:2], xticks=nu0*np.arange(0,max_nu_int,1),xticklabels=labels, xlabel=rf'$\nu_0$= {nu0:.2f} PHz', xlim=(-1e-3,nu0*max_nu_int))
                #Y=Y*soft_window(X.to_numpy(), omega_mid/2/np.pi, omega_max/2/np.pi)
                #print(Y[0])
                #Y=Y[X>0.1]#*soft_window(X, 3.5, 4.5)#*soft_window(X, 1., 1.5)#*soft_window(X, .5, .4)#*soft_window(X,3, 3.2)#*soft_window(X, 2.9,3.1)#*soft_window(X, 1.3, 1.1)  
                #X=X[X>0.1]#Y=Y*soft_window(X, .8,.25)
                #### worked for low bandwidth pulse soft_window(X, 1., 1.5)###
                Y_imag=np.imag(Y)#*soft_window(X, 1., 2.0)
                Y_real=np.real(Y)#*soft_window(X, 1., 2.0)   
                #Y_imag=np.insert(np.imag(Y)*soft_window(X, 0.25, .005), 0, 0)
                #f_real_extrap=interp1d(X, np.real(Y), fill_value='extrapolate')
                #Y_real=np.insert(np.real(Y),0, f_real_extrap(0))
                X_omega=X*2*np.pi
                #X=np.insert(X,0,0)

                # freq_time_analysis=False
                # if freq_time_analysis:
                #     Y0=Y*soft_window(X, .5, 2.2)*soft_window(X, .2, 0)
                #     del_amp, time =IFT(X,Y0, omega0=.3)#, t_array=time)
                #     scan_IFT[f'time_{col}']=time
                #     scan_IFT[col]=del_amp
                #     axes[2,0].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                #     axes[2,0].set(ylabel=r'G(t) in low frequncy regime', xlabel='time (fs)')

                #     Y1=Y*soft_window(X, 1.3, 2.2)*soft_window(X, .5, 0)
                #     del_amp, time =IFT(X,Y1, omega0=.85)#, t_array=time)
                #     scan_IFT[f'time_{col}']=time
                #     scan_IFT[col]=del_amp
                #     axes[2,1].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                #     axes[2,1].set(ylabel=r'G(t) in middle frequncy regime', xlabel='time (fs)')

                #     Y2=Y*soft_window(X, 1.9, 2.2)*soft_window(X, 1.5, 0)
                #     del_amp, time =IFT(X,Y2, omega0=1.625)#, t_array=time)
                #     scan_IFT[f'time_{col}']=time
                #     scan_IFT[col]=del_amp
                #     axes[2,0].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'tRecX, max={max(abs(place_holder)):.2e} {nl} folder: {col}')
                #     axes[2,0].set(ylabel=r'G(t) in high frequncy regime', xlabel='time (fs)')#, ylim=(0,.5e-17))


                Y3=Y_real+1j*Y_imag   
                
                tau2=np.linspace(tau[0], tau[-1], 50*len(tau))
                #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                del_amp =IFT(X_omega,Y3, t_array=tau2)
                scan_IFT[f'time_{col}']=tau2
                scan_IFT[col]=del_amp
                axes[1,2].plot(scan_IFT[f'time_{col}'].to_numpy(), normalize_only(np.real(scan_IFT[col].to_numpy())), label=f'tRecX, max={max(abs(np.real(scan_IFT[col]))):.2e} {nl} folder: {col}', linewidth=0.5) 
                
            
            # del_amp =IFT(X,Ys.to_numpy(), t_array=tau2)*max(abs(del_amp))/max(abs(amp_s))
            # scan_IFT[col]=del_amp
            # axes[1,2].plot(scan_IFT[f'time_{col}'], 2*np.real(scan_IFT[col]), label=f'S(tau) reconstructed and normalised {col}')
                axes[1,2].set(ylabel=r'G(tau)', xlabel='time (fs)')#, ylim=(0,.5e-17))         
                axes[1,2].legend(fontsize=6)
                scn=scan_table
                #scn.index=scn.index*1e15
                #scn.plot(ax=axes[0,2], xlabel='delay (fs)', ylabel=f'current in direction : {val[-1]}', legend=False)
                #axes[0,2].set_xticks(scn.index[::3])
                #axes[0,2].grid()
                #cen=0#-6.807653999999999e-15
                lg=laser_g.get_group((type_name,center))
                laser_table=lg.pivot(columns=column_group, values=["Apot[1]", "Field[0]", "Field[1]"])
                if len(laser_injec)!=0:
                    laser_injec_table= l_inj.pivot(columns=column_group, values=["Apot[1]", "Field[0]"])
                    laser_table["Apot[1]"]=laser_table["Apot[1]"]#-laser_injec_table["Apot[1]"]
                
                # laser_table["Field[0]"]=laser_table["Field[0]"]/laser_table["Field[0]"].abs().max()
                # laser_table["Apot[1]"]=laser_table["Apot[1]"]/laser_table["Apot[1]"].abs().max()
                lsr=laser_table
                lsr['time(fs)']=lsr.index.values*AU.second*1e15
                ax_tmp=axes[1,2].twinx()
                lsr.plot(x='time(fs)', y=["Field[0]"],ax=ax_tmp,ylabel=rf'$E_{{pump}}(t)$ $\tau$={center:.1f}', legend=False, color='grey')
                ax_tmp.grid(False)
                ax_tmp.set_ylim(axes[1,2].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[1,2].get_ylim()[1], ax_tmp.get_ylim()[1])
                
                
                if(check_recollision):

                    t=51.82#850/AU.nm/AU.speed_of_light/2.3


                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm1.25e+14_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()

                    #%%
                    E_inj1=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm7.35e+13_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()
                    E_inj2=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    # plt.plot(time_inj+t, -1*E_inj_exact)
                    # plt.plot(time_inj-t, -1*E_inj_exact)

                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm1.41e+13_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()
                    E_inj3=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    # plt.plot(time_inj+2*t, 1*E_inj_exact)
                    # plt.plot(time_inj-2*t, 1*E_inj_exact)

                    Field_E_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Pump_injIntWcm7.45e+11_.csv"
                    E_inj_t=pd.read_csv(Field_E_file, index_col=0)
                    time_inj=E_inj_t['t(au)'].to_numpy()
                    E_inj_exact=E_inj_t['Field(t)'].to_numpy()
                    E_inj4=interp1d(time_inj, E_inj_exact, kind='cubic', fill_value=(0, 0), bounds_error=False)
                    # plt.plot(time_inj+3*t, -1*E_inj_exact)
                    # plt.plot(time_inj-3*t, -1*E_inj_exact)
                    #%%
                    time_inj=np.linspace(time_inj[0], 6*t, 100000)

                    ax_tmp.plot(time_inj*AU.fs, E_inj1(time_inj)-E_inj2(time_inj+t)-E_inj2(time_inj-t)
                                +E_inj3(time_inj+2*t)+E_inj3(time_inj-2*t)-E_inj4(time_inj+3*t)
                                -E_inj4(time_inj-3*t), label='reconstructed', color='grey')
                    ax_tmp.grid(False)
                    # %%

                    Ion_prob=0
                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm1.25e+14_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current1=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)

                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm7.35e+13_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+2*np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current2=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)#

                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm1.41e+13_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+2*np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current3=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)

                    S_tRecX_file=f"png/10_May/DC/{drive_wave}nm_{phase}_Drive/Current_injIntWcm7.45e+11_.csv"
                    tRecX_current=pd.read_csv(S_tRecX_file, index_col=0)
                    S_tRecX=tRecX_current["S(tau)"]
                    V_tRecX=tRecX_current['V_x(tau)']
                    Ion_prob=Ion_prob+2*np.mean(tRecX_current['P_ion(tau)'])
                    # S_tRecX=S_tRecX-S_tRecX[0]
                    # S_tRecX=S_tRecX#/0.00565
                    t_tRecX=tRecX_current["tau(au)"]
                    current4=interp1d(t_tRecX, S_tRecX, kind='cubic', fill_value=(0, 0), bounds_error=False)

                    t_tRecX=np.linspace(-3*t, 3*t, 1000)
                    #plt.plot(t_tRecX,current1(t_tRecX), label='only central half cycle', linestyle='dotted')
                    current_recon=current1(t_tRecX)-current2(t_tRecX+t)-current2(t_tRecX-t)+current3(t_tRecX+2*t)+current3(t_tRecX-2*t)-current4(t_tRecX+3*t)-current4(t_tRecX-3*t)
                    print(Ion_prob)
                    print(np.mean(P_ion_s))
                    Ion_prob=np.mean(P_ion_s)
                    axes[0,2].plot(t_tRecX*AU.fs, normalize_only(current_recon), label=f'recon. half-cycles, max={max(abs(current_recon)):.2e}')
                    V_reconstruct=-1/Ion_prob*(current1(t_tRecX)-current2(t_tRecX+t)-current2(t_tRecX-t)+current3(t_tRecX+2*t)+current3(t_tRecX-2*t)-current4(t_tRecX+3*t)-current4(t_tRecX-3*t))
                    axes[0,3].plot(t_tRecX*AU.fs, V_reconstruct, label='recon. half-cycles')
                    Del_V_recollision=V_interpolated(t_tRecX*AU.fs)-V_reconstruct
                    axes[2,3].plot(t_tRecX*AU.fs, Del_V_recollision, label='Recollision')
                    ax_tmp=axes[2,3].twinx()
                    ax_tmp.plot(time, normalize_only(amp_Eprobe), label=r'E_{probe}(t)', color='grey', alpha=0.7)
                    ax_tmp.plot(time, normalize_only(amp_inj_field), label=r'E_{inj}(t)', color='grey', alpha=0.3)
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)
                    ax_tmp=axes[2,3]
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)
                    ax_tmp.set_xlim(-3,3)
                    axes[2,3].set_xlabel(r'\tau (fs)')
                    axes[2,3].set_ylabel(r'$\Delta V^{recol.}$ (a.u.)')
                    spec_Del_V_recollision=FT(t_tRecX*AU.fs, Del_V_recollision, omega=omega)
                    axes[2,0].plot(X, abs(spec_Del_V_recollision), label='Recollision')
                    #axes[2,0].set_ylabel(r'$|\Delta V|$ (a.u.)')
                    axes[2,1].plot(X, np.unwrap(np.angle(spec_Del_V_recollision)), label='Recollision')
                    #axes[2,1].set_ylabel(r'$\phi(\Delta V)$ (rad)')


                    spec_f=np.conjugate(np.divide(spec_Del_V_recollision,spec_Eprobe))
                    axes[3,0].plot(X, np.abs(spec_f), label=f'Recollision{nl} folder: {col}')
                    axes[3,0].set(ylabel=r'|f^{recol.}|')
                    axes[3,1].plot(X, np.unwrap(np.angle(spec_f)), label=f'Recollision{nl} folder: {col}')
                    axes[3,1].set(ylabel=r'$\Phi(f^{recol.})$')
                    tau2=np.linspace(tau[0], tau[-1], 500*len(tau))
                    #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                    del_amp =IFT(omega,spec_f, t_array=tau2)
                    delV_IFT[f'time_{col}']=tau2
                    delV_IFT[col]=del_amp
                    axes[3,3].plot(delV_IFT[f'time_{col}'], np.real(delV_IFT[col]), label=f'Recollision {nl} folder: {col}', linewidth=0.5)
                    axes[3,3].set(ylabel=r'$f^{recol.}(t)$')
                    axes[3,3].set(xlabel=r'$time (fs)$')

                #plt.setp(axes[:,2], xticks=np.arange(-6, 7, 1), xlim=(-6.5, 6.5))
                # ep=expec_g.get_group((type_name,center))
                # SRn=S_Rn_g.get_group((type_name,center))
                # SRn_table=SRn.drop_duplicates().pivot(columns=column_group, values=["<Overlap>"])
                # # try:
                # #     ep_table=ep.pivot(columns=column_group, values=["<Ovr(Phi.Eta.Rn.Rn)>"])
                # # except:
                # #     ep_table=ep.pivot(columns=column_group, values=["<Ovr(Orbital&Phi.Phi.Eta.Rn.Rn)>"])
                # # #ep_table=ep_table.apply(lambda x: 1-x)
                # # ep_table['time(fs)']=ep_table.index.values*AU.second*1e15
                # SRn_table['time(fs)']=SRn_table.index.values*AU.second*1e15
                # SRn_table.plot(x='time(fs)', ax=axes[1,3], ylabel='density of free electrons', xlim=(tau[0],tau[-1]), legend=False, color='grey')#, logy=True)
                if os.path.exists(f'png/{file_name}/SFA'):
                    df_tmp=pd.read_csv(f'png/{file_name}/SFA', index_col=0)
                    Gamma=(df_tmp['G_gamma']+df_tmp['G_T_2']+df_tmp['G_T_3']+df_tmp['G_dip']*0).to_numpy()
                    S_SFA=[]
                    for delay in tau_au:
                        ApotX=np.interp(df_tmp.index.to_numpy()+delay, time_au, amp_l)
                        S_SFA.append(trapz(Gamma*ApotX, df_tmp.index.to_numpy()))
                    
                    del_v=-1*(amp_s/P_ion_s-S_SFA/np.sum(Gamma))


                    spec_v=FT(tau, del_v, omega=omega)
                    axes[2,1].plot(X, np.unwrap(np.angle(spec_v)), label=f'tRecX-SFA, {nl} folder: {col}')
                    axes[2,1].set(ylabel=r'$ \Phi(\Delta v, \nu)$')
                    axes[2,0].plot(X, np.abs(spec_v), label=f'tRecX-SFA,  {nl} folder: {col}')
                    axes[2,0].set(ylabel=r'$|\Delta v(\nu)|$')
                    del_v=interp1d(tau, del_v, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)(tau_interp)
                    axes[2,2].plot(tau_interp, del_v, label=f'tRecX-SFA, {nl} folder: {col}')
                    axes[2,2].set(ylabel=r'$\Delta v(\tau)$')
                    axes[2,2].set(xlabel=r'$\tau (fs)$')

                    spec_f=np.conjugate(np.divide(spec_v,spec_Eprobe))
                    axes[3,0].plot(X, np.abs(spec_f), label=f'{nl} folder: {col}')
                    axes[3,0].set(ylabel=r'|f|')
                    axes[3,1].plot(X, np.unwrap(np.angle(spec_f)), label=f'{nl} folder: {col}')
                    axes[3,1].set(ylabel=r'$\Phi(f)$')
                    tau2=np.linspace(tau[0], tau[-1], 500*len(tau))
                    #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                    del_amp =IFT(omega,spec_f, t_array=tau2)
                    delV_IFT[f'time_{col}']=tau2
                    delV_IFT[col]=del_amp
                    axes[3,2].plot(delV_IFT[f'time_{col}'], np.real(delV_IFT[col]), label=f'{nl} folder: {col}', linewidth=0.5)
                    axes[3,2].set(ylabel=r'$f(t)$')
                    axes[3,2].set(xlabel=r'$time (fs)$')

                    if(check_recollision):
                        del_v=-1*(interp1d(t_tRecX, V_reconstruct)(tau_au)-S_SFA/np.sum(Gamma))
                        ax22=axes[0,3]#.twinx()
                        spec_v=FT(tau, del_v, omega=omega)
                        axes[2,1].plot(X, np.unwrap(np.angle(spec_v)), label=f'half-SFA')
                        axes[2,1].set(ylabel=r'$ \Phi(\Delta v, \nu)$')
                        axes[2,0].plot(X, np.abs(spec_v), label=f'half-SFA,')
                        axes[2,0].set(ylabel=r'$|\Delta v(\nu)|$')
                        del_v=interp1d(tau, del_v, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)(tau_interp)
                        axes[2,2].plot(tau_interp, del_v, label=f'half-SFA')
                        axes[2,2].set(ylabel=r'$\Delta v(\tau)$')
                        axes[2,2].set(xlabel=r'$\tau (fs)$')


                        spec_f=np.conjugate(np.divide(spec_v,spec_Eprobe))
                        axes[3,0].plot(X, np.abs(spec_f), label=f'half')
                        axes[3,0].set(ylabel=r'|f|')
                        axes[3,1].plot(X, np.unwrap(np.angle(spec_f)), label=f'half')
                        axes[3,1].set(ylabel=r'$\Phi(f)$')
                        tau2=np.linspace(tau[0], tau[-1], 500*len(tau))
                        #del_amp, tau2 =IFT(X,Y3, omega0=0)            
                        del_amp =IFT(omega,spec_f, t_array=tau2)
                        delV_IFT[f'time_{col}']=tau2
                        delV_IFT[col]=del_amp
                        axes[3,2].plot(delV_IFT[f'time_{col}'], np.real(delV_IFT[col]), label=f'half', linewidth=0.5)
                        axes[3,2].set(ylabel=r'$f(t)$')
                        axes[3,2].set(xlabel=r'$time (fs)$')


                    ax22=axes[0,3]#.twinx()
                    S_SFA=interp1d(tau, S_SFA, kind='cubic', fill_value=(0,0), assume_sorted=True, bounds_error=False)(tau_interp)
                    ax22.plot(tau_interp,-1*S_SFA/np.sum(Gamma), label=f'SFA', color='red', linestyle=':', alpha=0.5)

                    


                    axes[1,3].plot(df_tmp.index.to_numpy()*AU.fs, np.cumsum(Gamma), label='SFA Ionization', color='red')
                    G_spec= FT(df_tmp.index.to_numpy()*AU.fs, Gamma, omega=omega, t0=0)
                    ang= np.unwrap(np.angle(G_spec))
                    G_om=abs(G_spec)



                    S_SFA, S_SFA_max=normalize(S_SFA)

                    ax22=axes[0,2]#.twinx()
                    ax22.plot(tau_interp,S_SFA, label=f'SFA, max={S_SFA_max:.2e}', color='red', linestyle=':', alpha=0.5)
                    Gamma, Gamma_max=normalize(Gamma)
                    G_om, G_om_max=normalize(G_om)
                    ang, ang_max=normalize(ang)
                    ax22=axes[1,2]#.twinx()
                    ax22.plot(df_tmp.index.to_numpy()*AU.fs, Gamma, label=f'SFA, G_max={Gamma_max:.2e}', color='red')
                    #ax22.grid(False)
                    ax22=axes[1,1]#.twinx()
                    ax22.plot(omega/2/np.pi,ang, label=f'SFA, max={ang_max:.2e}', color='red')
                    #ax22.grid(False)
                    ax22=axes[1,0]#.twinx()
                    ax22.plot(omega/2/np.pi,G_om, label=f'SFA, max={G_om_max:.2e}', color='red')
                    Current_SFA=G_spec*Yl
                    C_abs, C_abs_max=normalize(abs(Current_SFA))
                    C_angle, C_angle_max=normalize(np.unwrap(np.angle(Current_SFA)))
                    ax22=axes[0,0]#.twinx()
                    ax22.plot(omega/2/np.pi,C_abs, label=f'SFA, max={C_abs_max:.2e}', color='red')
                    ax22=axes[0,1]#.twinx()
                    ax22.plot(omega/2/np.pi,C_angle, label=f'SFA, max={C_angle_max:.2e}', color='red')


                    ax_tmp=axes[2,2].twinx()
                    ax_tmp.plot(time, -amp_Eprobe, color='grey')
                    ax_tmp.set_ylabel(rf'-$E_{{probe}}(t)$')
                    #ax_tmp.set_ylim(axes[2,2].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[2,2].get_ylim()[1], ax_tmp.get_ylim()[1])
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)
                    ax_tmp.grid(False)
                    ax_tmp=axes[2,2]
                    tmp=max(ax_tmp.get_ylim()[1],ax_tmp.get_ylim()[0], key=abs)
                    ax_tmp.set_ylim(-1.1*tmp,1.1*tmp)


                    ax_tmp=axes[2,0].twinx()
                    ax_tmp.plot(X, abs(spec_Eprobe), color='grey')
                    ax_tmp.set_ylabel(rf'$E_{{probe}}(\nu)$')
                    ax_tmp.set_ylim(axes[2,0].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[2,0].get_ylim()[1], ax_tmp.get_ylim()[1])
                    ax_tmp.grid(False)


                    ax_tmp=axes[2,1].twinx()
                    ax_tmp.plot(X, np.unwrap(np.angle(spec_Eprobe)), color='grey')
                    ax_tmp.set_ylabel(rf'$\phi (E_{{probe}},\nu)$')
                    #ax_tmp.set_ylim(axes[2,1].get_ylim()[0]*ax_tmp.get_ylim()[1]/axes[2,1].get_ylim()[1], ax_tmp.get_ylim()[1])
                    ax_tmp.grid(False)

                axes[1,2].get_shared_x_axes().join(axes[0,2], axes[1,2])
                axes[1,2].get_shared_x_axes().join(axes[0,2], axes[1,2])
                axes[1,2].get_shared_x_axes().join(axes[0,2], axes[1,2])
                axes[1,2].set_xlim(tau[0]-.5,tau[-1]+.5)
                axes[1,3].get_shared_x_axes().join(axes[0,3], axes[1,3])
                axes[1,3].get_shared_x_axes().join(axes[0,3], axes[1,3])
                axes[1,3].get_shared_x_axes().join(axes[0,3], axes[1,3])
                axes[1,3].set_xlim(tau[0]-.5,tau[-1]+.5)
                axes[1,1].grid(True)
                #axes[1,3].grid(True)
                handles, labels = AxV.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                AxV.legend(by_label.values(), by_label.keys(), loc='lower right')
                plt.setp(axes[:,2], xlim=(tau[0]-.5,tau[-1]+.5))
                # handles, labels = AxS.get_legend_handles_labels()
                # by_label = dict(zip(labels, handles))
                # AxS.legend(by_label.values(), by_label.keys(), loc='lower right')
                # AxS.set_ylabel(r'$A_{probe}$')
                #AxV.set_ylabel(r'$A_{probe}$')
                for i in range(nrows):
                    for j in range(ncols):
                        axes[i,j].legend()
                plt.tight_layout()
                # align_yaxis(AxS, 0, axes[0,2], 0)
                # align_yaxis(AxV, 0, axes[0,3], 0)
            pdf.savefig()
            #plt.show()
            plt.close()
            pdf.close()    
    return 0


def Gt_LPS_analytical_au(time, amplitude):
        E=amplitude
        dt=(time[-1]-time[0])/len(time)/4
        grid=np.arange(time[0], time[-1], dt)
        t_min=time[0]
        E=griddata(time, E, grid, method='cubic')
        time=grid
        Ip=0.5 # au
        print('ionisation potential (au): ', Ip)
        envelope, omega0=get_complex_envelope(E, dt, t_min)
        print('predicted central frequency (au): ',omega0,'=',omega0*AU.eV,'eV','=',1230/(omega0*AU.eV),'nm')
        G_t= (2**6)/3*np.square(abs(envelope))*np.power(omega0-Ip, 1.5)*(Ip**2.5)/omega0**6
        return time*AU.second*1e15, G_t

if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        file_name="Grid_Determination_30au"#input('Enter the name of the h5 file: ')
        file_name="Injection_wavelength"
        file_name="Intensity_60au"
    else:
        file_name=sys.argv[1]
    if '.h5' in file_name:
        file_name=file_name.replace('.h5', '')
    file=f'h5_small/{file_name}.h5'
    with pd.HDFStore(file, 'r') as store:
        scan, expec, laser, spec, S_Rn, expec_injection=[],[],[],[],[], []
        Injection_param, Drive_param= [],[]
        group_of_interest='/' 
        for (path, subgroups, subkeys) in store.walk(where=group_of_interest):
            for subgroup in subgroups:
                print("GROUP: {}/{}".format(path, subgroup))
            for subkey in subkeys:
                key = "/".join([path, subkey])
                df=store.get(key)
                df['type']=os.path.dirname(os.path.dirname(key.replace("/background", "").replace("/Injection_param", "")))
                df['name']=os.path.basename(os.path.dirname(key.replace("/background", "").replace("/Injection_param", "")))
                if "background" in key:
                    if 'expec' in key and 'S_Rn' not in key:
                        expec_injection.append(df)
                    continue
                elif "injection" in key:
                    if 'expec' in key and 'S_Rn' not in key:
                        expec_injection.append(df)
                    continue
                if  "Injection_param" in key:
                    Injection_param.append(df)
                elif  "Drive_param" in key:
                    Drive_param.append(df)
                elif "S_Rn" in key:
                    S_Rn.append(df)
                elif 'spec' in key:
                    spec.append(df)
                elif 'Laser' in key:
                    laser.append(df)
                elif 'expec' in key and 'S_Rn' not in key:
                    expec.append(df)
                elif 'delay_scan' in key:
                    scan.append(df)
    os.makedirs(f'png/{file_name}/', exist_ok=True)
    # plot_laser(laser, "Field", file_name)
    # plot_laser(laser, "Apot", file_name)
    # plot_current(scan, file_name, val='Kx')
    # plot_FT(scan, laser, expec, Injection_param, Drive_param, file_name,column_group='type',main_group='name')
    plot_FT(scan, laser, expec, Injection_param, Drive_param, file_name,column_group='name',main_group='type')
    #plot_expec_video(laser, expec, Injection_param, Drive_param, file_name, expec_injection=expec_injection)



