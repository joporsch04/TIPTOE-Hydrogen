#!/usr/bin/env python3
#%%
import numpy as np
#from matplotlib import pyplot as plt
import pandas as pd
import os   
#import hdf5py|
import glob
#import time
#from matplotlib.backends.backend_pdf import PdfPages
#from __init__ import FourierTransform as FT
#from __init__ import AtomicUnits as AU
from plot_hdf5 import plot_current, plot_laser, plot_FT, plot_FT_tiptoe
import plot_hdf5
#import matplotlib.pyplot as plt

show_plots = False
print_plots = True

font_size = 11
legend_font_size = 9

#plt.rcParams.update({'font.size': font_size})


BOX=""
#file_name=f'Grid_Determination_{BOX}au'
#file=f'hdf5_small/Grid_Determination_{BOX}au.hdf5'
#file_name=f'Convergence_{BOX}au'
#file=f'hdf5_small/Convergence_{BOX}{Filter}au.hdf5'
#file_name=f'SFA_2.3e14'
#file=f'hdf5_small/SFA_2.3e14{Filter}.hdf5'

Filter=""#"phicoe3_"#"au40"
#with hdf5py.File('mytestfile.hdf5', 'w') as hdf5:
import warnings
warnings.filterwarnings("ignore")
#%%
def produce_hdf5_file(file_name):
    file=f'hdf5_small/{file_name.replace("/","_")}.hdf5'
    print(f"data from {file_name} is being stored in {file}")
    with pd.HDFStore(file, 'w') as store:
        error_list=[]
        reason_list=[]
        background_drive={}
        pure_injection={}
        background_folder=''
        print(f"file_name: {file_name}")
        injection_folder=''
        for folder in glob.glob(f'{file_name}/*{Filter}*/'):
            print("root directory of the scan for which data is being fetched:", folder)
            delay_array,unit_array,kx,ky,kz,e_dens=[],[],[],[],[],[]
            Injection={}
            Drive={}
            # a list containing file names that store information in gnuplot format
            group = ['Laser', 'expec', 'S_Rn/expec', 'spec_total']
            for gr in group:
                gr=gr.replace('/','_')
                locals()[gr]=[] 
                locals()['background/'+gr]=[]
                locals()['Injection_back/'+gr]=[]
            #Laser, expec, S_Rn_expec, df_list, spec_total = [],[],[],[],[]
            for scan in glob.glob(f'{folder}/[0-9]*/'):
                try:
                    with open(scan+'err', 'r') as f:
                        one_char = f.read(1)
                        # if fetched then file is not empty
                        if one_char:
                            print(f.read())
                            error_list.append(scan+"inpc")
                            reason_list.append(f.read())
                            continue
                except:
                    try:
                        with open(os.path.dirname(scan)+'_err', 'r') as f:   
                            one_char = f.read(1)
                            # if fetched then file is not empty
                            if one_char:
                                print(f.read())
                                error_list.append(scan+"inpc")
                                reason_list.append(f.read())
                                continue
                    except:
                        pass
                with open(scan+"inpc",'r') as f:
                    lines=f.readlines()
                    for index, line in enumerate(lines):
                        if "Background - effects of drive field independent of the injection pulse" in line:
                            background_drive['delay(injection-test)']=[0]
                            with open(scan+"outspec",'r') as f: 
                                for line in f.readlines():
                                    if '<1><1><GridWeight>' in line:
                                        background_drive['e_dens']=float(line.split()[1])
                                    elif '<1><Q><GridWeight*Q>' in line:
                                        background_drive['kz']=float(line.split()[1])
                                    elif '<cos(Q)><sqrt(1-Q*Q)><GridWeight*Q>' in line:
                                        background_drive['kx']=float(line.split()[1])
                                    elif '<sin(Q)><sqrt(1-Q*Q)><GridWeight*Q>' in line:
                                        background_drive['ky']=float(line.split()[1])
                            store.append(folder+"background/delay_scan",(pd.DataFrame(background_drive)))                      
                            for gr in group:
                                file2=gr.replace('/','_')
                                file_dir = f"{scan}{gr}"
                                file2='background/'+file2
                                with open(file_dir, 'r') as f:
                                    comment=True
                                    for i in range(20):
                                        line=f.readline()
                                        if line.find('#') <0:
                                            headers = line_prev.replace('#','').replace('\n','').replace(',',' ').split() 
                                            if "spec_total" in file_dir: headers= ["kRn: sum[Phi,Eta,specRn]","density"]
                                            break
                                        line_prev=line
                                    df = pd.read_csv(file_dir, comment='#', delim_whitespace=True, names=headers, index_col=0)
                                    locals()[file2].append(df)           
                            background_folder=scan
                            break
                        if "Effects of injection pulse" in line:
                            pure_injection['delay(injection-test)']=[0]
                            try:
                                with open(scan+"outspec",'r') as f: 
                                    for line in f.readlines():
                                        if '<1><1><GridWeight>' in line:
                                            pure_injection['e_dens']=float(line.split()[1])
                                        elif '<1><Q><GridWeight*Q>' in line:
                                            pure_injection['kz']=float(line.split()[1])
                                        elif '<cos(Q)><sqrt(1-Q*Q)><GridWeight*Q>' in line:
                                            pure_injection['kx']=float(line.split()[1])
                                        elif '<sin(Q)><sqrt(1-Q*Q)><GridWeight*Q>' in line:
                                            pure_injection['ky']=float(line.split()[1])
                                            store.append(folder+"Injection_back/delay_scan",(pd.DataFrame(pure_injection)))                      
                                for gr in group:
                                    file2=gr.replace('/','_')
                                    file_dir = f"{scan}{gr}"
                                    file2='Injection_back/'+file2
                                    with open(file_dir, 'r') as f:
                                        comment=True
                                        for i in range(20):
                                            line=f.readline()
                                            if line.find('#') <0:
                                                headers = line_prev.replace('#','').replace('\n','').replace(',',' ').split() 
                                                if "spec_total" in file_dir: headers= ["kRn: sum[Phi,Eta,specRn]","density"]
                                                break
                                            line_prev=line
                                        df = pd.read_csv(file_dir, comment='#', delim_whitespace=True, names=headers, index_col=0)
                                        locals()[file2].append(df)           
                                injection_folder=scan
                                break
                            except:
                                print("input file for drive independent injection present but thhere were ererors in the simulation")
                                break
                            
                        
                        if "Laser:" in line:
                            drive_row=2 # 2 if no HH
                            peak_index=line.replace(','," ").split()[1:].index('peak')
                            peak_inject=float(lines[index+1].split(',')[peak_index].split()[0])
                            # inject_peak_unit=lines[index+1].split(',')[peak_index].split()[1]
                            # if inject_peak_unit=='OptCyc':

                            delay=float(lines[index+1].split(',')[peak_index].split()[0])-float(lines[index+drive_row].split(',')[peak_index].split()[0])
                            unit = lines[index+1].split(',')[peak_index].split()[1]
                            delay_array.append(delay)
                            unit_array.append(unit)
                            if not Drive:
                                intensity_index=line.replace(','," ").split()[1:].index('I(W/cm2)')
                                wavelength_index=line.replace(','," ").split()[1:].index('lambda(nm)')
                                fwhm_index=line.replace(','," ").split()[1:].index('FWHM')
                                try:
                                    Injection['Int']=[float(lines[index+1].split(',')[intensity_index].split()[0])]
                                except:
                                    Injection['Int']=[float(lines[index+1].split(',')[intensity_index].split()[0].split("*")[0])]
                                    #drive_row=3
                                Injection['wavelength']=[float(lines[index+1].split(',')[wavelength_index].split()[0])]
                                Injection['fwhm']=[float(lines[index+1].split(',')[fwhm_index].split()[0])]
                                Drive['Int']=[float(lines[index+drive_row].split(',')[intensity_index].split()[0])]
                                Drive['wavelength']=[float(lines[index+drive_row].split(',')[wavelength_index].split()[0])]
                                Drive['fwhm']=[float(lines[index+drive_row].split(',')[fwhm_index].split()[0].split('fs')[0])]
                                store.append(folder+"Injection_param", pd.DataFrame(Injection))
                                store.append(folder+"Drive_param", pd.DataFrame(Drive))
                            
                if scan!=background_folder and scan!=injection_folder:
                    try:
                        with open(scan+"outspec",'r') as f: 
                            for line in f.readlines():
                                if '<1><1><GridWeight>' in line:
                                    e_dens.append(float(line.split()[1]))
                                elif '<1><Q><GridWeight*Q>' in line:
                                    kz.append(float(line.split()[1]))
                                elif '<cos(Q)><sqrt(1-Q*Q)><GridWeight*Q>' in line:
                                    kx.append(float(line.split()[1]))
                                elif '<sin(Q)><sqrt(1-Q*Q)><GridWeight*Q>' in line:
                                    ky.append(float(line.split()[1]))
                    except:
                        e_dens.append(0.)
                        kz.append(0.)
                        kx.append(0.)
                        ky.append(0.)
                    for gr in group:
                        #if delay!=0: break
                        file2=gr.replace('/','_')
                        file_dir = f"{scan}{gr}"
                        try:
                            with open(file_dir, 'r') as f:
                                comment=True
                                for i in range(20):
                                    line=f.readline()
                                    if line.find('#') <0:
                                        headers = line_prev.replace('#','').replace('\n','').replace(',',' ').split() 
                                        if "spec_total" in file_dir: headers= ["kRn: sum[Phi,Eta,specRn]","density"]
                                        break
                                    line_prev=line
                                df = pd.read_csv(file_dir, comment='#', delim_whitespace=True, names=headers, index_col=0)
                                df['delay(injection-test)']=delay#scan.split('/')[-2]
                                locals()[file2].append(df)
                        except:
                            continue
                            
            for gr in group:
                gr=gr.replace('/','_')
                store.append(folder+gr, pd.concat(locals()[gr]))
                #gr='background/'+gr
                try:
                    store.append(folder+'background/'+gr, pd.concat(locals()['background/'+gr]))
                except:
                    print(f'this scan does not contain a simulation with only the drive pulse for {gr}!')
                #gr='injection/'+gr
                try:
                    store.append(folder+'Injection_back/'+gr, pd.concat(locals()['Injection_back/'+gr]))
                except:
                    print(f'this scan does not contain any simulation with only the injection pulse for {gr}!')
            # data = np.array([delay_array, kz, kx, ky, e_dens, unit_array])
            # data = data[:,np.argsort(data[0,:])]
            # store.append(folder+"delay_scan",(pd.DataFrame({'delay(injection-test)': data[0], 'Kz':data[1], 'Kx': data[2], 'Ky':data[3], 'e_density': data[4], 'unit': data[5]}).set_index('delay(injection-test)'))) 
            data = np.array([delay_array, kz, kx, ky, e_dens])
            data = data[:,np.argsort(data[0,:])]
            store.append(folder+"delay_scan",(pd.DataFrame({'delay(injection-test)': data[0], 'Kz':data[1], 'Kx': data[2], 'Ky':data[3], 'e_density': data[4]}).set_index('delay(injection-test)'))) 
        store.append('errors', pd.DataFrame({'failed_job_file_names':error_list, 'reason': reason_list}))
        if not error_list:
            print("All submitted jobs in this directory were completed succesfully!")
        else:
            print("the following runs could not be completed succefully :\n",store.get('errors'))
    warnings.filterwarnings("default")



#%%
import sys
REDO=False
#REDO=True
if __name__ == "__main__":
    file_name="DC_19April_noTrunc/350nm_Drive/N5"
    file_name="10_May/AC"
    file_name="850nm"
    #file_name="10_May/DC/700nm_sin_Drive"
    #file_name="10_May/DC/350nm_sin_Drive"
    #file_name="850nm"
    # file_name="85"
    name_hdf5=file_name.replace('/', '_')
    file=f'hdf5_small/{name_hdf5}.hdf5'
    if not os.path.exists(file) or REDO:
        produce_hdf5_file(file_name)
    #assert(os.path.exists(file_name+"/"))
    #produce_hdf5_file(file_name)
    with pd.HDFStore(file, 'r') as store:
        scan, expec, laser, spec, S_Rn=[],[],[],[],[]
        scan_drive, expec_drive, laser_drive, spec_drive, S_Rn_drive=[],[],[],[],[]
        scan_injec, expec_injec, laser_injec, spec_injec, S_Rn_injec=[],[],[],[],[]
        Injection_param, Drive_param, Injection_param_elliptic= [],[], []
        group_of_interest='/' 
        for (path, subgroups, subkeys) in store.walk(where=group_of_interest):
            for subgroup in subgroups:
                print(f"GROUP: {path}/{subgroup}")
            for subkey in subkeys:
                key = "/".join([path, subkey])
                df=store.get(key)
                df['type']=os.path.dirname(os.path.dirname(key.replace("/background", "").replace("/Injection_back", "").replace("/injection", "")))
                df['name']=os.path.basename(os.path.dirname(key.replace("/background", "").replace("/Injection_back", "").replace("/injection", "")))
                if "/background" in key:
                    if "S_Rn" in key:
                        S_Rn_drive.append(df)
                    elif 'spec' in key:
                        spec_drive.append(df)
                    elif 'Laser' in key:
                        laser_drive.append(df)
                    elif 'expec' in key and 'S_Rn' not in key:
                        expec_drive.append(df)
                    elif 'delay_scan' in key:
                        scan_drive.append(df)
                    continue
                elif "/Injection_back" in key or "/injection" in key:
                    if "S_Rn" in key:
                        S_Rn_injec.append(df)
                    elif 'spec' in key:
                        spec_injec.append(df)
                    elif 'Laser' in key:
                        laser_injec.append(df)
                    elif 'expec' in key and 'S_Rn' not in key:
                        expec_injec.append(df)
                    elif 'delay_scan' in key:
                        scan_injec.append(df)
                    continue

                if "Injection_param_elliptic" in key:
                    Injection_param_elliptic.append(df)
                elif  "Injection_param" in key:
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
    #plot_laser(laser, "Field", file_name)
    #plot_laser(laser, "Apot", file_name)
    #plot_current(scan, Injection_param, Drive_param, file_name, val='Kx')
    #%%
    #mport importlib
    # importlib.reload(sys.modules['plot_hdf5'])
    # from plot_hdf5 import plot_FT
    plot_FT_tiptoe(scan, laser, laser_injec,expec_injec, expec, S_Rn, Injection_param, Drive_param, file_name,column_group='type',main_group='name')
    #plot_FT(scan, laser, laser_injec, expec, S_Rn, Injection_param, Drive_param, file_name,column_group='name',main_group='type')
    # plot_expec_video(laser, expec, Injection_param, Drive_param, file_name)#, expec_injection=expec_injection)
# %%
