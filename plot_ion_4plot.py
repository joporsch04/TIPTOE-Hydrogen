import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from __init__ import FourierTransform, soft_window
from kernel import IonProb, IonRate, analyticalRate
from field_functions import LaserField


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


def read_ion_Prob_data(file_path):
    data = pd.read_csv(file_path, header=None)
    delay = pd.to_numeric(data.iloc[2].iloc[2:].values)[:-1]
    ion_y = pd.to_numeric(data.iloc[1].iloc[2:].values)[:-1]
    return delay, ion_y

def write_csv_prob(filename, delay, ion_y, ion_QS, ion_NA, ion_NA_reconstructed):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['delay', 'ion_y', 'ion_QS', 'ion_NA', 'ion_NA_reconstructed'])
        for i in range(len(delay)):
            writer.writerow([delay[i], ion_y[i], ion_QS[i], ion_NA[i], ion_NA_reconstructed[i]])

def plot_ion_4(ion_QS, ion_y, ion_na, ion_na_reconstructed, nArate, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    x_lim_ion_yield = 5
    phase_add = 0


    ax1.plot(delay*AU.fs, ion_y, label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax1.plot(delay*AU.fs, ion_QS, label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax1.plot(delay*AU.fs, ion_na, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax1.plot(delay*AU.fs, ion_na_reconstructed, label=rf'$\mathrm{{P}}_\mathrm{{nonAdRecon}}$')

    ax1.set_ylabel('Ionization Yield')
    ax1.set_xlabel('Delay (fs)')
    ax1.set_title('Ionization Yield with background')
    ax1.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
    ax1.legend(loc='center left')
    ax1.annotate(f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', xy=(0.72, 0.4), xycoords='axes fraction', fontsize=9, ha='left', va='center')
    
    ion_na=ion_na-ion_na[-1]
    ion_QS=ion_QS-ion_QS[-1] 
    ion_y=ion_y-ion_y[-1]
    ion_y=(ion_y[::-1]*soft_window(delay[::-1]*AU.fs, 4,5)*soft_window(delay[::-1]*AU.fs, -4,-5))[::-1]
    ion_na_reconstructed=ion_na_reconstructed-ion_na_reconstructed[-1]

    #ax2_2 = ax2.twinx()
    #ax2_2.plot(time*AU.fs, -field_probe_fourier_time, label='Probe Field', linestyle='--', color='gray')
    #ax2_2.set_ylabel('Probe Field')
    #ax2_2.legend(loc='lower left')

    ax2.plot(delay*AU.fs, ion_y, label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    # ax2.plot(delay*AU.fs, ion_QS/max(abs(ion_QS))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}\cdot${max(abs(ion_y))/max(abs(ion_QS)):.2f}')
    # ax2.plot(delay*AU.fs, ion_na/max(abs(ion_na))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}\cdot${max(abs(ion_y))/max(abs(ion_na)):.2f}')
    ax2.plot(delay*AU.fs, ion_QS, label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax2.plot(delay*AU.fs, ion_na, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax2.plot(delay*AU.fs, ion_na_reconstructed, label=rf'$\mathrm{{P}}_\mathrm{{nonAdRecon}}$')
    ax2.set_xlabel('Delay (fs)')
    ax2.set_ylabel('Ionization Yield')
    ax2.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
    ax2.legend()
    ax2.set_title('Ionization Yield without Background')
    ax2.annotate(f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', xy=(0.72, 0.16), xycoords='axes fraction', fontsize=9, ha='left', va='center')

    
    field_probe_fourier, omega = FourierTransform(time*AU.fs, field_probe_fourier_time, t0=0)
    field_probe_fourier=field_probe_fourier.flatten()
    omega=omega[abs(field_probe_fourier)>max(abs(field_probe_fourier))*0.05]
    omega=omega[omega>0]
    omega=np.linspace(omega[0], omega[-1], 5000)
    field_probe_fourier = FourierTransform(time*AU.fs, field_probe_fourier_time, omega, t0=0)



    ion_QS_fourier = FourierTransform(delay[::-1]*AU.fs, ion_QS[::-1], omega, t0=0)
    ion_y_fourier = FourierTransform(delay[::-1]*AU.fs, ion_y[::-1], omega, t0=0)
    ion_nonAdiabatic_fourier = FourierTransform(delay[::-1]*AU.fs, ion_na[::-1], omega, t0=0)
    ion_nonAdiabatic_reconstructed_fourier = FourierTransform(delay[::-1]*AU.fs, ion_na_reconstructed[::-1], omega, t0=0)



    ion_QS_resp=ion_QS_fourier/field_probe_fourier
    ion_y_resp=ion_y_fourier/field_probe_fourier
    ion_nonAdiabatic_resp=ion_nonAdiabatic_fourier/field_probe_fourier
    ion_nonAdiabatic_reconstructed_resp=ion_nonAdiabatic_reconstructed_fourier/field_probe_fourier
    ion_nonAdiabatic_reponse_full=FourierTransform(nArate[0]*AU.fs, nArate[1], omega, t0=0)
    # plt.close()
    # plt.plot(nArate[0]*AU.fs, nArate[1])
    # plt.plot(nArate[0]*AU.fs, nArate[2])
    # plt.show()
    # plt.close()
    # plt.close()
    # plt.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_reponse_full))
    # plt.plot(omega/2/np.pi, np.abs(FourierTransform(nArate[0]*AU.fs, nArate[2], omega, t0=0)))
    # plt.show()
    # plt.close()



    ax3.plot(omega/2/np.pi, np.abs(ion_y_resp), label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax3.plot(omega/2/np.pi, np.abs(ion_QS_resp), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_resp), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_reconstructed_resp), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticRecon}}$')
    ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_reponse_full), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticReconFull}}$')


    ax3.set_xlabel('Frequency (PHz)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Spectral Response Absolute Value')
    ax3.legend()


    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_y_resp)), label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_QS_resp)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_resp)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_reconstructed_resp)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdRecon}}$')

    ax4.set_xlabel('Frequency (PHz)')
    ax4.set_ylabel('Phase')
    ax4.set_title('Spectral Response Phase')
    ax4.legend(loc='lower left')
    ax4.annotate(f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', xy=(0.71, 0.16), xycoords='axes fraction', fontsize=9, ha='left', va='center')



    
    plt.tight_layout()

    pdf_filename = f'/home/user/TIPTOE-Hydrogen/plot_ion_pdf/plot_ion_{lam0_pump}_{lam0_probe}_{I_pump:.2e}_{I_probe:.2e}.pdf'
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
    
    #print(f"done {lam0_pump}_{lam0_probe}_{I_pump:.2e}_{I_probe:.2e}")

    # plt.show()
    # plt.close()

    ax3.clear()
    ax4.clear()





    ax3.plot(omega/2/np.pi, np.real(ion_y_resp), label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax3.plot(omega/2/np.pi, np.real(ion_QS_resp), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$') 
    ax3.plot(omega/2/np.pi, np.real(ion_nonAdiabatic_resp), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax3.plot(omega/2/np.pi, np.real(ion_nonAdiabatic_reconstructed_resp), label=rf'$\mathrm{{P}}_\mathrm{{nonAdRecon}}$')


    ax3.set_xlabel('Frequency (PHz)')
    ax3.set_ylabel('Real Respose')
    ax3.set_title('Spectral Response Real Value')
    ax3.legend()


    ax4.plot(omega/2/np.pi, (np.imag(ion_y_resp)), label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax4.plot(omega/2/np.pi, (np.imag(ion_QS_resp)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax4.plot(omega/2/np.pi, (np.imag(ion_nonAdiabatic_resp)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax4.plot(omega/2/np.pi, (np.imag(ion_nonAdiabatic_reconstructed_resp)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdRecon}}$')

    ax4.set_xlabel('Frequency (PHz)')
    ax4.set_ylabel('Imaginary Response')
    ax4.set_title('Spectral Response Imaginary Value')
    ax4.legend(loc='lower left')
    ax4.annotate(f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', xy=(0.7, 0.5), xycoords='axes fraction', fontsize=9, ha='left', va='center')

    plt.tight_layout()

    pdf_filename = f'/home/user/TIPTOE-Hydrogen/plot_ion_pdf_reIm/plot_ion_{lam0_pump}_{lam0_probe}_{I_pump:.2e}_{I_probe:.2e}.pdf'
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
    
    # plt.show()
    # plt.close()



def plot_ion_4_plotly(ion_QS, ion_y, ion_na, ion_na_reconstructed, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe):
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Ionization Yield with background', 'Ionization Yield without Background', 'Spectral Response Absolute Value', 'Spectral Response Phase'))

    
    x_lim_ion_yield = 5
    phase_add = 0
    
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_y, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$'), row=1, col=1)
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_QS, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{QS}}$'), row=1, col=1)
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_na, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$'), row=1, col=1)
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_na_reconstructed, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}\mathrm{{Reconstructed}}$'), row=1, col=1)
    #fig.add_annotation(x=0.72, y=0.4, text=f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', showarrow=False, row=1, col=1)

    ion_na=ion_na-ion_na[-1]
    ion_QS=ion_QS-ion_QS[-1]
    ion_y=ion_y-ion_y[-1]
    ion_y=(ion_y[::-1]*soft_window(delay[::-1]*AU.fs, 4,5)*soft_window(delay[::-1]*AU.fs, -4,-5))[::-1]
    ion_na_reconstructed=ion_na_reconstructed-ion_na_reconstructed[-1]


    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_y, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$', color='red'), row=1, col=2)
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_QS, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{QS}}$', color='grey'), row=1, col=2)
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_na, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$', color='blue'), row=1, col=2)
    fig.add_trace(go.Scatter(x=delay*AU.fs, y=ion_na_reconstructed, mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}\mathrm{{Reconstructed}}$', color='green'), row=1, col=2)
    #fig.add_annotation(x=0.72, y=0.16, text=f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', showarrow=False, row=1, col=2)


    field_probe_fourier, omega = FourierTransform(time*AU.fs, field_probe_fourier_time, t0=0)
    field_probe_fourier=field_probe_fourier.flatten()
    omega=omega[abs(field_probe_fourier)>max(abs(field_probe_fourier))*0.05]
    omega=omega[omega>0]
    omega=np.linspace(omega[0], omega[-1], 5000)
    field_probe_fourier = FourierTransform(time*AU.fs, field_probe_fourier_time, omega, t0=0)

    ion_QS_fourier = FourierTransform(delay[::-1]*AU.fs, ion_QS[::-1], omega, t0=0)
    ion_y_fourier = FourierTransform(delay[::-1]*AU.fs, ion_y[::-1], omega, t0=0)
    ion_nonAdiabatic_fourier = FourierTransform(delay[::-1]*AU.fs, ion_na[::-1], omega, t0=0)
    ion_nonAdiabatic_reconstructed_fourier = FourierTransform(delay[::-1]*AU.fs, ion_na_reconstructed[::-1], omega, t0=0)

    ion_QS_resp=ion_QS_fourier/field_probe_fourier
    ion_y_resp=ion_y_fourier/field_probe_fourier
    ion_nonAdiabatic_resp=ion_nonAdiabatic_fourier/field_probe_fourier
    ion_nonAdiabatic_reconstructed_resp=ion_nonAdiabatic_reconstructed_fourier/field_probe_fourier


    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.abs(ion_y_resp), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$/{np.abs(ion_y_resp).max():.2f}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.abs(ion_QS_resp), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{QS}}$/{np.abs(ion_QS_resp).max():.2f}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.abs(ion_nonAdiabatic_resp), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$/{np.abs(ion_nonAdiabatic_resp).max():.2f}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.abs(ion_nonAdiabatic_reconstructed_resp), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}\mathrm{{Reconstructed}}$/{np.abs(ion_nonAdiabatic_reconstructed_resp).max():.2f}'), row=2, col=1)
    fig.update_layout(showlegend=False)


    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.unwrap(np.angle(ion_y_resp)), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$', color='red'), row=2, col=2)
    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.unwrap(np.angle(ion_QS_resp)), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{QS}}$', color='grey'), row=2, col=2)
    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.unwrap(np.angle(ion_nonAdiabatic_resp)), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$', color='blue'), row=2, col=2)
    fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.unwrap(np.angle(ion_nonAdiabatic_reconstructed_resp)), mode='lines', name=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}\mathrm{{Reconstructed}}$', color='green'), row=2, col=2)


    fig.update_layout(title_text="Side By Side Subplots", width=1920, height=1080)
    fig.show()
    # fig.close()





file_params = [
    ("850nm_350nm_1.25e+14", 850, 1.25e14, 350, 1e10, 0.93, 0, -np.pi/2),
    ("850nm_350nm_7.5e+13", 850, 7.50e13, 350, 6.00e09, 0.93, 0, -np.pi/2),
    ("900nm_320nm_5e+14", 900, 5e14, 320, 4e10, 0.75, 0, -np.pi/2),
    ("1200nm_320nm_1e+14", 1200, 1e14, 320, 4e10, 0.75, 0, -np.pi/2),
    ("900nm_250nm_8e+13", 900, 8e13, 250, 6e8, 0.58, 0, -np.pi/2),
    ("900nm_250nm_9e+13", 900, 9e13, 250, 6e8, 0.58, 0, -np.pi/2),
    ("900nm_250nm_1e+14", 900, 1e14, 250, 6e8, 0.58, 0, -np.pi/2),
    ("900nm_250nm_1.1e+14", 900, 1.1e14, 250, 6e8, 0.58, 0, -np.pi/2),
]

params = {'Multiplier': 0.20433986962624848, 'Ip': 0.5, 'αPol': 0.0, 'gamma': 3.0, 'e0': 1.878222261763161, 'a0': 1, 'a1': 2.4652434578177242, 'b0': 0.0, 'b1': 0.0, 'b2': 8.284076304163763, 'p1': 4.5, 'd_par': 0.9050141234871912, 'd1': 0.1508356872478652, 'c2': 1.7833025057465495}

gamma=params['gamma']
c2=-16/9*params['b2']*params['c2']
Multiplier,  e0, a0, a1, b0, b1, b2, p1, d1, Ip, αPol= params['Multiplier'], gamma/2*params['e0'], 4*gamma*params['a0'], params['a1'], 4*gamma**2*params['b0'], 2*gamma*params['b1']*params['a1'], params['b2'], params['p1'], params['d1'], params['Ip'], params['αPol']
params={'Multiplier': Multiplier, 'E_g': Ip, 'αPol': αPol, 'e0': e0, 'a0': a0, 'a1': a1, 'b0': b0, 'b1': b1, 'b2': b2, 'p1': p1, 'd1': d1, 'c2': c2}
params_qs = {'Multiplier': Multiplier, 'E_g': Ip, 'αPol': αPol, 'e0': e0, 'a0': a0, 'a1': a1, 'b0': b0, 'b1': b1, 'b2': b2, 'p1': p1, 'd1': d1}


REDO_comp = False
for file_name, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep_pump, cep_probe in file_params:
    #lam0_probe = 200
    if REDO_comp:
        laser_pulses = LaserField(cache_results=True)
        delay, ion_y = read_ion_Prob_data(f"/home/user/TIPTOE-Hydrogen/process_all_files_output/ionProb_{file_name}.csv")
        ion_qs = []
        ion_na = []
        ion_na_reconstructed = []
        laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
        t_min, t_max = laser_pulses.get_time_interval()
        time_recon= np.arange(t_min, t_max+1, 1)
        dt_dE=1/np.gradient(laser_pulses.Electric_Field(time_recon),time_recon)
        ion_na_rate = IonRate(time_recon, laser_pulses, **params, dT=0.25/16)
        na_background=np.trapz(ion_na_rate, time_recon)
        na_grad=np.gradient(ion_na_rate, laser_pulses.Electric_Field(time_recon))
        na_grad2=np.gradient(na_grad, laser_pulses.Electric_Field(time_recon))
        laser_pulses.reset()
        for tau in delay:
            laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
            laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
            t_min, t_max = laser_pulses.get_time_interval()
            time=np.arange(t_min, t_max+1, 1)
            ion_qs.append(1-np.exp(-np.trapz(analyticalRate(time, laser_pulses, **params_qs), time)))
            ion_na.append(1-np.exp(-IonProb(laser_pulses, **params, dt=2, dT=0.25)))
            laser_pulses.reset()
            laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
            ion_na_reconstructed.append(1-np.exp(-na_background-np.trapz(na_grad*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
            laser_pulses.reset()
        output_file = f"/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ion_prob_{file_name}.csv"
        write_csv_prob(output_file, delay, ion_y, ion_qs, ion_na, ion_na_reconstructed)

    data_rate_delay = pd.read_csv(f"/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ion_prob_{file_name}.csv")
    delay=np.array(data_rate_delay['delay'].values)
    ion_y=np.array(data_rate_delay['ion_y'].values)
    ion_na=np.array(data_rate_delay['ion_NA'].values)
    ion_QS=np.array(data_rate_delay['ion_QS'].values)


    laser_pulses = LaserField(cache_results=True)
    laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
    t_min, t_max = laser_pulses.get_time_interval()
    time_recon= np.arange(t_min, t_max+1, 1)
    dt_dE=1/np.gradient(laser_pulses.Electric_Field(time_recon),time_recon)
    ion_na_rate = IonRate(time_recon, laser_pulses, **params, dT=0.25/16)
    ion_qs_rate=analyticalRate(time_recon, laser_pulses, **params_qs)
    na_grad=np.gradient(ion_na_rate, laser_pulses.Electric_Field(time_recon))
    qs_grad=np.gradient(ion_qs_rate, laser_pulses.Electric_Field(time_recon))

    try:
        ion_na_reconstructed=np.array(data_rate_delay['ion_NA_reconstructed'].values)
    except:
        ion_na_reconstructed = []
        na_background=np.trapz(ion_na_rate, time_recon)
        na_grad=np.gradient(ion_na_rate, laser_pulses.Electric_Field(time_recon))
        for tau in delay:
            laser_pulses.reset()
            laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
            ion_na_reconstructed.append(1-np.exp(-na_background-np.trapz(na_grad*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
            laser_pulses.reset()
        data_rate_delay['ion_NA_reconstructed']=np.array(ion_na_reconstructed)
        ion_na_reconstructed = np.array(data_rate_delay['ion_NA_reconstructed'].values)
        output_file = f"/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ion_prob_{file_name}.csv"
        write_csv_prob(output_file, delay, ion_y, ion_QS, ion_na, ion_na_reconstructed)


    probe=LaserField()
    probe.add_pulse(lam0_probe, I_probe, CEP=-np.pi/2, FWHM=FWHM_probe/AtomicUnits.fs)
    tmin, tmax=probe.get_time_interval()
    time=np.arange(tmin, tmax+1, 1.)
    field_probe_fourier_time=probe.Electric_Field(time)
    nArate=[time_recon, na_grad, qs_grad]
    plot_ion_4(ion_QS, ion_y, ion_na, ion_na_reconstructed, nArate, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe)

    #plot_ion_4_plotly(ion_QS, ion_y, ion_na, ion_na_reconstructed, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe)