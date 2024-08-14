import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv

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

def write_csv_prob(filename, delay, ion_y, ion_QS, ion_NA):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['delay', 'ion_y', 'ion_QS', 'ion_NA'])
        for i in range(len(delay)):
            writer.writerow([delay[i], ion_y[i], ion_QS[i], ion_NA[i]])

def plot_ion_4(ion_QS, ion_y, ion_na, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, phase_add, x_lim_ion_yield):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))




    ax1.plot(delay*AU.fs, ion_y, label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax1.plot(delay*AU.fs, ion_QS, label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax1.plot(delay*AU.fs, ion_na, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')

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

    ax2_2 = ax2.twinx()
    ax2_2.plot(time*AU.fs, -field_probe_fourier_time, label='Probe Field', linestyle='--', color='gray')
    ax2_2.set_ylabel('Probe Field')
    ax2_2.legend(loc='lower left')
    ax2.plot(delay*AU.fs, ion_y, label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    # ax2.plot(delay*AU.fs, ion_QS/max(abs(ion_QS))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}\cdot${max(abs(ion_y))/max(abs(ion_QS)):.2f}')
    # ax2.plot(delay*AU.fs, ion_na/max(abs(ion_na))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}\cdot${max(abs(ion_y))/max(abs(ion_na)):.2f}')
    ax2.plot(delay*AU.fs, ion_QS, label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax2.plot(delay*AU.fs, ion_na, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')
    ax2.set_xlabel('Delay (fs)')
    ax2.set_ylabel('Ionization Yield')
    ax2.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
    ax2.legend()
    ax2.set_title('Normalized Ionization Yield')
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



    ion_QS_resp=ion_QS_fourier/field_probe_fourier
    ion_y_resp=ion_y_fourier/field_probe_fourier
    ion_nonAdiabatic_resp=ion_nonAdiabatic_fourier/field_probe_fourier




    ax3.plot(omega/2/np.pi, np.abs(ion_y_resp), label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$/{np.abs(ion_y_resp).max():.2f}')
    ax3.plot(omega/2/np.pi, np.abs(ion_QS_resp), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$/{np.abs(ion_QS_resp).max():.2f}') 
    ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_resp), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$/{np.abs(ion_nonAdiabatic_resp).max():.2f}')


    ax3.set_xlabel('Frequency (PHz)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Spectral Response Absolute Value')
    ax3.legend()


    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_y_resp))+phase_add, label=rf'$\mathrm{{P}}_\mathrm{{tRecX}}$')
    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_QS_resp)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
    ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_resp)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabatic}}$')

    ax4.set_xlabel('Frequency (PHz)')
    ax4.set_ylabel('Phase')
    ax4.set_title('Spectral Response Phase')
    ax4.legend(loc='lower left')
    ax4.annotate(f'$\lambda_\mathrm{{Pump}}={lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={FWHM_probe}\mathrm{{fs}}$', xy=(0.31, 0.16), xycoords='axes fraction', fontsize=9, ha='left', va='center')



    
    plt.tight_layout()

    pdf_filename = f'/home/user/TIPTOE/plot_ion_pdf/plot_ion_{lam0_pump}_{lam0_probe}_{I_pump:.2e}_{I_probe:.2e}.pdf'
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)

    plt.show()
    plt.close()


file_params = [
    ("850nm_350nm_1.25e+14", 850, 1.25e14, 350, 1e10, 0.93, 0, 5),
    ("850nm_350nm_7.5e+13", 850, 7.50e13, 350, 6.00e09, 0.93, 0, 5),
    ("900nm_320nm_5e+14", 900, 5e14, 320, 4e10, 0.75, 0, 5),
    ("1200nm_320nm_1e+14", 1200, 1e14, 320, 4e10, 0.75, 0, 5),
    ("1200nm_320nm_5e+14", 1200, 5e14, 320, 4e10, 0.75, 0, 5)
]


params={'Multiplier': 0.3537672860419956, 'Ip': 0.5, 'αPol': 4.51, 'gamma': 2.2225332077079094, 'e0': 3.7538886827392335, 'a0': 1, 'a1': 1.0, 'b0': 1.0, 'b1': 1, 'b2': 1, 'c2': 2.445215663819714, 'p1': 3.5, 'd1': 0.22496851712539684}
gamma=params['gamma']
c2=-16/9*params['b2']*params['c2']
Multiplier,  e0, a0, a1, b0, b1, b2, p1, d1, Ip, αPol= params['Multiplier'], gamma/2*params['e0'], 4*gamma*params['a0'], params['a1'], 4*gamma**2*params['b0'], 2*gamma*params['b1']*params['a1'], params['b2'], params['p1'], params['d1'], params['Ip'], params['αPol']
params={'Multiplier': Multiplier, 'E_g': Ip, 'αPol': αPol, 'e0': e0, 'a0': a0, 'a1': a1, 'b0': b0, 'b1': b1, 'b2': b2, 'p1': p1, 'd1': d1, 'c2': c2}
params_qs = {'Multiplier': Multiplier, 'E_g': Ip, 'αPol': αPol, 'e0': e0, 'a0': a0, 'a1': a1, 'b0': b0, 'b1': b1, 'b2': b2, 'p1': p1, 'd1': d1}

REDO_comp = False
for file_name, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep, x_lim_ion_yield in file_params:

    if REDO_comp :
        laser_pulses = LaserField(cache_results=True)
        delay, ion_y = read_ion_Prob_data(f"/home/user/TIPTOE/process_all_files_output/ionProb_{file_name}.csv")
        ion_qs = []
        ion_na = []
        for tau in delay:
            laser_pulses.add_pulse(lam0_pump, I_pump, 0, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
            laser_pulses.add_pulse(lam0_probe, I_probe, -np.pi/2, FWHM_probe/AtomicUnits.fs, t0=-tau)
            t_min, t_max = laser_pulses.get_time_interval()
            time=np.arange(t_min, t_max+1, 1)
            ion_qs.append(1-np.exp(-np.trapz(analyticalRate(time, laser_pulses, **params_qs), time)))
            ion_na.append(1-np.exp(-IonProb(laser_pulses, **params, dt=2, dT=0.25)))
            ion_na_rate = IonRate(time, laser_pulses, **params, dT=0.25)
            laser_pulses.reset()
        output_file = f"/home/user/TIPTOE/plot_ion_tau_calc_output_data/ion_prob_{file_name}.csv"
        write_csv_prob(output_file, delay, ion_y, ion_qs, ion_na)

    data_rate_delay = data_time_field = pd.read_csv(f"/home/user/TIPTOE/plot_ion_tau_calc_output_data/ion_prob_{file_name}.csv")
    delay=np.array(data_rate_delay['delay'].values)
    ion_QS=np.array(data_rate_delay['ion_QS'].values)
    ion_na=np.array(data_rate_delay['ion_NA'].values)
    ion_y=np.array(data_rate_delay['ion_y'].values)


    probe=LaserField()
    probe.add_pulse(lam0_probe, I_probe, CEP=-np.pi/2, FWHM=FWHM_probe/AtomicUnits.fs)
    tmin, tmax=probe.get_time_interval()
    time=np.arange(tmin, tmax+1, 1.)
    field_probe_fourier_time=probe.Electric_Field(time)

    plot_ion_4(ion_QS, ion_y, ion_na, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep, x_lim_ion_yield)