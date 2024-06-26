import os, os.path
import csv
import pandas as pd
import numpy as np


def extractField_0(file_path):
    with open(file_path, 'r') as file:
        fifth_colomn = []
        for row, line in enumerate(file):
            if row < 6:
                continue
            line = line.strip()
            colomns = line.split()
            if len(colomns) >= 5:
                fifth_colomn.append(float(colomns[5]))
    return fifth_colomn

def extract_time(file_path):
    with open(file_path, 'r') as file:
        first_colomn = []
        for row, line in enumerate(file):
            if row < 6:
                continue
            line = line.strip()
            colomns = line.split()
            if len(colomns) >= 5:
                first_colomn.append(float(colomns[0]))
    return first_colomn


def process_all_laser_files(base_dir, delay_files_path, file_name_interpol_output, file_name_ionProb_output, file_name_field_0_output):

    count_files = [eintrag for eintrag in os.listdir(delay_files_path) if os.path.isdir(os.path.join(delay_files_path, eintrag)) and eintrag.isdigit()]
    files_number = max([int(eintrag) for eintrag in count_files]) # If the folders are numbered in ascending order, this is the highest number

    all_data = []

    IonProbDict = {}
    IonProbDict['index']=np.asarray(['probabilities', 'delays'])
    dicField={}
    time_array=np.arange(-1000, 1000+0.25, 0.25)
    dicField['t_au']=time_array
    for i in range(0, files_number+1):
        dir_path = os.path.join(delay_files_path, str(i)) #dir_path = os.path.join(base_dir, '850nm', '350_nm_Drive_dense', str(i))
        file_path = os.path.join(dir_path, 'Laser')
        file_path2= os.path.join(dir_path, 'outspec')
        if os.path.isfile(file_path2):
            time_column=extract_time(file_path)
            fifth_column = extractField_0(file_path)
            all_data.append(fifth_column)
            x=time_column
            y=fifth_column
            Y_interpolated = np.interp(time_array, x, y)
            dicField[f'Field_{i}'] = Y_interpolated
            with open(file_path2,'r') as f: 
                for line in f.readlines():
                    if '<1><1><GridWeight>' in line:
                        IonProb=float(line.split()[1])
            with open(os.path.join(dir_path, 'inpc'),'r') as f: 
                lines=f.readlines()
                for index, line in enumerate(lines):
                    if 'Laser:' in line:
                        peak_index=line.replace(','," ").split()[1:].index('peak')
                        peak_inject=float(lines[index+1].split(',')[peak_index].split()[0])
                        try:
                            delay=float(lines[index+1].split(',')[peak_index].split()[0])-float(lines[index+2].split(',')[peak_index].split()[0])
                        except:
                             delay=float(lines[index+1].split(',')[peak_index].split()[0])
                        unit = lines[index+1].split(',')[peak_index].split()[1]  
            IonProbDict[f'Field_{i}']=[IonProb, delay]
        else:
            print(f"File not found: {file_path}")
            all_data.append([])
    pd.DataFrame(dicField).to_csv(file_name_interpol_output)

    pd.DataFrame(IonProbDict).to_csv(file_name_ionProb_output)

    max_rows = max(len(data) for data in all_data)
    max_cols = len(all_data)

    sorted_data = []
    for row_index in range(max_rows):
        row = []
        for col_index in range(max_cols):
            if row_index < len(all_data[col_index]):
                row.append(all_data[col_index][row_index])
            else:
                row.append('')
        sorted_data.append(row)

    with open(file_name_field_0_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([i for i in range(1, 37)] + [41])

        for row_index, row_data in enumerate(sorted_data):
            writer.writerow([row_index + 1] + row_data)