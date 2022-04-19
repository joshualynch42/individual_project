import pandas as pd
import os, os.path
import numpy as np

key_coords = pd.read_csv(r"D:\Josh\github\individual_project\simulation\simulation_data\key_coords.csv")
path = "D:/Josh/github/individual_project/simulation/simulation_data/key_images/"

all_lines = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
            'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M',
            'SPACE', 'LEFT', 'UP', 'DOWN', 'RIGHT']

counter = 0
total = 0

for letter in all_lines:
    row = key_coords.loc[key_coords['Key'] == letter]
    x = row['X'].to_numpy()[0]
    y = row['Y'].to_numpy()[0]

    new_path = path + letter + '/'

    for file in os.listdir(new_path):
        temp_path = new_path + file
        file = file.replace('_0.png', '').replace('--37.0', '').replace('--36.0', '').replace('--35.0', '')
        file = file.split('_')
        x_file = float(file[0])
        y_file = float(file[1])

        total += 1

        if x_file < x - 1.0 or x_file > x + 1.0:
            counter += 1
            os.remove(temp_path)
        elif y_file < y - 1.0 or y_file > y + 1.0:
            counter += 1
            os.remove(temp_path)

print('removed {} out of {} files'.format(counter, total))
