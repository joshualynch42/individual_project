import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

key_coords = pd.read_csv(r"D:\Josh\github\individual_project\simulation\simulation_data\key_coords.csv")

x, y = key_coords['X'], key_coords['Y']
x = x.to_numpy()
y = y.to_numpy()

im_x, im_y = key_coords['IM_X'], key_coords['IM_Y']
im_x = im_x.to_numpy()
im_y = im_y.to_numpy()

# line1 = key_coords.iloc[0:10]
# line2 = key_coords.iloc[10:19]
# line3 = pd.concat([key_coords.iloc[19:26], key_coords.iloc[28:29]])
# line4 = pd.concat([key_coords.iloc[26:28], key_coords.iloc[29:31]])

# #calculating average x coordinate
# mean_x1 = [line1['X'].mean()]*len(line1) # ~ 147.6
# mean_x2 = [line2['X'].mean()]*len(line2) # ~ 166.4
# mean_x3 = [line3['X'].mean()]*len(line3) # ~ 186.0
# mean_x4 = [line4['X'].mean()]*len(line4) # ~ 205.8

# mean_line_x1 = [147.6]*len(line1)
# mean_line_x2 = [166.4]*len(line2)
# mean_line_x3 = [186]*len(line3)
# mean_line_x4 = [205.8]*len(line4)
#
# plt.plot(y, x, 'ro')
plt.plot(im_y, im_x, 'ro')

#plotting the average x coordinate for each line
# plt.plot(line1['Y'], mean_line_x1, 'bo')
# plt.plot(line2['Y'], mean_line_x2, 'bo')
# plt.plot(line3['Y'], mean_line_x3, 'bo')
# plt.plot(line4['Y'], mean_line_x4, 'bo')

#plotting the furthest points taken during data collection
# plt.plot(y-2, x-2, 'go')
# plt.plot(y+2, x-2, 'go')
# plt.plot(y-2, x+2, 'go')
# plt.plot(y+2, x+2, 'go')
#
# plt.xlabel('Y')
# plt.ylabel('X')
plt.show()
