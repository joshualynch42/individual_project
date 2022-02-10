import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

key_coords = pd.read_csv('D:\Josh\github\individual_project\simulation_data\key_coords.csv')

# Y is the independent axis
first_line_x = key_coords.iloc[0:10]['X'].to_numpy()
first_line_y = key_coords.iloc[0:10]['Y'].to_numpy()

x = key_coords.iloc[0:26]['X'].to_numpy()
y = key_coords.iloc[0:26]['Y'].to_numpy()

result = np.polyfit(first_line_y, first_line_x, 2)

y_result = np.linspace(np.amin(first_line_y), np.amax(first_line_y), len(first_line_y))
x_result = y_result*result[0] + result[1]

plt.plot(first_line_y, first_line_x, '-r')
plt.plot(y_result, x_result, '-g')
plt.show()

# error calc

print(first_line_x)
print(x_result)

mse = ((x_result-first_line_x)**2).mean()
print(mse)
