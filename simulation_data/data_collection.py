from cri.robot import SyncRobot
from cri.controller import MagicianController as Controller
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera
from vsp.processor import CameraStreamProcessor, AsyncProcessor
import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
import pandas as pd
import string
import time
import numpy as np

alphabet_list = list(string.ascii_uppercase)
key_coords = pd.read_csv(r"D:\Users\Josh\github\individual_project\simulation_data\key_coords.csv")

line1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
line2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
line3 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
line4 = ['SPACE', 'LEFT', 'UP', 'DOWN', 'RIGHT']


def make_sensor(): # amcap: reset all settings; autoexposure off; saturdation max
    camera = CvPreprocVideoCamera(source=1,
                crop=[320-128-10, 240-128+10, 320+128-10, 240+128+10],
                size=[128, 128],
                threshold=[61, -5],
                exposure=-6)
    for _ in range(5): camera.read() # Hack - camera transient

    return AsyncProcessor(CameraStreamProcessor(camera=camera,
                display=CvVideoDisplay(name='sensor'),
                writer=CvImageOutputFileSeq()))

# Move robot and collect data
with SyncRobot(Controller()) as robot, make_sensor() as sensor:
    robot.linear_speed = 40
    robot.coord_frame = [0, 0, 0, 0, 0, 0] # careful

    #z is universal
    z = -35

    for current_letter in line4:
        x, y = key_coords.loc[key_coords['Key'] == current_letter]['X'], key_coords.loc[key_coords['Key'] == current_letter]['Y']
        x = x.to_numpy()[0]
        y = y.to_numpy()[0]
        print('current letter is {}'.format(current_letter))
        # add variation to x, y and z coords for training
        for x_diff in np.linspace(-2, 2, 5):
            new_x = x + x_diff
            for y_diff in np.linspace(-2, 2, 5):
                new_y = y + y_diff
                for z_diff in np.linspace(-2, 0, 3):
                    new_z = z + z_diff
                    varaaa = r"D:\Users\Josh\github\individual_project\simulation_data\key_images\{}\{}_{}-{}.png".format(current_letter,new_x,new_y,new_z)
                    robot.move_linear([new_x, new_y, -20, 0, 0, 0]) #move horizontal
                    robot.move_linear([new_x, new_y, new_z, 0, 0, 0]) #move vertical - down
                    frames_sync = sensor.process(num_frames=1, start_frame=1, outfile=varaaa)
                    robot.move_linear([new_x, new_y, -20, 0, 0, 0]) #move vertical - up
                    time.sleep(1)
    #frames_sync = sensor.process(num_frames=5, start_frame=1, outfile=r'D:\Josh\github\individual_project\simulation_data\{}_frames.png'.format(current_letter))

    #set robot to a Home
    robot.move_linear([168, -80, 0, 0, 0, 0])
# display results
# import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
# for i, frame in enumerate(frames_sync):
#     plt.title(f'Frame {i+1}')
#     plt.imshow(frame, cmap="Greys")
#     plt.xticks([]); plt.yticks([]);
#     plt.pause(0.1)
# plt.show()
