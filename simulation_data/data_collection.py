from cri.robot import SyncRobot
from cri.controller import MagicianController as Controller
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera
from vsp.processor import CameraStreamProcessor, AsyncProcessor
import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
import pandas as pd
import string

alphabet_list = list(string.ascii_lowercase)
print(alphabet_list)
key_coords = pd.read_csv('D:\Josh\github\individual_project\simulation_data\key_coords.csv')

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
    robot.linear_speed = 10
    robot.coord_frame = [0, 0, 0, 0, 0, 0] # careful

    #for current_letter in alphabet_list:

    # get coords for current letter
    current_letter = alphabet_list[0] #temp
    x, y = df.loc[current_letter]['X'], df.loc[current_letter]['Y']

    robot.move_linear([x, y, 0, 0, 0, 0])
    frames_sync = sensor.process(num_frames=5, start_frame=1, outfile=r'C:/Temp/frames_sync.png')
    #frames_sync = sensor.process(num_frames=5, start_frame=1, outfile=r'D:\Josh\github\individual_project\simulation_data\{}_frames.png'.format(current_letter))

# display results
import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
for i, frame in enumerate(frames_sync):
    plt.title(f'Frame {i+1}')
    plt.imshow(frame, cmap="Greys")
    plt.xticks([]); plt.yticks([]);
    plt.pause(0.1)
plt.show()
