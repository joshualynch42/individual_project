# directly tests use of api
# based on dobot test code 'ControlDobot'
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera
from vsp.processor import CameraStreamProcessor, AsyncProcessor
from cri.dobot.magician import DobotDllType as dobot
import os

#Path to dependent dlls
dll_path = r'D:\Users\Josh\github\individual_project\cri\dobot\magician'
# os.environ["PATH"] += os.pathsep + os.pathsep.join([dll_path])

#Load Dll and get the CDLL object
os.chdir(dll_path)
api = dobot.load()

#Connect Dobot
state = dobot.ConnectDobot(api, "", 115200)[0]
print("Connect status:", state)

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

if state == 0:

    #Clear Command Queued
    dobot.SetQueuedCmdClear(api)

    #Async Motion Params Setting
    dobot.SetHOMEParams(api, *[200,]*4, isQueued = 1)
    print('Home Parameters',dobot.GetHOMEParams(api))
    dobot.SetPTPJointParams(api, *[100,]*8, isQueued = 1)
    print('PTP Joint Paramters',dobot.GetPTPJointParams(api))
    dobot.SetPTPCommonParams(api, *[100,]*2, isQueued = 1)
    print('PTP Common Parameters',dobot.GetPTPCommonParams(api))

    # moving to these coords
    dobot.SetPTPCmd(api, dobot.PTPMode.PTPMOVJXYZMode,
                        200, 0, 0, 0, isQueued = 1)

    last_index = dobot.SetPTPCmd(api, dobot.PTPMode.PTPMOVJXYZMode,
                        180, 30, 30, 20, isQueued = 1)[0]

    print(dobot.GetQueuedCmdCurrentIndex(api)[0])
    #Execute Command Queue
    dobot.SetQueuedCmdStartExec(api)

    while last_index > dobot.GetQueuedCmdCurrentIndex(api)[0]:
        dobot.dSleep(100)
        print(dobot.GetQueuedCmdCurrentIndex(api)[0])

    dobot.SetQueuedCmdStopExec(api)

    # display results
    import matplotlib.pylab as plt # needs to be after initialising controller (strange bug)
    for i, frame in enumerate(frames_async):
        plt.title(f'Frame {i+1}')
        plt.imshow(frame, cmap="Greys")
        plt.xticks([]); plt.yticks([]);
        plt.pause(0.1)
    plt.show()

#Disconnect Dobot
dobot.DisconnectDobot(api)
