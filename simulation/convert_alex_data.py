import cv2
import pandas as pd

file_loc = r"D:\Josh\braille_rl-master\braille_rl-master\data\supervised_data\arrows\train\videos"
df = pd.read_csv(r"D:\Josh\braille_rl-master\braille_rl-master\data\supervised_data\arrows\train\targets_video.csv")

def find_letter(video_name):
    row = df.loc[df['sensor_video'] == video_name]
    letter = row['obj_lbl'].to_numpy()[0]
    return letter

for i in range(400):
    i += 1

    video_name = r'\video_' + str(i)
    vid_name = 'video_' + str(i)
    full_name = file_loc + video_name + '.mp4'
    vidcap = cv2.VideoCapture(full_name)
    success,image = vidcap.read()
    count = 0

    letter = find_letter(vid_name+'.mp4')
    while success:
        file_name = r"D:\Josh\github\individual_project\simulation\simulation_data\alex_key_images\{}{}.png".format(letter, video_name)
        cv2.imwrite(file_name, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
