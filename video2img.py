
from video2images import Video2Images


#video_filepath: 	source path of the video
#out_dir: Output directory for saving images. If not specified
#a folder will be made in current directory and saved

#capture_rate: No. of frames you want to capture per second.
#For e.g if my capture_rate= 20 then only first 20
#frames will captured out of input Frames per second


#Video2Images(video_filepath="checkerboard_iphone.avi", save_format=".png", capture_rate=1,
#            out_dir="stream_checker_board")
import os
import glob
import cv2


#video_filename = '0202weißlicht_discov2.mp4'
video_filename = 'D:\Documents\Studium\Studienarbeit\GIT_ORB_SLAM_MATLAB\original_videos\A.mpg'
#img_folder = 'img_weisslicht_discover2'
img_folder = 'D:\Documents\Studium\Studienarbeit\GIT_ORB_SLAM_MATLAB\original_videos\img_A'

vidcap = cv2.VideoCapture(video_filename)
success, image = vidcap.read()
#files = glob.glob('img_weisslicht_discover2/*')
files = glob.glob('D:\Documents\Studium\Studienarbeit\GIT_ORB_SLAM_MATLAB\original_videos\img_A/*')
for f in files:
  os.remove(f)


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    # if hasFrames:
    #  cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    return hasFrames, image

count = 1
sec = 0
frameRate =0.04  # Change this number to 1 for each 1 second
success = 1
while success:
    count = count + 1
    sec = sec + frameRate

    sec = round(sec, 2)
    success, img = getFrame(sec)
    if success:
        cv2.imwrite(os.path.join(img_folder, "frame{:d}.png".format(count)), img)  #save frame as JPEG file
    if sec > 250:
        success = False

print("{} images are extacted in {}.".format(count, img_folder))