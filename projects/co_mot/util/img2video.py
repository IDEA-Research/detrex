import os
import cv2

video_root = '/home/hadoop-vacv/yanfeng/data/dancetrack/val/dancetrack0004/img1'
imgs_path = [os.path.join(video_root, tmp) for tmp in os.listdir(video_root)]

imgs_path = sorted(imgs_path)

for ith, img_p in enumerate(imgs_path):
    print(ith)
    img = cv2.imread(img_p)
    img_h, img_w, _ = img.shape
    if ith==0:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('demo_video.avi', fourcc, 25.0, (img_w, img_h))

    out.write(img)
out.release()