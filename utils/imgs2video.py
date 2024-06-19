import os
import cv2

def imgs2video(dataset):
	file_dir = f'./sample/{dataset}/'
	list = []
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			list.append(file)

	video = cv2.VideoWriter(f'./sample/{dataset}_train.mp4',
	                        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),  8,  (2066, 776))

	for i in range(1, len(list)):
		img = cv2.imread(f'./sample/{dataset}/{list[i - 1]}')
		# print(img.shape)
		video.write(img)
	video.release()

