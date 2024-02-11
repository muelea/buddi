# From Python
# It requires OpenCV installed for Python
import sys
sys.path.append('../../python')
import pyopenpose as op
import cv2
import os
from sys import platform
import argparse
import time
import glob
import os.path as osp
import subprocess
from tqdm import tqdm

def move_rename_kp(args, IMGoutdir, json_fn, IMGname):
	os.makedirs(IMGoutdir, exist_ok=True)
	subprocess.call(["mv", json_fn, osp.join(IMGoutdir, IMGname + '.json')])

def main(args):
		if args.img_dir[-1] != '/' or args.out_dir[-1] != '/':
			print('Try again. Path shoudl have / at the end')
			sys.exit(1)

		# Custom Params (refer to include/openpose/flags.hpp for more parameters)
		params = dict()
		params["model_folder"] = "../../models/"
		params["face"] = True
		params["hand"] = True
		params["write_json"] = osp.join(args.out_dir, 'temp')
		os.makedirs(osp.join(args.out_dir, 'temp'), exist_ok=True)
		assert len(os.listdir(osp.join(args.out_dir, 'temp'))) == 0, 'temp file not empty'
		# Starting OpenPose
		opWrapper = op.WrapperPython()
		opWrapper.configure(params)
		opWrapper.start()

		# read images in dir
		s = time.time()
		print('start reading images ... ')
		IMGS = glob.glob(osp.join(args.img_dir, '**'), recursive=True)
		IMGS = sorted([x for x in IMGS if x.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'bmp', 'JPG']])
		# IMGS = [x for x in IMGS if 'SelfContact_190723_00170_TA' in x]
		# comment next line in if images are .bmp from scanner
		#IMGS = sorted([x for x in IMGS if x.split('.')[-2] == '07_C' and 'simple' not in x])
		print('done reading {} images after {} seconds'.format(len(IMGS), time.time()-s))

		# run openpose for each iamge
		for i, IMG in tqdm(enumerate(IMGS), total=len(IMGS)):
			try:
				#start = time.time()

				IMGpath = osp.dirname(IMG.replace(args.img_dir, '')).strip('/') #make sure slashes are removed
				IMGname = '.'.join(osp.basename(IMG).split('.')[:-1])
				KPoutdir = osp.join(args.out_dir,'keypoints', IMGpath)
				IMGoutdir = osp.join(args.out_dir, 'images', IMGpath)
				os.makedirs(IMGoutdir, exist_ok=True)
				os.makedirs(KPoutdir, exist_ok=True)

				# Process images
				datum = op.Datum()
				imageToProcess = cv2.imread(IMG)
				datum.cvInputData = imageToProcess
				opWrapper.emplaceAndPop([datum])

				# display if flag is True
				if args.display:
					cv2.imshow("Image", datum.cvOutputData)
					key = cv2.waitKey(15)
					if key == 27: break
				if args.imgsave:
					print(osp.join(IMGoutdir, IMGname + '.png'))
					cv2.imwrite(osp.join(IMGoutdir, IMGname + '.png'), datum.cvOutputData)

				#end = time.time()
				#print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

				# move json file to right directory
				# print("Body keypoints: \n" + str(datum.poseKeypoints))
				op_fn = glob.glob(osp.join(args.out_dir, 'temp', '*.json'))
				assert len(op_fn) == 1, 'Too many json files in folder.'
				#read_i = int(readIMGname[0].split('/')[-1].split('_')[0])

				move_rename_kp(args, KPoutdir, op_fn[0], IMGname)

			except Exception as e:
				print(e, i, IMG)
				pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_dir", default="../../examples/media/", help="dir with images, also in subfolders")
	parser.add_argument("--out_dir", default="../../examples/media/", help="dir where to write openpose files")
	parser.add_argument("--display", type=lambda x: x.lower() in ['True', 'true', '1'], help="Enable to disable the visual display.")
	parser.add_argument("--imgsave", type=lambda x: x.lower() in ['True', 'true', '1'], help="Save openpose iamge.")
	args = parser.parse_args()
	main(args)