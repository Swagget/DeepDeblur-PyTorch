"""
Optical flow based selective sharpening.
"""

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class selective_sharpening:
    def __init__(self):
        self.kernel_10 = np.ones((10, 10), np.uint8)
        self.kernel_3 = np.ones((3, 3), np.uint8)
        self.optical_flow_parameters = {
            "pyr_scale" : 0.8,
            "levels" : 15,
            "winsize" : 5,
            "iterations" : 10,
            "poly_n" : 5,
            "poly_sigma" : 0,
            "flags" : 10
        }

    def greyfy(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def convert_optical_flow_to_flow_magnitudes(self, optical_flow):
        optical_flow_magnitude = np.zeros(optical_flow.shape[:2])
        for y in tqdm(range(optical_flow.shape[0])):
            for x in range(optical_flow.shape[1]):
                optical_flow_magnitude[y, x] = np.sqrt(optical_flow[y, x, 0] ** 2 + optical_flow[y, x, 1] ** 2)
        return optical_flow_magnitude

    def clean_frame(self, previous_frame, current_frame, sharpened_current_frame):
        flow_output = cv.calcOpticalFlowFarneback(self.greyfy(previous_frame), self.greyfy(current_frame), None,
                                                  pyr_scale=self.optical_flow_parameters["pyr_scale"],
                                                  levels=self.optical_flow_parameters["levels"],
                                                  winsize=self.optical_flow_parameters["winsize"],
                                                  iterations=self.optical_flow_parameters["iterations"],
                                                  poly_n=self.optical_flow_parameters["poly_n"],
                                                  poly_sigma=self.optical_flow_parameters["poly_sigma"],
                                                  flags=self.optical_flow_parameters["flags"])

        optical_flow_magnitude = self.convert_optical_flow_to_flow_magnitudes(flow_output)

        t, optical_flow_mask = cv.threshold(optical_flow_magnitude, 3, maxval=255, type=cv.THRESH_BINARY)

        merged_image = self.merge_images_with_mask(blurry_image=current_frame, sharp_image=sharpened_current_frame,
                                                   mask=optical_flow_mask)
        return merged_image

    def merge_images_with_mask(self, blurry_image, sharp_image, mask):
        merged_image = cv.bitwise_and(blurry_image,blurry_image,mask = mask-1)
        cropped_sharp = cv.bitwise_and(sharp_image,sharp_image,mask = mask)
        merged_image = cv.bitwise_or(merged_image,cropped_sharp)
        return merged_image

    def create_optical_flow_blurriness_mask(self, optical_flow_magnitude, optical_flow_threshold):
        optical_flow_mask = np.zeros(optical_flow_magnitude.shape, dtype=np.uint8)
        for y in range(optical_flow_magnitude.shape[0]):
            for x in range(optical_flow_magnitude.shape[1]):
                if optical_flow_magnitude[y, x] > optical_flow_threshold:
                    optical_flow_mask[y, x] = True
        return optical_flow_mask

    def erode_and_dialate_mask(self, mask):
        mask = cv.erode(mask, self.kernel_10, iterations=1)
        mask = cv.erode(mask, self.kernel_3, iterations=1)
        mask = cv.dilate(mask, self.kernel_10, iterations=10)
        return mask

    def video_to_frames(self, video_path, output_path = "./temp"):
        vidcap = cv.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        all_frames = []
        while success:
            cv.imwrite(output_path + "/frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            all_frames.append(image)
            print('Read a new frame: ', success)
            count += 1
        return all_frames

    def convert_frames_to_video(self, all_frames, pathOut, fps):
        frame_array = []
        # files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
        #for sorting the file names properly
        # files.sort(key = lambda x: int(x[5:-4]))
        for img in all_frames:
            # filename=pathIn + files[i]
            #reading each files
            # img = cv.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            # print(filename)
            #inserting the frames into an image array
            frame_array.append(img)
        out = cv.VideoWriter(pathOut,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
        for output_frame in frame_array:
            # writing to a image array
            out.write(output_frame)
        out.release()

    def end_to_end_sharpening(self, video_path):
        os.mkdir("./temp")
        os.mkdir("./temp/blurry_frames")
        os.mkdir("../results")
        all_blurry_frames = self.video_to_frames(video_path = video_path, output_path = "./temp/blurry_frames")
        # Surya's code to run the network here.
        os.system("python main.py - -save_dir GOPRO_L1 - -dataset GOPRO_Large") # Not sure if it'll work.
        array_of_outputs_from_generator_in_order = []
        self.output_frames = []
        for frame_number in range(1, len(all_blurry_frames)):
            self.output_frames.append(self.clean_frame(previous_frame=all_blurry_frames[frame_number-1],
                                                       current_frame=all_blurry_frames[frame_number],
                                                       sharpened_current_frame=array_of_outputs_from_generator_in_order[frame_number]))
        self.convert_frames_to_video(all_frames=self.output_frames, pathOut = "../results/clean_video.mp4" , fps = 30) # I think 30?