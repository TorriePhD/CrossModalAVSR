#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
from tqdm import tqdm
warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name="resnet50"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )
        self.landmark_detector = FANPredictor(device=device, model=None)

    def __call__(self, video_frames,selectedFace=None):
        #selected face is the x,y coordinates of the face to be selected normalized from 0 to 1
        landmarks = []
        for frame in video_frames:
            detected_faces = self.face_detector(frame, rgb=False)
            face_points, _ = self.landmark_detector(frame, detected_faces, rgb=True)
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                chosen_id, max_size = 0, 0
                if selectedFace is not None:
                    closestDist = 1000
                for idx, bbox in enumerate(detected_faces):
                    bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    #normalize center from 0 to 1
                    center = center[0] / frame.shape[1], center[1] / frame.shape[0]
                    if selectedFace is not None:
                        dist = (center[0]-selectedFace[0])**2 + (center[1]-selectedFace[1])**2
                        if dist < closestDist:
                            closestDist = dist
                            chosen_id = idx
                    elif bbox_size > max_size:
                        chosen_id, max_size = idx, bbox_size
                landmarks.append(face_points[chosen_id])
        return landmarks
