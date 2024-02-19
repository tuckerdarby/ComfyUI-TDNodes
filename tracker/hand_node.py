import os
import folder_paths
import numpy as np


folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)


class HandTrackerNode:
    def __init__(self) -> None:
        ...
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), 
                "pose_keypoints": ("POSE_KEYPOINT",), 
            },
        }

    RETURN_TYPES = ("TRACKING", "IMAGE")
    FUNCTION = "track"
    CATEGORY = "tracking"

    # class_id.tracker_id.frame = [x0, y0, x1, y1, w,]

    def track(self, images, pose_keypoints):
        B, H, W, C = images.shape
        tracked = {
            'hand': {
                0: []
            }
        }

        for keypoint in pose_keypoints:
            person = keypoint['people'][0]
            right_hand = person['hand_right_keypoints_2d']
            x0, y0, x1, y1 = W, H, 0, 0
            i = 0
            for point in right_hand:
                if i == 3:
                    i = 0
                if i == 0:
                    x0 = min(point*W, x0)
                    x1 = max(point*W, x1)
                elif i == 1:
                    y0 = min(point*H, y0)
                    y1 = max(point*H, y1)
                i += 1
            
            tracked['hand'][0].append([x0, y0, x1, y1, W, H])
                    
        return (tracked, images)
