import cv2
import numpy as np
from fastsam import FastSAM, FastSAMPrompt
import torch

class FastSAMutils:
    def __init__(self, model_path='./weights/FastSAM-x.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize FastSAM model
        :param model_path: Path to FastSAM model weights
        :param device: Device to run model on (cuda or cpu)
        """
        self.model = FastSAM(model_path)
        self.device = device
        
    def segment_with_point_prompt(self, zed_image, point_2d, point_label=1):
        """
        Segment an image using a point prompt with ZED SDK images
        :param zed_image: ZED SDK image object (from retrieve_image)
        :param point_cloud: ZED SDK point cloud object (from retrieve_measure)
        :param point_2d: (x, y) coordinates of the point prompt in image coordinates
        :param point_label: 1 (foreground) or 0 (background)
        :return: Tuple of (binary mask, segmented_point_cloud)
        """
        # Convert ZED image to numpy array
        image_np = cv2.cvtColor(zed_image, cv2.COLOR_BGRA2RGB)
        
        # Run inference
        everything_results = self.model(image_np, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # Prepare prompt
        prompt_process = FastSAMPrompt(image_np, everything_results, device=self.device)
        
        # Point prompt format: [[x, y, label]]
        point_prompt = [[point_2d[0], point_2d[1], point_label]]
        
        # Get mask from point prompt
        ann = prompt_process.point_prompt(points=point_prompt, pointlabel=[point_label])
        mask = ann[0].astype(np.uint8)  # Get the first mask
        
        return mask

    def check_contour_valid_point_cloud(self, mask, point_cloud):
        # Extract contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out points with NaN or Inf in the point cloud
        valid_contours = []
        for contour in contours:
            valid_contour = []
            for point in contour:
                x, y = int(point[0,0]), int(point[0,1])
                point3D = point_cloud.get_value(x,y)
                if np.all(np.isfinite(point3D[1])):
                    valid_contour.append((x, y))
            if valid_contour:
                valid_contours.append(valid_contour)
        
        # Update contours with valid points
        contours = [np.array(valid_contour).reshape(-1, 1, 2) for valid_contour in valid_contours]
        return contours


    def left_right_point_extractor(self, mask, point_cloud, contours):
        """
        """
        contours = self.check_contour_valid_point_cloud(mask, point_cloud)    

        # Y Distance to the leftmost and rightmost points
        leftmost_point = None
        rightmost_point = None
        for contour in contours:
            for point in contour:
                x, y = int(point[0,0]), int(point[0,1])
                point3D = point_cloud.get_value(x,y)
                if leftmost_point is None or point3D[1][1] > leftmost_point: # Point [1][1] refers to Y axis, horizontal distance.
                    leftmost_point = point3D[1][1]
                if rightmost_point is None or point3D[1][1] < rightmost_point:
                    rightmost_point = point3D[1][1]
  
        return leftmost_point, rightmost_point