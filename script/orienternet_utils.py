import torch
import numpy as np
import sys
sys.path.append("../../modules/OrienterNet/")

from maploc.demo import Demo
from maploc.osm.tiling import TileManager

"""
Script to obtain the initial angle using OrienteNet. The choosen orientation will
be the one of the closest predicted point to the prior_address (aka inital_point)
"""

def dense_rotations(prob, thresh=0.01, skip=10, s=1 / 15, k=3, c="k", w=None, **kwargs
):
    t = torch.argmax(prob, -1)
    yaws = t.numpy() / prob.shape[-1] * 360
    prob = prob.max(-1).values / prob.max()
    mask = prob > thresh
    masked = prob.masked_fill(~mask, 0)
    max_ = torch.nn.functional.max_pool2d(
        masked.float()[None, None], k, stride=1, padding=k // 2
    )
    mask = (max_[0, 0] == masked.float()) & mask
    indices = np.where(mask.numpy() > 0)

    # Return the orientations at the valid indices
    orientations = yaws[indices]
    positions = indices[::-1]  # (x, y) coordinates
    
    return orientations, positions

def ori_pos_orienternet(image, prior_latlon):
    
    
    # Increasing the number of rotations increases the accuracy but requires more GPU memory.
    # The highest accuracy is achieved with num_rotations=360
    # but num_rotations=64~128 is often sufficient.
    # To reduce the memory usage, we can reduce the tile size in the next cell.
    demo = Demo(num_rotations=256, device="cuda")

    image, camera, gravity, proj, bbox = demo.read_input_numpy_image( # Auto extracts camera calibration parameters
    image,
    prior_latlon=prior_latlon,
    focal_length=706.391, # Hardcoded
    tile_size_meters=64)  # The smaller the better (if the prior address is good)
    
    # Query OpenStreetMap for this area
    tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter)
    canvas = tiler.query(bbox)

    # Run the inference
    uv, yaw, prob, neural_map, image_rectified = demo.localize(
    image, camera, canvas, gravity=gravity
)
    # Get the orientations and positions
    orientations, positions = dense_rotations(prob, w=0.005, s=1 / 25)

    # Convert prior_latlon to canvas coordinate system
    xy_prior = np.array(canvas.to_uv(proj.project(proj.latlonalt[:2]))) 
    print("Initial", xy_prior)
 
    # positions is a tuple of (y_array, x_array)
    positions = np.stack(positions, axis=1)  # shape (N, 2)
    print("Positions:", positions)

    # Find closest position to origin (0, 0)
    distances = np.linalg.norm(positions - xy_prior, axis=1)
    closest_idx = np.argmin(distances)
    print("Distances:", distances)
    print("Closest index:", closest_idx)

    # Get the best orientation
    angle = orientations[closest_idx]
    print("Selected angle:", angle)
    print("All orientations:", orientations)

    return angle



