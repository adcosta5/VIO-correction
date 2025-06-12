import os
import sys
import cv2
import json
import enum
import argparse
import numpy as np
import pyzed.sl as sl
import pathlib as Path
import matplotlib.pyplot as plt

from collections import deque
from shapely.geometry import Polygon

from plot_utils import RealTimePlotter
from fastSAM_utils import FastSAMutils
from correction_utils import boundary_correction, point_correction, multipoint_correction, merge_geoseries_obstacles
from utils import GT_reader, street_segmentation, quaternion_to_rotation_matrix, create_transformation_matrix, rotation_matrix_z
import sys
from orienternet_utils import ori_pos_orienternet

def main(seq):
    # Initialize FastSAM
    fastsam = FastSAMutils()

    # Get input parameters
    svo_input_path = opt.input_svo_file
    output_dir = opt.output_path_dir
    ground_truth = opt.ground_truth
    plot = opt.plot
    show_FastSAM = opt.show_FastSAM
    correction_type = opt.correction_type

    if not os.path.isdir(output_dir):
        sys.stdout.write("Output directory doesn't exist. Check permission or create it.\n",
                         output_dir, "\n")
        exit()

    # Create zed object
    zed = sl.Camera()

    # Specify SVO path parameter
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path) 
    init.svo_real_time_mode = False         # Don't convert in realtime
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD # Right-handed, z-up, x-forward
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Better quality
    init.enable_right_side_measure = False

    # Open the SVO file 
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # Prepare single image containers
    left_image = sl.Mat()
    point_cloud = sl.Mat()

    zed_pose = sl.Pose() # Visual-Inertial SLAM pose
    py_transform = sl.Transform()  # Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if (err != sl.ERROR_CODE.SUCCESS):
            exit(-1)
    
    rt_param = sl.RuntimeParameters()

    if ground_truth:
        max_lat, min_lat, max_lon, min_lon, zone_number, initial_point, initial_angle, initial_point_latlon = GT_reader(seq)
        
    else:
        # Hardcode for 00
        max_lat, min_lat, max_lon, min_lon = 426199.3624994039,425978.7850167572,4581793.943468097,4581560.487520885
        zone_number = 31
        initial_point = (426191.96, 4581761.48)
        initial_angle = 180
        initial_point_latlon = (41.3839954, 2.1172553)

    zone = "+proj=utm +zone=" + str(zone_number) + " +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    edges, road_area, walkable_area_gdf,building_area, crossings_area, railway_area, green_area = street_segmentation(initial_point_latlon,zone,area=750)

    merged_obstacles = merge_geoseries_obstacles(road_area, building_area, railway_area,green_area)

    # Define fov_box parameters for the point and multipoint correction
    view_dist = 20 # 20 meters viewing distance

    # Define the box corners relative to the point
    box_corners = np.array([[-0.5, -10.0], [-0.5, 10.0], [11.5, 10], [11.5, -10.0]]) # Box 20 by 20m 

    if plot:
        # Initialize real-time plotter
        plotter = RealTimePlotter(
            edges, road_area, building_area, crossings_area, 
            railway_area, green_area, min_lat, min_lon, max_lat, max_lon
        )

        # Add straight line at angle = 0 (horizontal line going east)
        # Create a straight line from the initial point extending eastward
        line_length = 200  # Length of the line in meters
        start_point = initial_point
        end_point = (initial_point[0] + line_length, initial_point[1])  # Angle = 0 means going east (positive X direction)
        
        # Add the straight lines to the plotter
        plotter.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                       color='red', linewidth=2, linestyle='--', alpha=0.7, label='Reference Line - East (0°)')
        initial_angle = 90
        angle_rad = np.deg2rad(initial_angle)
        end_point_angle = (
            initial_point[0] + line_length * np.cos(angle_rad),
            initial_point[1] + line_length * np.sin(angle_rad)
        )
        plotter.ax.plot([initial_point[0], end_point_angle[0]],
                        [initial_point[1], end_point_angle[1]],
                        color='blue', linewidth=2, linestyle='--', alpha=0.7, label=f'Reference Line - North (90°)')
        
        plotter.ax.legend()

        plt.pause(0.01)
        plotter.start_animation()

    sys.stdout.write("Reading SVO... Use Ctrl-C to interrupt.\n")
    nb_frames = zed.get_svo_number_of_frames()
    is_first_frame = True

    # Initialize translation buffer
    point_buffer = deque(maxlen=10)
    smoothed_translation = None

    try: 
        while True:
            err = zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()
                
                # Retrieve left image and point locud
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                image_np = left_image.get_data()
                point_prompt = (600,700)

                if correction_type != "boundary":
                    if svo_position % 5 == 0: # Perform image segmentation every 5 frames
                        mask = fastsam.segment_with_point_prompt(image_np, point_prompt) 
                        contours = fastsam.check_contour_valid_point_cloud(mask, point_cloud)
                        if correction_type == "point":
                            left_distance, right_distance = fastsam.left_right_point_extractor(mask, point_cloud, contours)

                if show_FastSAM:
                    # To visualize the mask:
                    cv2.imshow("Segmentation Mask", mask * 255)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # Retrieve translation and orientation 
                zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)

                translation = zed_pose.get_translation()
                orientation = zed_pose.get_orientation()

                if is_first_frame:

                    first_point = (translation.get()[0] + initial_point[0],
                                   translation.get()[1] + initial_point[1],
                                   translation.get()[2])

                    # Obtain  the initial angle with OrienterNet
                    # Convert image_np to orienternet format (RGB)
                    image_ori = image_np[:,:,:3]
                    image_ori = image_ori[:, :, [2, 1, 0]]
                    uncorrected_angle = ori_pos_orienternet(image_ori, np.array(initial_point_latlon))

                    initial_angle = 90 - uncorrected_angle
                    print(f"The initial angle is: {initial_angle}")
                    angle_rad = np.deg2rad(initial_angle)

                     # Add arrow for the initial orientation from OrienterNet
                    if plot:
                        arrow_length = 30  # Length of the arrow in meters
                        arrow_end = (
                            initial_point[0] + arrow_length * np.cos(angle_rad),
                            initial_point[1] + arrow_length * np.sin(angle_rad)
                        )
                        
                        plotter.ax.arrow(initial_point[0], initial_point[1], 
                                        arrow_end[0] - initial_point[0], arrow_end[1] - initial_point[1],
                                        head_width=5, head_length=7, fc='green', ec='green', 
                                        width=1, label=f'OrienterNet Initial Angle ({initial_angle:.1f}°)')
                        
                        # Redraw legend to include the new arrow
                        plotter.ax.legend()
                        plt.pause(0.01)  # Update the plot

                    R_matrix = rotation_matrix_z(angle_rad)
                    # R_matrix = quaternion_to_rotation_matrix((orientation.get()[0],orientation.get()[1],orientation.get()[2],orientation.get()[3]))
                    transf_matrix = create_transformation_matrix(first_point,R_matrix)
                    corrected_transf_matrix = transf_matrix

                    is_first_frame = False  # Ensure this runs only once
                
                else:
                    
                    R_matrix = quaternion_to_rotation_matrix((orientation.get()[0],orientation.get()[1],orientation.get()[2],orientation.get()[3]))
                    T_matrix = create_transformation_matrix((translation.get()[0],translation.get()[1],translation.get()[2]),R_matrix)
                    transf_matrix = prev_transf @ T_matrix
                    corrected_transf_matrix = prev_corrected_transf @ T_matrix

                # print(f"Estimated trajectory: {translation.get()[0]}, {translation.get()[1]}") # Debug
                # print(f"Transformation Matrix: \n {transf_matrix}") # Debug
                # print(f"Corrected Transformation Matrix: \n {corrected_transf_matrix}")
                
                estimated_odom = (transf_matrix[0,3], transf_matrix[1,3])
                corrected_odom = (corrected_transf_matrix[0,3], corrected_transf_matrix[1,3])
                # corrected_odom = (transf_matrix[0,3], transf_matrix[1,3])
                # print(f"Estimated point: {estimated_odom}\n") # Debug
                point_buffer.append(estimated_odom)

                if len(point_buffer) >= 2:
                    if len(point_buffer) == 10:
                        past_x, past_y = point_buffer[0]  # 10-frame window
                    else:
                        past_x, past_y = point_buffer[0]  # First available point
                    
                    dx = estimated_odom[0] - past_x
                    dy = estimated_odom[1] - past_y
                    line_angle = np.arctan2(dy, dx)
                else:
                    line_angle = 0

                # Rotate the box to adjust the orientation of the trajectory
                rotation_matrix_line = np.array([[np.cos(line_angle), -np.sin(line_angle)],
                                [np.sin(line_angle), np.cos(line_angle)]])
                rotated_corners = np.dot(box_corners, rotation_matrix_line.T)

                rotated_box_coords = rotated_corners + np.array([corrected_odom[0], corrected_odom[1]])
                fov_box = Polygon(rotated_box_coords)

                if correction_type == "boundary":
                    adjusted_point = boundary_correction(corrected_odom, merged_obstacles, crossings_area)
                    # print(f"Adjusted point: {adjusted_point}\n") #Debug

                    corrected_transf_matrix[0,3] = adjusted_point[0]
                    corrected_transf_matrix[1,3] = adjusted_point[1]
                    
                    if plot:
                        point_added = plotter.add_est_point(transf_matrix[0,3], transf_matrix[1,3])
                        corrected_point_added = plotter.add_corr_point(adjusted_point[0], adjusted_point[1])

                        if svo_position % 5 == 0: # Update every 5 frames
                            plt.pause(0.001) # Short pause to allow GUI updates


                elif correction_type == "point":
                    if svo_position > 50 and (svo_position % 5 == 0 ):
                        intersecting_objects, corrected_point = point_correction(corrected_odom, view_dist, merged_obstacles, building_area, crossings_area, fov_box, line_angle, left_distance, right_distance)

                        corrected_transf_matrix[0,3] = corrected_point[0]
                        corrected_transf_matrix[1,3] = corrected_point[1]

                        if plot:
                            point_added = plotter.add_est_point(transf_matrix[0,3], transf_matrix[1,3])
                            corrected_point_added = plotter.add_corr_point(corrected_transf_matrix[0,3], corrected_transf_matrix[1,3])

                            if hasattr(fov_box, 'exterior'):  # Ensure it's a valid Polygon
                                plotter.add_temporary_elements(fov_box, intersecting_objects, 0)

                            if svo_position % 5 == 0: # Update every 5 frames
                                plt.pause(0.001) # Short pause to allow GUI updates

                elif correction_type == "multipoint":
                    print(f"SVO_POSITION: {svo_position}")
                    if svo_position > 50 and (svo_position % 5 == 0):
                    # if svo_position == 1270: 
                        intersecting_objects, rotated_point_cloud, corrected_point = multipoint_correction(corrected_odom, view_dist, merged_obstacles, building_area, crossings_area, fov_box, line_angle, mask, point_cloud, contours)
                        print(f"Estimated Point: {estimated_odom}")
                        print(f"Corrected Point (multipoint): {corrected_point}")

                        corrected_transf_matrix[0,3] = corrected_point[0]
                        corrected_transf_matrix[1,3] = corrected_point[1]

                        if plot:
                            point_added = plotter.add_est_point(transf_matrix[0,3], transf_matrix[1,3])
                            corrected_point_added = plotter.add_corr_point(corrected_transf_matrix[0,3], corrected_transf_matrix[1,3])

                            # if hasattr(fov_box, 'exterior'):  # Ensure it's a valid Polygon
                            #         plotter.add_temporary_elements(fov_box, intersecting_objects, 0)

                            if svo_position % 5 == 0:
                                plt.pause(0.001)

                else: 

                    if is_first_frame:
                        print(f"No correction is being executed")

                    if plot:
                        point_added = plotter.add_est_point(transf_matrix[0,3], transf_matrix[1,3])

                        if point_added and len(plotter.est_x) % 5 == 0:
                            plt.pause(0.001)  # Short pause to allow GUI updates
                        

                prev_transf = transf_matrix
                prev_corrected_transf = corrected_transf_matrix
        
            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    except KeyboardInterrupt:
        sys.stdout.write("\nProcessing interrupted by user.\n")
    finally:
        zed.close()
        if plot:
            plotter.close()
            plt.show()  # Keep plot open after processing

    return 0

if __name__ == "__main__":
    
    seq = "20"
    input_svo_file = "../../datasets/BIEL/svofiles/IRI_" + seq + ".svo2"
    output_dir = "."
    ground_truth = True
    plot = True 
    show_FastSAM = False
    correction_type = "point"

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_svo_file', type=str, default=input_svo_file, help='Path to the .svo file')
    parser.add_argument('--output_path_dir', type = str, default = output_dir, help = 'Path to a directory, where .png will be written, if mode includes image sequence export')
    parser.add_argument('--plot', type = bool, default = plot, help= "True for a plot with the trajectory")
    parser.add_argument('--ground_truth', type = bool, default = ground_truth, help= "True if there exists a ground truth for the sequence")
    parser.add_argument('--show_FastSAM', type = bool, default = show_FastSAM, help= "True to show FastSAM results")
    parser.add_argument('--correction_type', type = str, default = correction_type, help = "Select the type of correction: boundary, point, multipoint")
    opt = parser.parse_args()

    if not opt.input_svo_file.endswith(".svo") and not opt.input_svo_file.endswith(".svo2"): 
        print("--input_svo_file parameter should be a .svo file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if len(opt.output_path_dir)==0 :
        print("In mode ",opt.mode,", output_path_dir parameter needs to be specified.")
        exit()
    if not os.path.isdir(opt.output_path_dir):
        print("--output_path_dir parameter should be an existing folder but is not : ",opt.output_path_dir,"Exit program.")
        exit()
    main(seq)
