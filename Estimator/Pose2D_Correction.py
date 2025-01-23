import re
import math
import numpy as np
import logging
from Base_Pose_Estimator import BasePoseEstimator,LimbAssignment

class PoseCorrection(BasePoseEstimator):
    def __init__(self, body_parts_list,connections,left_limbs=None, right_limbs=None):
        super().__init__(body_parts_list, connections)
        self.limb_assignment = LimbAssignment(self.labels, self.connections)
        self.left_limbs = self.limb_assignment.left_limbs
        self.right_limbs = self.limb_assignment.right_limbs
        self.switch_map = self._create_switch_map()
    
    
    def _create_switch_map(self):
        # Initialize an empty dictionary for the switch map
        switch_map = {}

        # Iterate through the body parts list to create the switch map
        for part in self.labels:
            if "left" in part:
                # Create the switched counterpart by replacing left with right
                right_part = part.replace("left", "right")
            elif part.startswith('l'):
                right_part = part.replace("l", "r", 1)
            else:
                continue

            # Check if the right counterpart exists in the body parts list
            if right_part in self.labels:
                left_index = self.labels.index(part)
                right_index = self.labels.index(right_part)

                # Add the mapping only if it doesn't already exist
                if left_index not in switch_map and right_index not in switch_map:
                    switch_map[left_index] = right_index
                    switch_map[right_index] = left_index

        return switch_map
    
    
    def switch_sides_2d(self, body_coordinates):
        
  
        """
        Switch the left and right body parts for a single frame of coordinates.
        
        :param body_coordinates: List of body coordinates for a single frame.
        :return: List of body coordinates with left-right sides switched.
        """
        # Create a new list for switched coordinates using list comprehension
       
        switched_coordinates = [
            body_coordinates[self.switch_map[i]] if i in self.switch_map else body_coordinates[i]
            for i in range(len(body_coordinates))
        ]
    
        return switched_coordinates
   
    def switch_sides_2d_all_frames(self, frames):
        """
        Switch the left and right body parts for all frames.
        
        :param frames: List of frames, where each frame is a list of body coordinates.
        :return: List of frames with left-right sides switched.
        """
        return [self.switch_sides_2d(frame) for frame in frames]
    
    def switch_sides_mirror(self, points1,points2,frame=1):
        height1 = self.get_body_height(points1[1])
        height2 = self.get_body_height(points2[1])
        if height1 > height2:
            points2=self.switch_sides_2d_all_frames(points2)
        else:
            points1=self.switch_sides_2d_all_frames(points1)
        return points1,points2
    
    def get_body_height(self, points: list) -> float:
        points = np.array(points)
        head_point = self._get_head_point(points)
        hip_point = self._get_hip_point(points)
        distance = math.sqrt((head_point[0] - hip_point[0])**2 + (head_point[1] - hip_point[1])**2)
        return distance

    def _get_head_point(self, points: np.ndarray) -> np.ndarray:
        if "head" in self.labels:
            return points[self.labels.index("head")]
        elif "neck" in self.labels:
            return points[self.labels.index("neck")]
        elif "headtop" in self.labels:
            return points[self.labels.index("headtop")]
        elif "leftshoulder" in self.labels and "rightshoulder" in self.labels:
            lshoulder_idx = self.labels.index("leftshoulder")
            rshoulder_idx = self.labels.index("rightshoulder")
            return [(points[lshoulder_idx][0] + points[rshoulder_idx][0]) / 2, (points[lshoulder_idx][1] + points[rshoulder_idx][1]) / 2]
        else:
            raise ValueError("No valid head or neck keypoint found in the list of body parts")

    def _get_hip_point(self, points: np.ndarray) -> np.ndarray:
        if "hip" in self.labels:
            return points[self.labels.index("hip")]
        elif "lefthip" in self.labels and "righthip" in self.labels:
            lefthip_idx = self.labels.index("lefthip")
            righthip_idx = self.labels.index("righthip")
            return [(points[lefthip_idx][0] + points[righthip_idx][0]) / 2, (points[lefthip_idx][1] + points[righthip_idx][1]) / 2]

        elif "leftankle" in self.labels and "rightankle" in self.labels:
            lankle_idx = self.labels.index("leftankle")
            rankle_idx = self.labels.index("rightankle")
            return [(points[lankle_idx][0] + points[rankle_idx][0]) / 2, (points[lankle_idx][1] + points[rankle_idx][1]) / 2]
        else:
            raise ValueError("No valid hip keypoint found in the list of body parts")
        
    def get_avg_widths(self, frames, lshoulder_idx, rshoulder_idx, lhip_idx, rhip_idx):
        """
        Calculate the average shoulder and hip widths across all frames.
        
        :param frames: List of frames containing keypoints.
        :param lshoulder_idx: Index of the left shoulder in the keypoints.
        :param rshoulder_idx: Index of the right shoulder in the keypoints.
        :param lhip_idx: Index of the left hip in the keypoints.
        :param rhip_idx: Index of the right hip in the keypoints.
        :return: Tuple containing average shoulder width and average hip width.
        """
        total_shoulder_width = 0
        total_hip_width = 0
        count = len(frames)

        for keypoints in frames:
            shoulder_width = abs(keypoints[lshoulder_idx][0] - keypoints[rshoulder_idx][0])
            hip_width = abs(keypoints[lhip_idx][0] - keypoints[rhip_idx][0])

            total_shoulder_width += shoulder_width
            total_hip_width += hip_width

        avg_shoulder_width = total_shoulder_width / count
        avg_hip_width = total_hip_width / count

        return avg_shoulder_width, avg_hip_width

   
    def correct_left_right_switch(self, real_world, mirror,threshold=0):
        """
        Detect and correct left-right side switches for two bodies in both real-world and mirror frames.
        :param real_world: List of frames for the real-world keypoints.
        :param mirror: List of frames for the mirror keypoints.
        :return: Real-world and mirror frames with corrected keypoints, and dictionaries of switched frames.
        """
        upper_limbs, lower_limbs, feet = self.generate_limbs()
        
        
        
        # Precompute index pairs for all the limb groups
        pairs_indices_upper = self.get_pairs_indices(upper_limbs)
        print(pairs_indices_upper)
        pairs_indices_lower = self.get_pairs_indices(lower_limbs)
        pairs_indices_feet = self.get_pairs_indices(feet)

        # Dynamic index retrieval for shoulder and hip joints
        lshoulder_idx, rshoulder_idx = self.get_joint_indices(upper_limbs, "shoulder")
        print(lshoulder_idx, rshoulder_idx)
        lhip_idx, rhip_idx = self.get_joint_indices(lower_limbs, "hip")

        def calculate_average_widths(frames):
            return self.get_avg_widths(frames, lshoulder_idx, rshoulder_idx, lhip_idx, rhip_idx)
        
        def is_switched(keypoints, pairs_indices, avg_width, threshold=0.0):
            """
            Check if keypoints are switched based on their x-coordinates, with a threshold for body rotation.
            :param keypoints: List of keypoints in the frame.
            :param pairs_indices: List of (left_idx, right_idx) pairs for body parts.
            :param threshold: Minimum x-difference to account for natural body rotation.
            :return: True if all keypoints are switched, False otherwise.
            """
            for left_idx, right_idx in pairs_indices:
                
                left_x, right_x = keypoints[left_idx][0], keypoints[right_idx][0]
                if left_x > right_x + (threshold * avg_width):
                    return False  # Natural variation, not switched
                if left_x < right_x - (threshold * avg_width):
                    return True   # Clear case of switching
            return False  # No switch detected

        # Helper function to correct a full set of switched keypoints
        def correct_switch(keypoints, pairs_indices):
            for left_idx, right_idx in pairs_indices:
                keypoints[left_idx], keypoints[right_idx] = keypoints[right_idx], keypoints[left_idx]
                
        corrected_world, corrected_mirror, switched_world, switched_mirror = [], [], {}, {}
        
        avg_shoulder_width, avg_hip_width = calculate_average_widths(real_world)
        print(avg_shoulder_width)
        avg_shoulder_width_mirror, avg_hip_width_mirror = calculate_average_widths(mirror)

        for i, (world_keypoints, mirror_keypoints) in enumerate(zip(real_world, mirror)):
        
            shoulders_switched = is_switched(world_keypoints, [(lshoulder_idx, rshoulder_idx)], avg_shoulder_width, threshold)
            hips_switched = is_switched(world_keypoints, [(lhip_idx, rhip_idx)], avg_hip_width, threshold)

            if shoulders_switched:
                correct_switch(world_keypoints, pairs_indices_upper)
                switched_world[i] = switched_world.get(i, "") + "upper"
            if hips_switched:
                correct_switch(world_keypoints, pairs_indices_lower)
                correct_switch(world_keypoints, pairs_indices_feet)
                switched_world[i] = switched_world.get(i, "") + "lower" if not switched_world.get(i) else ", lower"
            
            # Mirror checks
            shoulders_switched_mirror = is_switched(mirror_keypoints, [(lshoulder_idx, rshoulder_idx)], avg_shoulder_width_mirror, threshold)
            hips_switched_mirror = is_switched(mirror_keypoints, [(lhip_idx, rhip_idx)], avg_hip_width_mirror, threshold)
            
            if shoulders_switched_mirror:
                correct_switch(mirror_keypoints, pairs_indices_upper)
                switched_mirror[i] = switched_mirror.get(i, "") + "upper"
            if hips_switched_mirror:
                correct_switch(mirror_keypoints, pairs_indices_lower)
                correct_switch(mirror_keypoints, pairs_indices_feet)
                switched_mirror[i] = switched_mirror.get(i, "") + "lower" if not switched_mirror.get(i) else ", lower"

            # Append corrected keypoints
            corrected_world.append(world_keypoints)
            corrected_mirror.append(mirror_keypoints)

        return corrected_world, corrected_mirror, switched_world, switched_mirror
    
    
    def get_pairs(self, limbs):
        """
        Generate pairs of left and right limbs.
        
        :param limbs: List of limbs.
        :return: List of pairs of left and right limbs.
        """
        pairs = []
        for limb in limbs:
            left_limb = [f"left{joint}" for joint in limb]
            right_limb = [f"right{joint}" for joint in limb]
            pairs.append((left_limb, right_limb))
        return pairs

    def generate_limbs(self):
        """
        Generate the upper limbs, lower limbs, and feet lists.
        
        :return: Tuple containing upper limbs, lower limbs, and feet lists.
        """
        upper_limbs = self.get_pairs([["shoulder", "elbow", "wrist"]])
        lower_limbs = self.get_pairs([["hip", "knee", "ankle"]])
        feet = self.get_pairs([["bigtoe", "smalltoe", "heel"]])
        return upper_limbs, lower_limbs, feet

    # Helper function for indexing
    def get_pairs_indices(self, limb_group):
        """
        Precompute index pairs for limbs.
        
        :param limb_group: List of limb pairs.
        :return: List of tuples containing the indices of the left and right limb pairs.
        """
        pairs_indices = []
        for limb in limb_group:
            left_indices = [self.labels.index(part) for part in limb[0]]
            right_indices = [self.labels.index(part) for part in limb[1]]
            pairs_indices.extend(zip(left_indices, right_indices))
        return pairs_indices
    def get_joint_indices(self, limb_group, joint_type):
        """Get indices for left and right joints of a specific type."""
        for left, right in limb_group:
            for joint_1,joint_2 in zip(left, right):
                if joint_type in joint_1 and joint_type in joint_2:
                    return self.labels.index(joint_1), self.labels.index(joint_2)
        return None, None
    
        
            
    def euclidean_distance(self,point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def calculate_centroid(self,points):     
        """Calculate the centroid of a list of points."""
        return np.mean(points, axis=0)  
    
    def correct_pose_assignment(self, points_realworld_all_frames, points_mirror_all_frames):
        """
        Correct keypoints based on centroid distances to determine if switching is needed.
        :param points_realworld_all_frames: List of frames with keypoints for the real-world.
        :param points_mirror_all_frames: List of frames with keypoints for the mirror.
        :return: Corrected keypoints for real-world and mirror frames.
        """
        corrected_realworld = [points_realworld_all_frames[0]]
        corrected_mirror = [points_mirror_all_frames[0]]
        
        centroid_world = self.calculate_centroid(points_realworld_all_frames[0])
        centroid_mirror = self.calculate_centroid(points_mirror_all_frames[0])
        swtiched_frames = []
        count=0
        
        print(len(points_realworld_all_frames))
        print(len(points_mirror_all_frames))
        for i in range(1, len(points_realworld_all_frames)):
            point_world = points_realworld_all_frames[i]
            point_mirror = points_mirror_all_frames[i]
            
            current_centroid_world = self.calculate_centroid(point_world)
            current_centroid_mirror = self.calculate_centroid(point_mirror)
            
            # Calculate distances between centroids
            world_distance_to_world = self.euclidean_distance(centroid_world, current_centroid_world)
            world_distance_to_mirror = self.euclidean_distance(centroid_mirror, current_centroid_world)
            mirror_distance_to_world = self.euclidean_distance(centroid_world, current_centroid_mirror)
            mirror_distance_to_mirror = self.euclidean_distance(centroid_mirror, current_centroid_mirror)
            
            # Correct based on distances
            if world_distance_to_world < world_distance_to_mirror:
                corrected_realworld.append(self.switch_sides_2d(point_world))
                corrected_mirror.append(self.switch_sides_2d(point_mirror))
            else:
                if mirror_distance_to_world < mirror_distance_to_mirror:
                    corrected_realworld.append(self.switch_sides_2d(point_mirror))
                    corrected_mirror.append(self.switch_sides_2d(point_world))
                    swtiched_frames.append(i)   
                    count=count+1
                elif mirror_distance_to_world > mirror_distance_to_mirror:
                    corrected_realworld.append(self.switch_sides_2d(point_world))
                    corrected_mirror.append(self.switch_sides_2d(point_mirror))
        
        return corrected_realworld, corrected_mirror, swtiched_frames
    
    def calculate_distances_from_centroid(self,keypoints, centroid):
        """Calculate Euclidean distances of keypoints from the centroid."""
        return np.linalg.norm(keypoints - centroid, axis=1)


    def calculate_all_centroids(self, keypoints_all_frames):
        return [self.calculate_centroid(np.array(keypoints)) for keypoints in keypoints_all_frames]
    
    def correct_deviation(self, keypoints_all_frames, deviations, max_frame_diff=5):
        corrected_frames = keypoints_all_frames.copy()

        def find_nearest_non_deviation(index, deviations, direction):
            step = 1 if direction == 'forward' else -1
            i = index + step
            while 0 <= i < len(keypoints_all_frames):
                if i not in deviations:
                    return keypoints_all_frames[i], i
                i += step
            return None, None

        for i in deviations:
            if i > 0 and i < len(keypoints_all_frames) - 1:
                prev_keypoints, _ = find_nearest_non_deviation(i, deviations, 'backward')
                next_keypoints, _ = find_nearest_non_deviation(i, deviations, 'forward')

                if prev_keypoints is not None and next_keypoints is not None:
                    corrected_frames[i] = np.mean([prev_keypoints, next_keypoints], axis=0).tolist()
                elif prev_keypoints is not None:
                    corrected_frames[i] = prev_keypoints
                elif next_keypoints is not None:
                    corrected_frames[i] = next_keypoints
                else:
                    print(f"Frame {i} could not be corrected")
        return corrected_frames
    
    # def detect_pose_deviation(self, keypoints_all_frames, centroid_std_factor=5, distance_std_factor=5):
    #     """
    #     Detect poses that deviate significantly from the previous pose based on keypoint and centroid differences.
        
    #     :param keypoints_all_frames: List of frames, each containing keypoints for a pose.
    #     :param centroid_std_factor: Factor of standard deviation for centroid threshold.
    #     :param distance_std_factor: Factor of standard deviation for keypoint distance threshold.
    #     :return: List of frame indices with detected deviations.
    #     """
    #     deviations = []
    #     centroid_distances = []
    #     keypoint_distance_diffs = []
        
    #     # Precompute centroids and distances
    #     for i in range(1, len(keypoints_all_frames)):
    #         prev_keypoints = np.array(keypoints_all_frames[i - 1])
    #         curr_keypoints = np.array(keypoints_all_frames[i])
            
    #         prev_centroid = self.calculate_centroid(prev_keypoints)
    #         curr_centroid = self.calculate_centroid(curr_keypoints)
            
    #         centroid_distance = np.linalg.norm(prev_centroid - curr_centroid)
    #         centroid_distances.append(centroid_distance)
            
    #         prev_distances = self.calculate_distances_from_centroid(prev_keypoints, prev_centroid)
    #         curr_distances = self.calculate_distances_from_centroid(curr_keypoints, curr_centroid)
            
    #         keypoint_distance_diff = np.mean(np.abs(prev_distances - curr_distances))
    #         keypoint_distance_diffs.append(keypoint_distance_diff)
        
    #             # Calculate the mean and standard deviation
    #         centroid_mean = np.mean(centroid_distances)
    #         centroid_std = np.std(centroid_distances)
    #         distance_mean = np.mean(keypoint_distance_diffs)
    #         distance_std = np.std(keypoint_distance_diffs)
            
    #         # print(f"Centroid distances - Mean: {centroid_mean}, Std: {centroid_std},Frame:{i}")
    #         # print(f"Keypoint distance differences - Mean: {distance_mean}, Std: {distance_std},Frame:{i}")
            
    #         # Calculate dynamic thresholds based on mean and standard deviation
    #         centroid_threshold = centroid_mean + centroid_std_factor * centroid_std
    #         distance_threshold = distance_mean + distance_std_factor * distance_std
    #     def calculate_deviation(prev_keypoints, curr_keypoints):
    #         """Calculate deviation based on both centroid and keypoint distance differences."""
    #         prev_centroid = self.calculate_centroid(prev_keypoints)
    #         curr_centroid = self.calculate_centroid(curr_keypoints)

    #         # Centroid deviation check
    #         centroid_distance = np.linalg.norm(prev_centroid - curr_centroid)
    #         #print(f"Centroid distance: {centroid_distance}")
    #         if centroid_distance > centroid_threshold:
    #             return True

    #         # Keypoint distance deviation check
    #         prev_distances = self.calculate_distances_from_centroid(prev_keypoints, prev_centroid)
    #         curr_distances = self.calculate_distances_from_centroid(curr_keypoints, curr_centroid)

    #         keypoint_distance_diff = np.mean(np.abs(prev_distances - curr_distances))
    #         #print(f"Keypoint distance difference: {keypoint_distance_diff}")
    #         return keypoint_distance_diff > distance_threshold

    #     def check_last_non_consecutive_deviation(deviations):
    #         """Find the last non-consecutive deviation in the list."""
    #         last_non_consecutive = deviations[0]
    #         for i in range(1, len(deviations)):
    #             if deviations[i] - deviations[i - 1] != 1:
    #                 last_non_consecutive = deviations[i - 1]
    #         return last_non_consecutive

    #     for i in range(1, len(keypoints_all_frames)):
    #         if deviations:
    #             # If deviations exist, check the last non-consecutive frame
    #             last_non_consecutive = check_last_non_consecutive_deviation(deviations)
    #             prev_keypoints = np.array(keypoints_all_frames[last_non_consecutive - 1])
    #         else:
    #             # Use the previous frame for comparison
    #             prev_keypoints = np.array(keypoints_all_frames[i - 1])
            
    #         curr_keypoints = np.array(keypoints_all_frames[i])
    #         #print(f"Checking frame {i}...")
    #         # Check if the pose deviates
    #         if calculate_deviation(prev_keypoints, curr_keypoints):
    #             deviations.append(i)

    #     return deviations
    
    
    def detect_pose_deviation(self,keypoints_all_frames, distance_threshold=50, centroid_threshold=50):
        """
        Detect poses that deviate significantly from the previous pose based on keypoint and centroid differences.
        
        :param keypoints_all_frames: List of frames, each containing keypoints for a pose.
        :param distance_threshold: Threshold for detecting significant deviation in keypoint distances.
        :param centroid_threshold: Threshold for detecting significant deviation in centroid movement.
        :return: List of frame indices with detected deviations.
        """
        deviations = []
        def calculate_distances_from_centroid(keypoints, centroid):
            """Calculate Euclidean distances of keypoints from the centroid."""
            return np.linalg.norm(keypoints - centroid, axis=1)
        
        def calculate_deviation(prev_keypoints, curr_keypoints):
            """Calculate deviation based on both centroid and keypoint distance differences."""
            prev_centroid = self.calculate_centroid(prev_keypoints)
            curr_centroid = self.calculate_centroid(curr_keypoints)

            # Centroid deviation check
            centroid_distance = np.linalg.norm(prev_centroid - curr_centroid)
            if centroid_distance > centroid_threshold:
                return True

            # Keypoint distance deviation check
            prev_distances = calculate_distances_from_centroid(prev_keypoints, prev_centroid)
            curr_distances = calculate_distances_from_centroid(curr_keypoints, curr_centroid)

            keypoint_distance_diff = np.mean(np.abs(prev_distances - curr_distances))
            return keypoint_distance_diff > distance_threshold

        def check_last_non_consecutive_deviation(deviations):
            """Find the last non-consecutive deviation in the list."""
            last_non_consecutive = deviations[0]
            for i in range(1, len(deviations)):
                if deviations[i] - deviations[i - 1] != 1:
                    last_non_consecutive = deviations[i - 1]
            return last_non_consecutive

        for i in range(1, len(keypoints_all_frames)):
            if deviations:
                # If deviations exist, check the last non-consecutive frame
                last_non_consecutive = check_last_non_consecutive_deviation(deviations)
                prev_keypoints = np.array(keypoints_all_frames[last_non_consecutive - 1])
            else:
                # Use the previous frame for comparison
                prev_keypoints = np.array(keypoints_all_frames[i - 1])
            
            curr_keypoints = np.array(keypoints_all_frames[i])
            
            # Check if the pose deviates
            if calculate_deviation(prev_keypoints, curr_keypoints):
                deviations.append(i)

        return deviations

    logging.basicConfig(level=logging.INFO)  
    # def correct_deviation(self, keypoints_all_frames, deviations, max_frame_diff=5):
    #     """
    #     Correct deviations by interpolating keypoints between nearest non-deviation frames.
        
    #     :param keypoints_all_frames: List of frames, each containing keypoints for a pose.
    #     :param deviations: List of frame indices with detected deviations.
    #     :param max_frame_diff: Maximum allowed frame difference for finding non-deviation frames.
    #     :return: List of frames with corrected keypoints.
    #     """
    #     if not deviations:
    #         logging.info("No deviations to correct.")
    #         return keypoints_all_frames  # No deviations to correct
        
    #     corrected_frames = keypoints_all_frames.copy()

    #     def find_nearest_non_deviation_frame(index, direction):
    #         """
    #         Find the nearest non-deviation frame in the specified direction ('forward' or 'backward').
            
    #         :param index: Current frame index with a deviation.
    #         :param direction: Search direction - either 'forward' or 'backward'.
    #         :return: Tuple of (keypoints from nearest valid frame, frame index), or (None, None) if not found.
    #         """
    #         step = 1 if direction == 'forward' else -1
    #         i = index + step
    #         while 0 <= i < len(keypoints_all_frames):
    #             if i not in deviations and abs(i - index) <= max_frame_diff:
    #                 return np.array(keypoints_all_frames[i]), i
    #             i += step
    #         return None, None

    #     for i in deviations:
    #         # Skip correcting first and last frame deviations
    #         if 0 < i < len(keypoints_all_frames) - 1:
    #             # Find the nearest non-deviation frames both forward and backward
    #             prev_keypoints, prev_frame = find_nearest_non_deviation_frame(i, 'backward')
    #             next_keypoints, next_frame = find_nearest_non_deviation_frame(i, 'forward')

    #             if prev_keypoints is not None and next_keypoints is not None:
    #                 # Interpolate the keypoints between nearest non-deviation frames
    #                 median_keypoints = np.mean([prev_keypoints, next_keypoints], axis=0)
    #                 corrected_frames[i] = median_keypoints.tolist()
                   

    #             elif prev_keypoints is not None and abs(prev_frame - i) <= max_frame_diff:
    #                 # Use only previous keypoints if next keypoints are unavailable or too far
    #                 corrected_frames[i] = prev_keypoints.tolist()
                   

    #             elif next_keypoints is not None and abs(next_frame - i) <= max_frame_diff:
    #                 # Use only next keypoints if previous keypoints are unavailable or too far
    #                 corrected_frames[i] = next_keypoints.tolist()
                   

    #             else:
    #                logging.error(f"Deviation at frame {i} could not be corrected due to lack of nearby non-deviation frames.")
        
    #     return corrected_frames
    
    def correct_deviation(self,keypoints_all_frames, deviations):
        """
        Correct deviations by calculating the median of points between previous and next frame.
        
        :param keypoints_all_frames: List of frames, each containing keypoints for a pose.
        :param deviations: List of frame indices with detected deviations.
        :return: List of frames with corrected keypoints.
        """
        corrected_frames = keypoints_all_frames.copy()
        
        for i in deviations:
            if 0 < i < len(keypoints_all_frames) - 1:
                prev_keypoints = np.array(keypoints_all_frames[i-1])
                curr_keypoints = np.array(keypoints_all_frames[i])
                next_keypoints = np.array(keypoints_all_frames[i+1])
                
                # Calculate the median pose from previous and next frames
                median_keypoints = np.mean(np.array([prev_keypoints, next_keypoints]), axis=0)
                
                # Replace the current frame with the median keypoints
                corrected_frames[i] = median_keypoints.tolist()
        
        return corrected_frames
        
        
 
