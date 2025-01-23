import re
import numpy as np

import re
import numpy as np

class BasePoseEstimator:
    """
    This class is a base class for all pose estimators.
    It constructs the skeleton structure of the human body to be used in other classes.
    """
    
    # Define a dictionary of body part synonyms (shared by all child classes)
    body_part_synonyms = {
        'wrist': ['wrist', 'hand'],
        'ankle': ['ankle', 'foot'],
        'hip': ['hip', 'thigh', 'upperthigh', 'uthigh'],
        'knee': ['knee', 'shin', 'uppershin'],
        'elbow': ['elbow', 'forearm', 'farm'],
        'shoulder': ['shoulder', 'upperarm', 'uarm'],
    }

    left_right_synonyms = {
        'left': ['left', 'l'],
        'right': ['right', 'r'],
    }

    def __init__(self, body_parts_list, connections):
        """
        Initialize the BasePoseEstimator with body parts and connections.
        
        Parameters:
        - body_parts_list: List of body parts.
        - connections: List of tuples representing connections between body parts.
        """
    
        
        # Normalize the body parts and connections
        self.labels = [self.normalize_keypoint(part) for part in body_parts_list]
        self.connections = [(self.normalize_keypoint(part1), self.normalize_keypoint(part2)) for part1, part2 in connections]
        
        
    def get_body_length(self,point1, point2):
        """Calculate the Euclidean distance between two points."""
        if point1 is None or point2 is None:
            return 0
        return np.linalg.norm(point1 - point2)

    def find_median_point(self,point1, point2):
        """Calculate the midpoint between two points."""
        if point1 is None:
            return point2
        if point2 is None:
            return point1
        return (point1 + point2) / 2

    def get_body_height(self,points3D):
        """
        Calculate the body height using available 3D points and body labels.
        
        Parameters:
        - points3D: 3D coordinates of body keypoints.
        - labels: List of body part labels.
        - pose_estimator: An instance of the BasePoseEstimator for normalizing body parts.
        
        Returns:
        - body_height: Estimated body height.
        """
        def get_point(label):
            """Helper function to safely get 3D point for a body part."""
            alternative_names = {
                'leftankle': ['leftankle', 'lankle'],
                'lefthip': ['lefthip', 'lhip'],
                'leftknee': ['leftknee', 'lknee'],
                'righthip': ['righthip', 'rhip'],
                'neck': ['neck'],
                'head': ['head'],
                'headtop': ['headtop', 'headtop']
            }

            # Check if the label exists in the alternative names
            for key, values in alternative_names.items():
                if label in values:
                    # Find the index of the standardized key
                    if key in self.labels:
                        return points3D[:3, self.labels.index(key)]
            
            return None

        # Extract relevant points, using normalized keypoints
        left_ankle = get_point('leftankle')
        left_hip = get_point('lefthip')
        left_knee = get_point('leftknee')
        right_hip = get_point('rigthhip')
        neck = get_point('neck')
        head = get_point('head')
        head_top = get_point('headtop')

        # Prioritize hip, if available, otherwise calculate it from left/right hips
        hip = get_point('hip')  # Use 'hip' if available
        if hip is None:
            hip = self.find_median_point(left_hip, right_hip)  # Fallback to median of left and right hips

        # Calculate body segments, safely handling missing points
        left_leg = self.get_body_length(left_ankle, left_knee) if left_ankle is not None and left_knee is not None else 0
        left_thigh = self.get_body_length(left_hip, left_knee) if left_hip is not None and left_knee is not None else 0
        torso = self.get_body_length(hip, neck) if hip is not None and neck is not None else 0
        head_segment = self.get_body_length(neck, head) if neck is not None and head is not None else 0
        head_top_segment = self.get_body_length(head, head_top) if head is not None and head_top is not None else 0

        # Calculate total body height
        body_height = left_leg + left_thigh + torso + head_segment + head_top_segment
        return body_height
        
    def normalize_keypoint(self, keypoint):
        """
        Normalize the keypoint by converting to lowercase, removing non-alphanumeric characters, 
        and applying synonym mapping if applicable. Retains the synonym key for left/right prefix.
        
        Parameters:
        - keypoint: The keypoint to normalize.
        
        Returns:
        - Normalized keypoint with synonym key for left/right prefix (if applicable).
        """
        # Remove spaces, underscores, and lowercase the keypoint
        keypoint = re.sub(r'[\s_]', '', keypoint).lower()

        # Check for left/right prefixes
        synonym_prefix = ""
        for side, synonyms in self.left_right_synonyms.items():
            for syn in synonyms:
                if keypoint.startswith(syn):
                    synonym_prefix = side
                    # Remove the left/right part from the keypoint
                    keypoint = keypoint[len(syn):]
                    break
            if synonym_prefix:
                break
        
        # Apply synonym mapping to the remaining keypoint
        for standard_name, synonyms in self.body_part_synonyms.items():
            if any(synonym in keypoint for synonym in synonyms):
                return f"{synonym_prefix}{standard_name}"
        
        # Return the original keypoint if no match is found
        return f"{synonym_prefix}{keypoint}"

    
class LimbAssignment(BasePoseEstimator):
    """
    This class extends BasePoseEstimator to assign limbs to the human body.
    It categorizes limbs into left and right based on the body parts.
    """
    
    def __init__(self, body_parts_list, connections, right_limbs=None, left_limbs=None):
        """
        Initialize the LimbAssignment with body parts, connections, and optional limb assignments.
        
        Parameters:
        - body_parts_list: List of body parts.
        - connections: List of tuples representing connections between body parts.
        - right_limbs: Optional dictionary of right limbs.
        - left_limbs: Optional dictionary of left limbs.
        """
        # Call the constructor of the base class
        super().__init__(body_parts_list, connections)
        self.right_limbs = right_limbs
        self.left_limbs = left_limbs
        
        # Assign limbs if not provided
        self.assign_limbs()
         
    def assign_limbs(self):
        """
        Assign limbs to the human body by categorizing connections into left and right limbs.
        """
        if self.left_limbs is None and self.right_limbs is None:
            self.left_limbs = {}
            self.right_limbs = {}
            left_counter = 1
            right_counter = 1

            for connection in self.connections:
                start, end = connection
                
                # Check if the connection is between opposite sides (e.g., left hip to right hip)
                is_opposite_sides = ('left' in start.lower() and 'right' in end.lower()) or \
                                    ('right' in start.lower() and 'left' in end.lower())
                
                # Skip connections that are between opposite sides (e.g., left hip to right hip)
                if is_opposite_sides:
                    continue
                
                # Check if the keypoint is a left limb
                is_left = start.startswith('l') or start.endswith('l') or 'left' in start.lower()

                # Check if the keypoint is a right limb
                is_right = start.startswith('r') or start.endswith('r') or 'right' in start.lower()

                if is_left:
                    limb_name = self.get_limb_name(start, end)
                    if limb_name is not None:  # Only add if limb_name is not None
                        self.left_limbs[f"left_{limb_name}"] = (start, end)
                    else:
                        self.left_limbs[f"left_body_part_{left_counter}"] = (start, end)
                        left_counter += 1
                elif is_right:
                    limb_name = self.get_limb_name(start, end)
                    if limb_name is not None:  # Only add if limb_name is not None
                        self.right_limbs[f"right_{limb_name}"] = (start, end)
                    else:
                        self.right_limbs[f"right_body_part_{right_counter}"] = (start, end)
                        right_counter += 1

    def get_limb_name(self, start, end):
        """
        Define logic to categorize limbs based on body parts.
        
        Parameters:
        - start: The starting body part of the connection.
        - end: The ending body part of the connection.
        
        Returns:
        - The name of the limb category or None if it cannot be categorized.
        """
        # Handle arm-related limbs
        if 'shoulder' in start or 'elbow' in start:
            return 'arm' if 'wrist' not in end else 'forearm'
        elif 'wrist' in start:
            return 'forearm' if 'elbow' in end else 'hand'  # Assume 'wrist' connects to 'hand' if not connecting to 'elbow'
        
        # Handle leg-related limbs
        elif 'hip' in start or 'knee' in start:
            return 'thigh' if 'ankle' not in end else 'leg'
        elif 'ankle' in start:
            return 'leg' if 'knee' in end else None  # Assume 'ankle' connects to 'foot' if not connecting to 'knee'
        
        return None
