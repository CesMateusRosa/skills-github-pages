import re
import numpy as np
import matplotlib.pyplot as plt
from Base_Pose_Estimator import BasePoseEstimator

class OcclusionEstimator(BasePoseEstimator):
    def __init__(self, body_parts_list, connections):
       
        super().__init__(body_parts_list, connections)
        self.body_segments = self.create_body_segments(self.connections)
        
  
    def create_body_segments(self,connections):
        body_segments = {}
        for part1, part2 in connections:
            if part1 not in body_segments:
                body_segments[part1] = []
            if part2 not in body_segments:
                body_segments[part2] = []
            
            body_segments[part1].append(part2)
            body_segments[part2].append(part1)
        return body_segments
    
    def calculate_vector(self, p1, p2):
        return np.array([-(p2[0] - p1[0]), p2[1] - p1[1]])
    
    def reflect_vector(self,vector,normal,direction):
            # Normalize the normal vector to ensure it's a unit vector
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        direction=self.normalize_keypoint(direction)
        # Reflect the vector: v_reflected = v - 2 * (v . n) * n
        vector = np.array(vector)
        if direction=="mirrortoworld":
            return vector - 2 * np.dot(vector, normal) * normal
        elif direction=="worldtomirror":
            return vector + 2 * np.dot(vector, normal) * normal
       

    def calculate_segment_vector(self,connection):
        return self.calculate_vector(connection[0], connection[1])
    
    def calculate_intersection_point(self, line1_point1, line1_point2, line2_point1, line2_point2):
        p1 = np.array(line1_point1)
        p2 = np.array(line1_point2)
        p3 = np.array(line2_point1)
        p4 = np.array(line2_point2)
        
        d1 = p2 - p1
        d2 = p4 - p3
        
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denominator) < 1e-10:  # Handle near-parallel lines
            return None  # Lines are parallel or coincident
        
        t1 = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denominator
        intersection = p1 + t1 * d1
        return intersection.tolist()

    def find_epipole(self, points1, points2):
        epipole = []
        lines = []

        # Use all pairs of points to compute intersections
        for i in range(len(points1)):
            for j in range(i + 1, len(points1)):  # Ensure unique pairs
                intersection = self.calculate_intersection_point(points1[i], points2[i], 
                                                                points1[j], points2[j])
                if intersection:
                    epipole.append(intersection)
                    lines.append([points1[i], points2[i], points1[j], points2[j]])

        if epipole:
            return np.median(epipole, axis=0), lines  # Return the mean of all intersection points
        else:
            return None, []
        
    def plot_pose_epipole(self,image, points2D, points2D_2, epipole, epipole_lines):
        fig, ax = plt.subplots()
        if image is not None:
            ax.imshow(image)
        #ax.invert_yaxis()
        for connection in self.connections:
                color = 'red' if 'left' in connection[0] else 'blue'
                color_2 = 'pink' if 'left' in connection[0] else 'lightblue'
                ax.plot([points2D[self.labels.index(connection[0])][0], 
                             points2D[self.labels.index(connection[1])][0]], 
                            [points2D[self.labels.index(connection[0])][1], 
                             points2D[self.labels.index(connection[1])][1]], color=color)

                ax.plot([points2D_2[self.labels.index(connection[0])][0], 
                             points2D_2[self.labels.index(connection[1])][0]], 
                            [points2D_2[self.labels.index(connection[0])][1], 
                             points2D_2[self.labels.index(connection[1])][1]], color=color_2)

            # New code to draw lines used for epipole calculation
        for line in epipole_lines:
                line1_point1, line1_point2, line2_point1, line2_point2 = line
                ax.plot([line1_point1[0], epipole[0]], [line1_point1[1], epipole[1]], 'g--')
                ax.plot(line1_point2[0], line1_point2[1], 'mo')
                ax.plot([line2_point1[0], epipole[0]], [line2_point1[1], epipole[1]], 'g--')
                ax.plot(line2_point2[0], line2_point2[1], 'mo')
                
        ax.plot(epipole[0], epipole[1], 'ro')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()
        
        
    def compute_epiline(self, epipole, point,extension_factor=1000):
        return np.cross(epipole, point)*extension_factor
    
    def compute_intersection_vector_line(self, application_point, vector, line_point1, line_point2, extension_factor=1000):
        """
        Computes the intersection of an extended vector from the application point with an extended line.
        
        Parameters:
        - application_point: The starting point of the vector (list or np.array of 2D coordinates).
        - vector: The direction vector to be extended from the application point (list or np.array of 2D coordinates).
        - line_point1: First point on the line.
        - line_point2: Second point on the line.
        - extension_factor: Factor by which the line is extended in both directions.
        
        Returns:
        - Intersection point (list), or raises an error for parallel lines.
        """
        # Convert points to numpy arrays
        ap = np.array(application_point)
        v = np.array(vector)
        lp1 = np.array(line_point1)
        lp2 = np.array(line_point2)
        
        # Extend the line by scaling its direction
        line_dir = lp2 - lp1
        lp1_extended = lp1 - extension_factor * line_dir
        lp2_extended = lp2 + extension_factor * line_dir
        
        # Recalculate the direction vector for the extended line
        extended_line_dir = lp2_extended - lp1_extended
        
        # Parametric equation for the vector: ap + t * v
        # Parametric equation for the extended line: lp1_extended + u * (lp2_extended - lp1_extended)
        
        # Solving for t and u where both parametric equations intersect:
        denominator = v[0] * extended_line_dir[1] - v[1] * extended_line_dir[0]
        
        # Check for parallel lines
        if abs(denominator) < 1e-10:
            raise ValueError("Parallel lines: No intersection can be computed.")
        
        # Calculate the parameter t for the vector and u for the extended line when they intersect
        t = ((lp1_extended[0] - ap[0]) * extended_line_dir[1] - (lp1_extended[1] - ap[1]) * extended_line_dir[0]) / denominator
        
        # Calculate the intersection point using the parametric equation of the vector
        intersection = ap + t * v
        
        # Return the intersection point as a list
        return tuple(intersection)
        
        
    def find_corresponding_body_joint(self,missing_point,occluded_list=None):
        # Find the corresponding point in the body segments
        missing_segment = self.body_segments[missing_point]
        for segment in missing_segment:
            # if segment not in occluded_list:
                return segment
        return None
        
    def estimate_occlusion(self, missing_point, points_missing, points_correct, epipole,normal,direction):
            missing_point=self.normalize_keypoint(missing_point)
            #Computes the 2D coordinates of the occluded point.
            #Parameters:
            #- missing_point: String with the name of the missing point.
            #- points2D: Set of 2D coordinates for the body with occluded point.
            #- points2D_2: Set of 2D coordinates for the body without occluded point.
            #- epipole: 2D coordinates of the epipole.
            
            #Returns: 
            #- 2D coordinates of the occluded point.
            
            # Find the index of the missing point in the body parts list
            missing_index = self.labels.index(self.normalize_keypoint(missing_point))
            
            # Find the corresponding point in the body segments
            application_point=self.find_corresponding_body_joint(missing_point)
            print(application_point)
            aplication_index = self.labels.index(application_point)
            
            # Compute the direction vector from the application point to the missing point  
            vector=self.calculate_vector(points_correct[aplication_index], points_correct[missing_index])
            reflected_vector=self.reflect_vector(vector,np.array(normal),direction)
            
            # Compute the intersection point of the vector with the epipolar line
            intersection_point=self.compute_intersection_vector_line(points_missing[aplication_index], reflected_vector, epipole, points_correct[missing_index])
            
            return intersection_point
    
        
        
    def plot_occlusion_pose(self, missing_point, points_missing, points_correct, epipole, normal, direction,image=None):
        """
        Plots the body pose with the estimated occlusion point, epipolar line, and vector between the application point and the missing point.

        Parameters:
        - missing_point: String, name of the missing keypoint.
        - points_missing: 2D coordinates for the body with occluded point.
        - points_correct: 2D coordinates for the body without occluded point (reference points).
        - epipole: 2D coordinates of the epipole.
        - normal: 2D normal vector for reflection.
        - image: Optional background image for the plot.
        """
        fig, ax = plt.subplots()

        # Check if the image is a valid 2D or 3D array before attempting to plot it
        if image is not None and len(image.shape) in [2, 3]:
            ax.imshow(image)
        else:
            print("No valid image provided, skipping background image.")

        # Normalize the keypoint name to ensure consistency in indexing
        missing_point = self.normalize_keypoint(missing_point)
        
        # Find the index of the missing point in the body parts list
        missing_index = self.labels.index(missing_point)

        # Find the corresponding application point
        application_point_name = self.find_corresponding_body_joint(missing_point, points_missing)
        application_index = self.labels.index(application_point_name)

        # Calculate the direction vector from the application point to the missing point
        vector = self.calculate_vector(points_correct[application_index], points_correct[missing_index])

        # Reflect the vector using the provided normal vector
        reflected_vector = self.reflect_vector(vector, np.array(normal),direction)

        # Compute the intersection point of the reflected vector and the epipolar line
        intersection_point = self.compute_intersection_vector_line(points_missing[application_index], 
                                                                reflected_vector, 
                                                                epipole, 
                                                                points_correct[missing_index])

        # Plot the correct pose (reference)
        for connection in self.connections:
            color = 'blue' if 'left' in connection[0] else 'green'
            ax.plot([points_correct[self.labels.index(connection[0])][0], 
                    points_correct[self.labels.index(connection[1])][0]], 
                    [points_correct[self.labels.index(connection[0])][1], 
                    points_correct[self.labels.index(connection[1])][1]], color=color)

        # Plot the pose with the missing point
        for connection in self.connections:
            color = 'red' if 'left' in connection[0] else 'orange'
            ax.plot([points_missing[self.labels.index(connection[0])][0], 
                    points_missing[self.labels.index(connection[1])][0]], 
                    [points_missing[self.labels.index(connection[0])][1], 
                    points_missing[self.labels.index(connection[1])][1]], color=color, linestyle='--')

        # Plot the epipole and the intersection point (estimated occluded point)
        ax.plot(epipole[0], epipole[1], 'go', label='Epipole')
        ax.plot(intersection_point[0], intersection_point[1], 'ro', label='Estimated Occluded Point')

        # Plot the epipolar line between the epipole and the application point
        epipole_point = points_correct[missing_index]
        ax.plot([epipole[0], epipole_point[0]], 
                [epipole[1], epipole_point[1]], 'b--', label='Epipolar Line')

        # Plot the vector between the application point and the intersection point (estimated occlusion)
        application_point=points_missing[application_index]
        ax.arrow(application_point[0], application_point[1], 
                intersection_point[0] - application_point[0], 
                intersection_point[1] - application_point[1], 
                head_width=5, head_length=5, fc='purple', ec='purple', label='Application-to-Occlusion Vector')

        # Plot the reflected vector
        ax.arrow(application_point[0], application_point[1], 
                reflected_vector[0], reflected_vector[1], 
                head_width=5, head_length=5, fc='cyan', ec='cyan', label='Reflected Vector')
        if image is None:
            ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
        plt.title('Pose Estimation with Occluded Point, Epipolar Line, and Vectors')
        plt.show()
          
    def plot_intersection_with_line(self, application_point, vector, line_point1, line_point2, intersection_point, image=None):
        """
        Plots the vector from the application point and its intersection with an extended line.

        Parameters:
        - application_point: The starting point of the vector (list or np.array of 2D coordinates).
        - vector: The direction vector (list or np.array of 2D coordinates).
        - line_point1: First point on the line (list or np.array of 2D coordinates).
        - line_point2: Second point on the line (list or np.array of 2D coordinates).
        - intersection_point: The computed intersection point between the vector and the extended line.
        - image: Optional background image for the plot.
        """
        fig, ax = plt.subplots()
        
        if image is not None:
            ax.imshow(image)
        
        ax.plot(application_point[0], application_point[1], 'ro', label='Application Point')
        vector_endpoint = application_point + vector
        ax.plot([application_point[0], vector_endpoint[0]], [application_point[1], vector_endpoint[1]], 'g--', label='Vector')
        ax.plot([line_point1[0], line_point2[0]], [line_point1[1], line_point2[1]], 'b-', label='Line Segment')
        ax.plot(intersection_point[0], intersection_point[1], 'mo', label='Intersection Point')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.invert_yaxis()
        ax.legend()
        plt.title("Intersection of Vector with Line")
        plt.show()