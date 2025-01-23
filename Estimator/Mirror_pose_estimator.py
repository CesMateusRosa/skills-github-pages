import numpy as np
import cv2
import re
import importlib
import Base_Pose_Estimator
importlib.reload(Base_Pose_Estimator)
from Base_Pose_Estimator import BasePoseEstimator, LimbAssignment
class MirrorPoseEstimator(BasePoseEstimator):
    def __init__(self, body_parts_list, connections,width, height, n_iter=100, threshold=1,d=2, focal_lengths=None):
        """
        Initialize the MirrorPoseEstimator with body parts list, connections, height, width, distance, and optional focal lengths.
        """
        super().__init__(body_parts_list, connections)
        print(self.labels)
        self.limb_assignment = LimbAssignment(self.labels, self.connections)
        
        self.left_limbs = {k: v for k, v in self.limb_assignment.left_limbs.items() if not k.startswith('left_body_part_')}
        self.right_limbs = {k: v for k, v in self.limb_assignment.right_limbs.items() if not k.startswith('right_body_part_')}
        
        
        # self.left_limbs = {k: v for k, v in self.limb_assignment.left_limbs.items()}
        # self.right_limbs = {k: v for k, v in self.limb_assignment.right_limbs.items()}
        
        self.focal_lengths = focal_lengths or [1,12, 18, 32, 34, 36, 38, 40, 100, 120, 180, 320, 340, 360, 380, 400, 
                                                500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 
                                                1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 
                                                2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3800, 4000]
        self.d = d
        self.height = height
        self.width = width
        self.n_iter=n_iter
        self.threshold=threshold
        
        
        
    def transpose_if_needed(self, points):
        """
        Ensure points are in the correct shape (2, N). If they are in shape (N, 2), transpose them.
        """
        points = np.array(points)
        if points.shape[0] != 2 and points.shape[1] == 2:
            points = points.T
        return points
    
    def calculate_normal_vector(self, points1, points2):
        """
        Calculate the normal vector using points1 and points2.
        """
        N = np.empty((0, 3))
        for pts1, pts2 in zip(points1, points2):
            x1, y1 = pts1
            x2, y2 = pts2
            A = np.array([-y2 + y1, (x2 - x1), (x1 * y2 - x2 * y1)])
            N = np.vstack((N, A))
        
        u, s, vh = np.linalg.svd(N)
        min_singular_value_index = np.argmin(s)
        v = vh.T
        normal = v[:, -1]
        normal = normal / np.linalg.norm(normal)
        if normal[2] <0:
            normal = -normal
        
        return normal
    
    
    
    def normalize_points(self,points):
            """
            Normalize points by translating them so that their centroid is at the origin
            and scaling them so that the average distance from the origin is sqrt(2).
            
            Parameters:
            points (ndarray): Nx2 array of points to be normalized.
            
            Returns:
            normalized_points (ndarray): Nx2 array of normalized points.
            T (ndarray): 3x3 normalization matrix.
            
            Raises:
            ValueError: If the input points array does not have the correct shape.
            """
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("Input points must be a Nx2 array.")
            
            # Compute centroid
            centroid = np.mean(points, axis=0)
            
            # Shift points to the centroid
            shifted_points = points - centroid
            
            # Compute the average distance of the points from the origin
            mean_distance = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))
            
            # Compute the scaling factor so that the average distance is sqrt(2)
            scale = np.sqrt(2) / mean_distance
            
            # Construct the normalization matrix
            T = np.array([[scale, 0, -scale * centroid[0]],
                        [0, scale, -scale * centroid[1]],
                        [0, 0, 1]])
            
            # Apply normalization to the points
            ones_column = np.ones((points.shape[0], 1))
            points_homogeneous = np.hstack((points, ones_column))
            normalized_points = (T @ points_homogeneous.T).T
            
            return normalized_points[:, :2], T

    def get_rotation_matrix_mirror(self, n):
        """
        Compute the rotation matrix for mirroring based on the normal vector n.
        """
        I = np.eye(len(n))  # Identity matrix
        n = np.array(n).reshape(-1, 1)  # Reshape n to a column vector
        R = I - 2 * np.dot(n, n.T)
        return R

    def get_translation_vector_mirror(self, N):
        """
        Compute the translation vector for mirroring based on the normal vector N and distance d.
        """
        return 2 * N * self.d

    def get_virtual_camera_matrix(self, R, T, K):
        """
        Compute the virtual camera matrix using rotation matrix R, translation vector T, and intrinsic matrix K.
        """
        return np.dot(K, np.hstack((R, T.reshape(-1, 1))))

    def triangulate_points_mirror(self, points1, points2, P, P1):
        """
        Triangulate points from two views using the projection matrices P and P1.
        """
        points1=self.transpose_if_needed(points1)
        points2=self.transpose_if_needed(points2)
      
        
        points = cv2.triangulatePoints(P, P1, points1, points2)
        points3D = points / points[3]
        return points3D

    def homogenous_to_cartesian(self, points):
        """
        Convert homogeneous coordinates to Cartesian coordinates.
        """
        x_points1 = points[0, :] / points[2, :]
        y_points1 = points[1, :] / points[2, :]
        point = [(x, y) for x, y in zip(x_points1, y_points1)]
        return point

    def cartesian_to_homogenous(self, points):
        """
        Convert Cartesian coordinates to homogeneous coordinates.
        """
        points_array = np.array(points)
       
        
        points_homogeneous = [np.array([point[0], point[1], 1]) for point in points_array]
        return np.array(points_homogeneous).T

    def skew_matrix(self, v):
        """
        Compute the skew-symmetric matrix of vector v.
        """
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def get_residue_ransac(self, points1, points2, N, R):
        """
        Compute the residue for RANSAC based on points1, points2, normal vector N, and rotation matrix R.
        """
        Residue = []
        points1 = self.cartesian_to_homogenous(points1)
        points2 = self.cartesian_to_homogenous(points2)
        
        for i in range(len(points2[0])):
            residue = points1.T[i] @ self.skew_matrix(N) @ R @ points2[:, i]
            Residue.append(residue)
        return Residue

    def ransac(self, points1, points2, K, n_iter=None, t=None):
        """
        Perform RANSAC to estimate the best normal vector N and inliers.
        """
        
        num_points = points1.shape[0]
        maxnum_inliers = 0
        best_N = None
        best_inliers = []
        n_iter = n_iter if n_iter is not None else self.n_iter
        t = t if t is not None else self.threshold
        
        for _ in range(n_iter):
            idx = np.random.choice(num_points, 2, replace=False)
            N = self.calculate_normal_vector(points1[idx], points2[idx])
            R = self.get_rotation_matrix_mirror(N)
            Residue = self.get_residue_ransac(points1, points2, N, R)
          
            inliers = 0
            inlier_positions = []
            for j, res in enumerate(Residue):
                if abs(res) < t:
                    inliers += 1
                    inlier_positions.append(j)
            if inliers > maxnum_inliers:
                maxnum_inliers = inliers
                best_N = N
                best_inliers = inlier_positions
        
        if not best_inliers:
            return None
        else:
            best_N = self.calculate_normal_vector(points1[best_inliers], points2[best_inliers])
            return best_N, best_inliers
        
       
    def optimal_focal_length(self, points1, points2,max_body_height,mim_body_height):
        """
        Determine the optimal focal length by iterating over a range of focal lengths,
        computing the transformation and triangulation, and selecting the focal length
        with the best score.
            """
        
        max_score = 0
        best_focal_length = None
       

        pts1 = np.array(points1)
      
        pts2 = np.array(points2)
      
        for f in self.focal_lengths:
            
            K1 = np.array([[f, 0, self.width / 2],
                           [0, f,  self.height / 2],
                           [0, 0, 1]])
            
        
            
            points_mirror_homogeneous = self.cartesian_to_homogenous(pts1)
            points_realworld_homogeneous = self.cartesian_to_homogenous(pts2)
            points1_intrinsic = points_realworld_homogeneous.T @ np.linalg.inv(K1).T
            points2_intrinsic = np.linalg.inv(K1) @ points_mirror_homogeneous

            pts1_intrinsic = np.array(self.homogenous_to_cartesian(points1_intrinsic.T))
            pts2_intrinsic = np.array(self.homogenous_to_cartesian(points2_intrinsic))   

            N, _ = self.ransac(pts1_intrinsic, pts2_intrinsic, K1, n_iter=self.n_iter, t=self.threshold)
         
            if N is None:
                continue
            else:
                R = self.get_rotation_matrix_mirror(N)
                T = self.get_translation_vector_mirror(N)
                P2 = self.get_virtual_camera_matrix(R, T, K1)
                
                P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(-1, 1)))
                P1 = K1 @ P1
             
                points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
                left_distances = []
                right_distances = []

                # Calculate distances for each limb
                for limb, (start_label, end_label) in self.left_limbs.items():
                   
                    start_point = points3D[:3, self.labels.index(start_label)]
                    end_point = points3D[:3, self.labels.index(end_label)]
                    distance = np.linalg.norm(end_point - start_point)
                    left_distances.append(distance)

                # Calculate distances for each right limb
                for limb, (start_label, end_label) in self.right_limbs.items():
                    # print(limb)
                    # print(start_label)
                    # print(end_label)
                    # print("")
                    start_point = points3D[:3, self.labels.index(start_label)]
                    end_point = points3D[:3, self.labels.index(end_label)]
                    distance = np.linalg.norm(end_point - start_point)
                    right_distances.append(distance)
                    
                
                   # Calculate the score by comparing left and right limb lengths
                limb_scores = []
                for left_limb, right_limb in zip(left_distances, right_distances):
                    limb_score = min(left_limb, right_limb) / max(right_limb, left_limb)
                    limb_scores.append(limb_score)
                
                # Calculate the mean score for this focal length
                score = np.mean(limb_scores)
                
                
                body_height =self.get_body_height(points3D)
                
            
                
                    
                if body_height > max_body_height:
                    penalty = 1.0 - (body_height - max_body_height) / max_body_height # Decreases score as height exceeds 2.10 meters
                    score *= max(0.1, penalty)  # Ensure score doesn't go below 0.1

                # Apply a penalty if body height is less than 1.50 meters
                elif body_height < mim_body_height:
                    penalty = 1.0 - (mim_body_height - body_height) / mim_body_height  # Decreases score as height is below 1.50 meters
                    score *= max(0.1, penalty)  # Ensure score doesn't go below 0.1

                    
                # print(f"Adjusted Score for focal length {f}: {score}\n")
                
                # Update the best focal length if this score is higher
                if score > max_score:
                    max_score = score
                    best_focal_length = f
        if best_focal_length is None:
            best_focal_length = 1
        return best_focal_length,max_score
    
    # def optimal_focal_length(self, points1, points2):
    #     """
    #     Determine the optimal focal length by iterating over a range of focal lengths,
    #     computing the transformation and triangulation, and selecting the focal length
    #     with the best score.
    #     """
        
    #     max_score = 0
    #     best_focal_length = None
       

    #     pts1 = np.array(points1)
    #     pts2 = np.array(points2)

    #     for f in self.focal_lengths:
    #         K1 = np.array([[f, 0, 640],
    #                        [0, f, 240],
    #                        [0, 0, 1]])
            
    #         points_mirror_homogeneous = self.cartesian_to_homogenous(points1)
    #         points_realworld_homogeneous = self.cartesian_to_homogenous(points2)
    #         points1_intrinsic = points_realworld_homogeneous.T @ np.linalg.inv(K1).T
    #         points2_intrinsic = np.linalg.inv(K1) @ points_mirror_homogeneous

    #         pts1_intrinsic = np.array(self.homogenous_to_cartesian(points1_intrinsic.T))
    #         pts2_intrinsic = np.array(self.homogenous_to_cartesian(points2_intrinsic))   

    #         N, _ = self.ransac(pts1_intrinsic, pts2_intrinsic, K1, n_iter=30, t=0.1)
    #         if N is None:
    #             continue
    #         else:
    #             R = self.get_rotation_matrix_mirror(N)
    #             T = self.get_translation_vector_mirror(N)
    #             P2 = self.get_virtual_camera_matrix(R, T, K1)
    #             P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(-1, 1)))
    #             P1 = K1 @ P1
    #             points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
    #             left_distances = []
    #             right_distances = []

    #             # Calculate distances for each limb
    #             for limb, (start_label, end_label) in self.left_limbs.items():
    #                 start_point = points3D[:3, self.labels.index(start_label)]
    #                 end_point = points3D[:3, self.labels.index(end_label)]
    #                 distance = np.linalg.norm(end_point - start_point)
    #                 left_distances.append(distance)

    #             # Calculate distances for each right limb
    #             for limb, (start_label, end_label) in self.right_limbs.items():
    #                 start_point = points3D[:3, self.labels.index(start_label)]
    #                 end_point = points3D[:3, self.labels.index(end_label)]
    #                 distance = np.linalg.norm(end_point - start_point)
    #                 right_distances.append(distance)
                
    #             score = np.mean([min(ld, rd) / max(ld, rd) for ld, rd in zip(left_distances, right_distances)])
    #             if score > max_score:
    #                 max_score = score
    #                 best_focal_length = f

    #     return best_focal_length,max_score

    
    
    def create_calibration_matrix(self, points1, points2, max_height=10,min_height=0.1):
        """
        Create a calibration matrix for the camera.

        Parameters:
        focal_length (float): The focal length of the camera.
        width (int): The width of the image.
        height (int): The height of the image.

        Returns:
        np.ndarray: The camera calibration matrix.

        """
  
        points1 = np.array(points1)
        points2 = np.array(points2)
        
        focal_length,_=self.optimal_focal_length(points1,points2,max_height,min_height)
   
        K = np.array([
            [focal_length, 0, self.width / 2],
            [0, focal_length, self.height / 2],
            [0, 0, 1]
        ])
        
        return K,focal_length
    
    def compute_normalized_normal_vector(self, points1, points2):
        
        #Calculate the Normal Vector using RANSAC for a single frame
        K,_= self.create_calibration_matrix(points1,points2)
        
        points_mirror_homogeneous = self.cartesian_to_homogenous(points1)
        points_realworld_homogeneous = self.cartesian_to_homogenous(points2)
        
        # Convert points to intrinsic coordinates
        points1_intrinsic = points_realworld_homogeneous.T @ np.linalg.inv(K).T
        points2_intrinsic = np.linalg.inv(K) @ points_mirror_homogeneous
        
        pts1_intrinsic = np.array(self.homogenous_to_cartesian(points1_intrinsic.T))
        pts2_intrinsic = np.array(self.homogenous_to_cartesian(points2_intrinsic)) 
        
        # Apply RANSAC to estimate the normal vector
        N, _ = self.ransac(pts1_intrinsic, pts2_intrinsic, K, n_iter=100, t=1)
        if N is None:
            raise ValueError("RANSAC failed to find a valid normal vector.")
        return N
    
    def get_3D_estimation(self, points_real_world, points_mirror):
        """
        Calculate the final 3D points and compute the intermediate steps.

        Parameters:
        points_real_world_all (list of np.ndarray): List of 2D points in the real world coordinates.
        points_mirror_all (list of np.ndarray): List of 2D points in the mirror coordinates.
        focal_length (float): The focal length of the camera.
        width (int): The width of the image.
        height (int): The height of the image.
        d (float): The distance between the mirror and the camera.

        Returns:
        np.ndarray: The final 3D points.
        
        """
       
        K,focal_length = self.create_calibration_matrix(points_real_world, points_mirror)
        print(K)
        points_mirror_homogeneous = self.cartesian_to_homogenous(points_mirror)
        points_realworld_homogeneous = self.cartesian_to_homogenous(points_real_world)
        
        # Convert points to intrinsic coordinates
        points1_intrinsic = points_realworld_homogeneous.T @ np.linalg.inv(K).T
        points2_intrinsic = np.linalg.inv(K) @ points_mirror_homogeneous
        
        pts1_intrinsic = np.array(self.homogenous_to_cartesian(points1_intrinsic.T))
        pts2_intrinsic = np.array(self.homogenous_to_cartesian(points2_intrinsic)) 
        
        # Apply RANSAC to estimate the normal vector
        N, _ = self.ransac(pts1_intrinsic, pts2_intrinsic, K, n_iter=self.n_iter, t=self.threshold)
 
        if N is None:
            raise ValueError("RANSAC failed to find a valid normal vector.")
        
        R = self.get_rotation_matrix_mirror(N)
        T = self.get_translation_vector_mirror(N)
        
        # Get the camera projection matrices
        P2 = self.get_virtual_camera_matrix(R, T, K)
        P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(-1, 1)))
        P1 = K @ P1
      
        # Triangulate points to get 3D coordinates
        pts1 = np.array(points_real_world)
      
        pts2 = np.array(points_mirror)
        pts1[:,1]=self.height-pts1[:,1]
        pts2[:,1]=self.height-pts2[:,1]
        points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
        
        return points3D,focal_length
    
    
    
    
    
    def triangulate_video(self, points_real_world_all, points_mirror_all,max_height=100,min_height=0):
        
        triangulated_points = []
        

        # focal_lengths_first_15 = []

        # # Select the first 5 frames
        # for i in range(5):
        #     f, _ = self.optimal_focal_length(points_real_world_all[i], points_mirror_all[i], max_height, min_height)
        #     focal_lengths_first_15.append(f)

        # # Select 10 more frames evenly spaced throughout the length
        # total_frames = len(points_real_world_all)
        # spacing = total_frames // 10

        # for i in range(5, total_frames, spacing):
        #     f, _ = self.optimal_focal_length(points_real_world_all[i], points_mirror_all[i], max_height, min_height)
        #     focal_lengths_first_15.append(f)

        # # Calculate the mean focal length
        # mean_focal_length = np.mean(focal_lengths_first_15)

        # # Compute the inverse of the absolute differences from the mean
        # differences = np.abs(np.array(focal_lengths_first_15) - mean_focal_length)

        # # Add a small epsilon to avoid division by zero (in case any focal length is exactly the mean)
        # epsilon = 1e-6
        # weights = 1 / (differences + epsilon)

        # # Normalize the weights to sum to 1
        # weights /= np.sum(weights)

        # # Compute the weighted average of the focal lengths
        # weighted_f = np.sum(np.array(focal_lengths_first_15) * weights)
        
        # f=weighted_f
        
    
        
        # K = np.array([[f, 0, self.width / 2],
        #               [0, f, self.height / 2],
        #               [0, 0, 1]])
        
        
        K,f = self.create_calibration_matrix(points_real_world_all[0], points_mirror_all[0])
        print(K)
        points_mirror_homogeneous = self.cartesian_to_homogenous(points_mirror_all[0])
        points_realworld_homogeneous = self.cartesian_to_homogenous(points_real_world_all[0])
        points1_intrinsic = points_realworld_homogeneous.T @ np.linalg.inv(K).T
        points2_intrinsic = np.linalg.inv(K) @ points_mirror_homogeneous
        
        pts1_intrinsic = np.array(self.homogenous_to_cartesian(points1_intrinsic.T))
        pts2_intrinsic = np.array(self.homogenous_to_cartesian(points2_intrinsic))
        
        N, _ = self.ransac(pts1_intrinsic, pts2_intrinsic, K, n_iter=100, t=1)
        R = self.get_rotation_matrix_mirror(N)
        T = self.get_translation_vector_mirror(N)
        P2 = self.get_virtual_camera_matrix(R, T, K)
        P1 = np.identity(3)
        P1 = np.hstack((P1, np.array([0, 0, 0]).reshape(-1, 1)))
        P1 = K @ P1
        
        pts1 = np.array(points_real_world_all[0])
        pts2 = np.array(points_mirror_all[0])
        pts1[:,1]=self.height-pts1[:,1]
        pts2[:,1]=self.height-pts2[:,1]
        points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
        triangulated_points.append(points3D)
        
        for points_realworld, points_mirror in zip(points_real_world_all[1:], points_mirror_all[1:]):
            pts1 = np.array(points_realworld)
            pts2 = np.array(points_mirror)
            pts1[:,1]=self.height-pts1[:,1]
            pts2[:,1]=self.height-pts2[:,1]
            points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
            triangulated_points.append(points3D)
        
        return np.array(triangulated_points),f
    
    def triangulate_video_gen(self, points_real_world_all, points_mirror_all,max_height=100,min_height=0):
        
        triangulated_points = []
        

      

                
        K,f= self.create_calibration_matrix(points_real_world_all[0], points_mirror_all[0])
        points_mirror_homogeneous = self.cartesian_to_homogenous(points_mirror_all[0])
        points_realworld_homogeneous = self.cartesian_to_homogenous(points_real_world_all[0])
        points1_intrinsic = points_realworld_homogeneous.T @ np.linalg.inv(K).T
        points2_intrinsic = np.linalg.inv(K) @ points_mirror_homogeneous
        
        pts1_intrinsic = np.array(self.homogenous_to_cartesian(points1_intrinsic.T))
        pts2_intrinsic = np.array(self.homogenous_to_cartesian(points2_intrinsic))
        
        N, _ = self.ransac(pts1_intrinsic, pts2_intrinsic, K, n_iter=100, t=1)
        R = self.get_rotation_matrix_mirror(N)
        T = self.get_translation_vector_mirror(N)
        P2 = self.get_virtual_camera_matrix(R, T, K)
        P1 = np.identity(3)
        P1 = np.hstack((P1, np.array([0, 0, 0]).reshape(-1, 1)))
        P1 = K @ P1
        
        pts1 = np.array(points_real_world_all[0])
        pts2 = np.array(points_mirror_all[0])
        pts1[:,1]=self.height-pts1[:,1]
        pts2[:,1]=self.height-pts2[:,1]
        points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
        triangulated_points.append(points3D)
        
        for points_realworld, points_mirror in zip(points_real_world_all[1:], points_mirror_all[1:]):
            pts1 = np.array(points_realworld)
            pts2 = np.array(points_mirror)
            pts1[:,1]=self.height-pts1[:,1]
            pts2[:,1]=self.height-pts2[:,1]
            points3D = self.triangulate_points_mirror(pts1, pts2, P1, P2)
            yield points3D
        
    
