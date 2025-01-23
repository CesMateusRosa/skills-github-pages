from Base_Pose_Estimator import BasePoseEstimator,LimbAssignment
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import Base_Pose_Estimator
importlib.reload(Base_Pose_Estimator)
from Base_Pose_Estimator import BasePoseEstimator, LimbAssignment
import scipy.optimize as opt

class FundamentalMatrixPoseEstimator(BasePoseEstimator):
    def __init__(self, body_parts_list, connections, width,height,n_iter=150,threshold=1,d=2,focal_lengths=None):
        super().__init__(body_parts_list, connections)
        self.d=d
        self.limb_assignment = LimbAssignment(self.labels, self.connections)
        self.left_limbs = {k: v for k, v in self.limb_assignment.left_limbs.items() if not k.startswith('left_body_part_')}
        self.right_limbs = {k: v for k, v in self.limb_assignment.right_limbs.items() if not k.startswith('right_body_part_')}
        self.focal_lengths = focal_lengths or [1,12, 18, 32, 34, 36, 38, 40, 100, 120, 180, 320, 340, 360, 380, 400, 
                                                500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 
                                                1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 
                                                2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3800, 4000]
        self.height = height
        self.width = width
        self.n_iter = n_iter
        self.threshold = threshold
        
    def _transpose_if_needed(self, points):
        """
        Ensure points are in the correct shape (2, N). If they are in shape (N, 2), transpose them.
        """
        points = np.array(points)
        if points.shape[0] != 2 and points.shape[1] == 2:
            points = points.T
        return points

    
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


    def find_fundamental_matrix_CV(self,points1,points2):
        pts1 = np.array(points1)
        pts2 = np.array(points2)
    
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)
        F_norm, mask = cv2.findFundamentalMat(pts1_norm,pts2_norm,cv2.FM_8POINT)
     
        if F_norm is None or F_norm.shape != (3, 3):
            print(f"Invalid fundamental matrix shape: {F_norm.shape if F_norm is not None else 'None'}")
            return None
        print(F_norm.shape)
        F_cv_norm = T2.T @ F_norm @ T1
        
        return F_cv_norm
    
    
    def find_fundamental_matrix(self,points1, points2):
        """
        Compute the fundamental matrix from corresponding points in two images.
        
        Parameters:
        points1 (ndarray): Nx2 array of points from the first image.
        points2 (ndarray): Nx2 array of points from the second image.
        
        Returns:
        F (ndarray): 3x3 fundamental matrix.
        """
        if points1.shape[0] != points2.shape[0]:
            raise ValueError("Input point arrays must have the same number of points.")
        
        # Normalize the points
        points1_norm, T1 = self.normalize_points(points1)
        points2_norm, T2 = self.normalize_points(points2)
        
        p
        
        points1
        # Initialize the design matrix A
        A_all = np.empty((0, 9))
        
        # Construct the matrix A
        for pts1, pts2 in zip(points1_norm, points2_norm):
            x1, y1 = pts1
            x2, y2 = pts2
            A = np.array([x1 * x2, x2 * y1, x2, y2 * x1, y1 * y2, y2, x1, y1, 1])
            A_all = np.vstack((A_all, A))
        
        # Perform SVD
        u, s, vh = np.linalg.svd(A_all)
        
        # Extract the fundamental matrix
        min_singular_value_index = np.argmin(s)
        F_norm = vh[min_singular_value_index].reshape(3, 3)

        # Enforce singularity
        U, S, V = np.linalg.svd(F_norm)
    
        S[-1] = 0
        F_norm = U @ np.diag(S) @ V
        
        # Denormalize
        F = T2.T @ F_norm @ T1
        
        return F
    
    def reflective_fundamental_matrix(self,points1, points2):
        """
        Solves for the reflective fundamental matrix using a modified
        eight-point algorithm based on skew-symmetric matrix construction.
        """
        # Step 1: Normalize points
        points1_norm, T1 = self.normalize_points(np.array(points1))
        points2_norm, T2 = self.normalize_points(np.array(points2))
        
        # Step 2: Construct the matrix A
        A_all = np.empty((0, 6))
        
        for p1, p2 in zip(points1_norm, points2_norm):
            u, v = p1
            u_prime, v_prime = p2
            A = np.array([-v_prime * u + u_prime * v, u_prime, v_prime, u, v, 1])
            A_all = np.vstack((A_all, A))
        
        # Step 3: Solve the least-squares problem
        U, S, Vh = np.linalg.svd(np.transpose(A_all)@A_all)
        
        # The solution is the eigenvector corresponding to the smallest singular value
        f_prime = Vh[-1, :]
        
        # Step 4: Construct the skew-symmetric fundamental matrix F'
        F_prime = np.array([
            [0, f_prime[0], f_prime[1]],
            [-f_prime[0], 0, f_prime[2]],
            [f_prime[3], f_prime[4], f_prime[5]]
        ])
        
        # Step 5: Denormalize F' to obtain the final fundamental matrix
        F = T2.T @ F_prime @ T1
        
        return F
        
    def calculate_intersection_of_lines(self,line1, line2):
        # Line format: ax + by + c = 0
        # line1 = [a1, b1, c1]
        # line2 = [a2, b2, c2]
        
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        denominator = a1 * b2 - a2 * b1
        if denominator == 0:
            return None  # Lines are parallel or coincident
        
        x = (b1 * c2 - b2 * c1) / denominator
        y = (a2 * c1 - a1 * c2) / denominator
    
        return [x, y]
    
    def find_epipole_from_F(self,lines1, lines2):
        length = lines1.shape[0]
        epipole = []
        lines = []

        for _ in range(length):
            idx = np.random.choice(length, 2, replace=False)
            intersection = self.calculate_intersection_of_lines(lines1[idx[0]], lines1[idx[1]])
            if intersection:  # Ensure intersection is not None
                epipole.append(intersection)
                lines.append([lines1[idx[0]], lines1[idx[1]]])

        if epipole:  # Ensure epipole list is not empty
            return np.mean(epipole, axis=0), lines
        else:
            return None, []
        

    def plot_pose_epipole(self,points2D, points2D_2, epipole, epipolar_lines_1, epipolar_lines_2, labels=None, mode="lines"):
            fig, ax = plt.subplots()
            pairs = self.connections
       
            if mode == 'lines':
                for pair in pairs:
                    color = 'red' if 'left' in pair[0] else 'blue'
                    color_2 = 'pink' if 'left' in pair[0] else 'lightblue'
                    ax.plot([points2D[self.labels.index(pair[0])][0], points2D[self.labels.index(pair[1])][0]], 
                                [points2D[self.labels.index(pair[0])][1], points2D[self.labels.index(pair[1])][1]], color=color)
                    ax.plot([points2D_2[self.labels.index(pair[0])][0], points2D_2[self.labels.index(pair[1])][0]], 
                                [points2D_2[self.labels.index(pair[0])][1], points2D_2[self.labels.index(pair[1])][1]], color=color_2)

                # Draw epipolar lines
                all_points = np.concatenate((points2D, points2D_2), axis=0)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            
            # Draw epipolar lines within the plot limits
            for r in epipolar_lines_1:
                x0, y0 = x_min, -(r[2] + r[0] * x_min) / r[1]
                x1, y1 = x_max, -(r[2] + r[0] * x_max) / r[1]
                ax.plot([x0, x1], [y0, y1], 'g--')

            for r in epipolar_lines_2:
                x0, y0 = x_min, -(r[2] + r[0] * x_min) / r[1]
                x1, y1 = x_max, -(r[2] + r[0] * x_max) / r[1]
                ax.plot([x0, x1], [y0, y1], 'b--')

            ax.plot(epipole[0], epipole[1], 'ro')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()
            
    def triangulate_points_fundamental(self,points1,points2, P, P1):
        points1 = points1.astype(float).T
        points2 = points2.astype(float).T
        points=cv2.triangulatePoints(P,P1,points1,points2)
        points3D = points/points[3]
        return points3D
        
  

            
    def get_essential_matrix(self,fundamental_matrix, intrinsic_matrix):
            essential_matrix = intrinsic_matrix.T@ fundamental_matrix @ intrinsic_matrix
            return essential_matrix

    # def get_rotation_and_translation_from_essential(self,essential_matrix):
    #     R_F,R2_F,t_F=cv2.decomposeEssentialMat(essential_matrix)
    #     return R_F,R2_F,t_F
    
    
    
    def decompose_reflective_essential_matrix(self,E):
        # Perform SVD on the essential matrix E
        U, S, Vt = np.linalg.svd(E)
        
        # Ensure the singular values are correct for an essential matrix
        sigma = (S[0] + S[1]) / 2.0
        S_corrected = np.diag([sigma, sigma, 0])
        
        # Define the matrix W
        W = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
        
        # Calculate the possible rotation matrices
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        
        # Ensure the determinant is -1 (reflective rotation matrix)
        if np.linalg.det(R1) > 0:
            R1 = -R1
        if np.linalg.det(R2) > 0:
            R2 = -R2
        
        # Calculate the reflected translation vector t'
        t_prime = U[:, 2]  # Third column of U
        
        # Reflection matrix D
        D = np.array([[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
        
        # Recover the original translation t
        t1 = t_prime
        t2 = -t1
        
        return R1, R2, t1, t2
            
    
    def get_virtual_camera_matrix(self,R,t_F,K):
            t_F = np.array(t_F)
            if t_F is None:
                return None
        
            T=2*self.d*t_F
            
            return np.dot(K, np.hstack((R, T.reshape(-1, 1))))
        
        
    # def choose_correct_R(self, R1, R2, t, points1, points2, K):
    #     pts1 = np.array(points1)
    #     pts2 = np.array(points2)
        
    #     if R1 is None or R2 is None:
    #         raise ValueError("Invalid rotation matrices.")
        
    #     # Define camera matrix for real camera
    #     P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Real camera is at origin
        
    #     # Create translation options (t and -t)
    #     T_options = [t, -t]
       
        
    #     # Generate all possible virtual camera matrices with the reflection
    #     P2_options = [self.get_virtual_camera_matrix(R, t_, K) for R in (R1, R2) for t_ in T_options]
        
    #     # Store corresponding R, t combinations
    #     R_t_options = [(R, t_) for R in (R1, R2) for t_ in T_options]
        
    #     def check_points(pts_3d, P2):
    #         # Reproject the 3D points into the second camera (virtual camera)
    #         pts_3d_cam2 = P2 @ pts_3d
            
    #         # Check if the points have positive depth in both cameras
    #         valid_real = np.all(np.round(pts_3d[2, :], 4) >= 0)
    #         valid_virtual = np.all(np.round(pts_3d_cam2[2, :], 4) >= 0)
           
    #         return valid_real and valid_virtual
        
    #     valid_poses = []
        
    #     # Iterate over all P2 and (R, t) combinations
    #     for P2, R_T in zip(P2_options, R_t_options):
    #         # Triangulate 3D points using P1 (real camera) and P2 (virtual camera)
    #         pts_3d = self.triangulate_points(pts1, pts2, K @ P1, P2)
            
    #         R, t_ = R_T
            
    #         # Check if points are in front of both cameras
    #         if check_points(pts_3d, P2):
    #             valid_poses.append((R, t_))
        
    #     # Select the first valid pose (if any)
    #     if valid_poses:
    #         R_correct, t_correct = valid_poses[0]
    #         return R_correct, t_correct
    #     else:
    #         return None, None
    
    
    def choose_correct_R(self, R1, R2, t, points1, points2, K):
        pts1 = np.array(points1)
        pts2 = np.array(points2)
        
        if R1 is None or R2 is None:
            raise ValueError("Invalid rotation matrices.")
        
        # Define camera matrix for real camera
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Real camera is at origin
        
        # Create translation options (t and -t)
        T_options = [t, -t]
        
        # Generate all possible virtual camera matrices with the reflection
        P2_options = [self.get_virtual_camera_matrix(R, t_, K) for R in (R1, R2) for t_ in T_options]
        
        # Store corresponding R, t combinations
        R_t_options = [(R, t_) for R in (R1, R2) for t_ in T_options]
        
        def check_points(pts_3d, P2):
            # Reproject the 3D points into the second camera (virtual camera)
            pts_3d_cam2 = P2 @ pts_3d
            
            # Check if the points have positive depth in both cameras
            valid_real = pts_3d[2, :] >= 0
            valid_virtual = pts_3d_cam2[2, :] >= 0
        
            return valid_real, valid_virtual
        
        best_R, best_t = None, None
        max_valid_count = 0
        
        # Iterate over all P2 and (R, t) combinations
        for P2, (R, t_) in zip(P2_options, R_t_options):
            # Triangulate 3D points using P1 (real camera) and P2 (virtual camera)
            pts_3d = self.triangulate_points_fundamental(pts1, pts2, K @ P1, P2)
            
            # Check if points are in front of both cameras
            valid_real, valid_virtual = check_points(pts_3d, P2)
            
            # Count how many points have positive depth in both cameras
            valid_count = np.sum(valid_real & valid_virtual)
            
            # Update the best R and t if this configuration has more valid points
            if valid_count > max_valid_count:
                max_valid_count = valid_count
                best_R, best_t = R, t_
        
        # Return the best R, t pair
        return best_R, best_t if max_valid_count > 0 else (None, None)        
            
    
    def optimal_focal_length(self, points1, points2,max_body_height,mim_body_height):
        """
        Determine the optimal focal length by iterating over a range of focal lengths,
        computing the transformation and triangulation, and selecting the focal length
        with the best score. Penalizes body height exceeding 2 meters.
        """
        max_score = 0
        best_focal_length = None
        pts1 = np.array(points1)
        pts2 = np.array(points2)
        
        # Estimate the fundamental matrix using RANSAC
        F, _ = self.ransac_fundamental_matrix_epipolar(pts1, pts2)
        print(F)
        # Iterate over all focal lengths
        for f in self.focal_lengths:
            print(f)
            
            # Create the intrinsic matrix for the current focal length
            K1 = np.array([[f, 0, self.width / 2],
                        [0, f, self.height / 2],
                        [0, 0, 1]])
            
            # Compute the essential matrix from the fundamental matrix
            E = self.get_essential_matrix(F, K1)
            
            # Decompose the essential matrix to get possible rotations and translations
            R1, R2, t_F, _ = self.decompose_reflective_essential_matrix(E)
            
            # Choose the correct R and t based on the triangulation validity
            R, T = self.choose_correct_R(R1, R2, t_F, pts1, pts2, K1)
            if R is None:
                continue
            
            # Compute the projection matrices for the real and virtual cameras
            P2 = self.get_virtual_camera_matrix(R, T, K1)
            P1 = np.hstack((np.identity(3), np.zeros((3, 1))))
            P1 = K1 @ P1
            
            # Triangulate the 3D points
            points3D = self.triangulate_points_fundamental(pts1, pts2, P1, P2)
            
            left_distances = []
            right_distances = []
            
            # Calculate distances for left limbs
            for limb, (start_label, end_label) in self.left_limbs.items():
                start_point = points3D[:3, self.labels.index(start_label)]
                end_point = points3D[:3, self.labels.index(end_label)]
                distance = np.linalg.norm(end_point - start_point)
                left_distances.append(distance)
            
            # Calculate distances for right limbs
            for limb, (start_label, end_label) in self.right_limbs.items():
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
            
            
            body_height = self.get_body_height(points3D)
            
            # print(f"Focal length: {f}, Score: {score}, Body height: {body_height}")
       
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
                
            print(f"Best focal length: {best_focal_length}, Score: {max_score}")
        
        if best_focal_length is None:
            best_focal_length = 1
        return best_focal_length,max_score
    
    
    def create_calibration_matrix(self, points1, points2,max_height=10,min_height=0.1):
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
        
        focal_length,_=self.optimal_focal_length(points1, points2,max_height,min_height)
      
        K = np.array([
            [focal_length, 0, self.width / 2],
            [0, focal_length, self.height / 2],
            [0, 0, 1]
        ])
        
        return K,focal_length
    

    
    def get_optimal_Fundamental_Matrix(self,points1,points2):
        
        
        pts1 = np.array(points1)
        pts2 = np.array(points2)
        F,_=cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
        return F
    
    def ransac_fundamental_matrix(self, points1, points2, n_iter=None,t=None):
        """
        Perform RANSAC to estimate the best fundamental matrix using the Sampson error.
        """
        
        n_iter = n_iter if n_iter is not None else self.n_iter
        t = t if t is not None else self.threshold
        num_points = points1.shape[0]
        max_inliers = 0
        best_F = None
        best_inliers = []
        print(n_iter)
        print(t)
        # Helper function to compute Sampson error
        def sampson_error(F, points1, points2):
            """
            Compute the Sampson error for each point correspondence.
            F: Fundamental matrix (3x3)
            points1: Points from the first image (Nx2)
            points2: Corresponding points from the second image (Nx2)
            Returns: Residuals (Nx1) representing the Sampson error for each correspondence.
            """
            # Convert points to homogeneous coordinates
            ones = np.ones((points1.shape[0], 1))
            points1_hom = np.hstack([points1, ones])  # Nx3
            points2_hom = np.hstack([points2, ones])  # Nx3

            # Epipolar lines
            lines1 = points2_hom @ F.T   # l1 = F' * p2
            lines2 = points1_hom @ F     # l2 = F * p1

            # Compute Sampson error for each correspondence
            num = np.sum(points2_hom * (F @ points1_hom.T).T, axis=1) ** 2
            denom = (lines1[:, 0] ** 2 + lines1[:, 1] ** 2 + lines2[:, 0] ** 2 + lines2[:, 1] ** 2)
            epsilon = 1e-8
            sampson_err = num / (denom + epsilon)

            return sampson_err

        for _ in range(n_iter):
            # Randomly select 8 point correspondences for the 8-point algorithm
            idx = np.random.choice(num_points, 8, replace=False)
            
            sample_points1 = points1[idx]
      
            sample_points2 = points2[idx]
             

            # Estimate the fundamental matrix F using the selected points
        
            F = self.find_fundamental_matrix(sample_points1, sample_points2)

            # Calculate the Sampson error for all correspondences
            residuals = sampson_error(F, points1, points2)

            # Count inliers
            inliers = np.where(residuals < t)[0]
            num_inliers = len(inliers)

            # Update the best fundamental matrix if the current one has more inliers
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_F = F
                best_inliers = inliers

        # Recompute the fundamental matrix using all inliers
        if best_inliers.size > 0:
            best_F = self.find_fundamental_matrix(points1[best_inliers], points2[best_inliers])
        print(best_inliers.size)

        return best_F, best_inliers

    
    def get_3D_estimation(self, points1, points2,max_height=10,min_height=0.1,n_iter=None,t=None):
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
        
        points1,points2=np.array(points1),np.array(points2)
        K,focal_length= self.create_calibration_matrix(points1, points2,max_height,min_height)
        print(K)
        
        # Apply RANSAC to estimate the fundamental matrix
        F, _ = self.ransac_fundamental_matrix_epipolar(points1,points2,t, n_iter)
        if F is None:
            raise ValueError("RANSAC failed to find a valid fundamental matrixr.")
        
        essential_matrix = self.get_essential_matrix(F, K)
        R1,R2,t,_=self.decompose_reflective_essential_matrix(essential_matrix)
        R,t=self.choose_correct_R(R1,R2,t,points1,points2,K)
        if R is None:
            raise ValueError("RANSAC failed to find a valid rotation matrix and translation vector.")
        
        # Get the camera projection matrices
        P2 = self.get_virtual_camera_matrix(R, t, K)
        P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(-1, 1)))
        P1 = K @ P1
        
        # Triangulate points to get 3D coordinates
        pts1 = np.array(points1)
        pts2 = np.array(points2)
        # pts1[:,1]=self.height-pts1[:,1]
        # pts2[:,1]=self.height-pts2[:,1]
        points3D = self.triangulate_points_fundamental(points1, points2, P1, P2)
        
        return points3D,focal_length
   
    
    
    def triangulate_video(self, points1, points2,max_height=10,min_height=0.1):
        triangulated_points = []
        
        points1 = np.array(points1)
        points2 = np.array(points2)
        points_real_world_all = points1
        points_mirror_all = points2
        focal_lengths_first_15 = []

        # Select the first 5 frames
        for i in range(5):
            f, _ = self.optimal_focal_length(points_real_world_all[i], points_mirror_all[i], max_height, min_height)
            focal_lengths_first_15.append(f)

        # Select 10 more frames evenly spaced throughout the length
        total_frames = len(points_real_world_all)
        spacing = total_frames // 10

        for i in range(5, total_frames, spacing):
            f, _ = self.optimal_focal_length(points_real_world_all[i], points_mirror_all[i], max_height, min_height)
            focal_lengths_first_15.append(f)

        # Calculate the mean focal length
        mean_focal_length = np.mean(focal_lengths_first_15)

        # Compute the inverse of the absolute differences from the mean
        differences = np.abs(np.array(focal_lengths_first_15) - mean_focal_length)

        # Add a small epsilon to avoid division by zero (in case any focal length is exactly the mean)
        epsilon = 1e-6
        weights = 1 / (differences + epsilon)

        # Normalize the weights to sum to 1
        weights /= np.sum(weights)

        # Compute the weighted average of the focal lengths
        weighted_f = np.sum(np.array(focal_lengths_first_15) * weights)
        
        f=weighted_f
        
        K = np.array([[f, 0, self.width / 2],
                      [0, f, self.height / 2],
                      [0, 0, 1]])
        
     
        P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(-1, 1)))
        P1 = K @ P1
      #Apply RANSAC to estimate the fundamental matrix
        prev_F = None
        prev_R = None
        prev_t = None

        for points_realworld, points_mirror in zip(points1, points2):
            F, _ = self.ransac_fundamental_matrix_epipolar(points_realworld, points_mirror)
            
            if F is None:
                if prev_F is None:
                    raise ValueError("RANSAC failed to find a valid fundamental matrix.")
                F = prev_F
            else:
                prev_F = F
            
            essential_matrix = self.get_essential_matrix(F, K)
            R1, R2, t1,_ = self.decompose_reflective_essential_matrix(essential_matrix)
            R, t = self.choose_correct_R(R1, R2, t1, points_realworld, points_mirror, K)
            
            if R is None or t is None:
                if prev_R is None or prev_t is None:
                    raise ValueError("RANSAC failed to find a valid rotation matrix and translation vector.")
                R = prev_R
                t = prev_t
            else:
                prev_R = R
                prev_t = t
            
            # Get the camera projection matrices
            P2 = self.get_virtual_camera_matrix(R, t, K)
            
            pts1 = np.array(points_realworld)
            pts2 = np.array(points_mirror)
            points3D = self.triangulate_points_fundamental(pts1, pts2, P1, P2)
            triangulated_points.append(points3D)

        return np.array(triangulated_points),f
    
    def triangulate_video_test(self, points1, points2,max_height=10,min_height=0.1):
        poses=[]
        for i in range(len(points1)):
            print(i)
            points1[i]=np.array(points1[i])
            points2[i]=np.array(points2[i])
            pose,_=self.get_3D_estimation(points1[i],points2[i],max_height,min_height)
            poses.append(pose)
        return poses
    
    def triangulate_video_solo(self, points1, points2, max_height=10, min_height=0.1):
        triangulated_points = []
        
        
        points1 = np.array(points1)
       
        points2 = np.array(points2)
     
        points_real_world_all = points1
        points_mirror_all = points2
        
        focal_lengths_first_15 = []

        # Select the first 5 frames
        for i in range(5):
            f, _ = self.optimal_focal_length(points_real_world_all[i], points_mirror_all[i], max_height, min_height)
            focal_lengths_first_15.append(f)

        # Select 10 more frames evenly spaced throughout the length
        total_frames = len(points_real_world_all)
        spacing = total_frames // 10

        for i in range(5, total_frames, spacing):
            f, _ = self.optimal_focal_length(points_real_world_all[i], points_mirror_all[i], max_height, min_height)
            focal_lengths_first_15.append(f)

        # Calculate the mean focal length
        mean_focal_length = np.mean(focal_lengths_first_15)

        # Compute the inverse of the absolute differences from the mean
        differences = np.abs(np.array(focal_lengths_first_15) - mean_focal_length)

        # Add a small epsilon to avoid division by zero (in case any focal length is exactly the mean)
        epsilon = 1e-6
        weights = 1 / (differences + epsilon)

        # Normalize the weights to sum to 1
        weights /= np.sum(weights)

        # Compute the weighted average of the focal lengths
        weighted_f = np.sum(np.array(focal_lengths_first_15) * weights)
        
        f = weighted_f
        
        # Intrinsic matrix with the computed focal length
        K = np.array([[f, 0, self.width / 2],
                    [0, f, self.height / 2],
                    [0, 0, 1]])

        P1 = np.hstack((np.identity(3), np.array([0, 0, 0]).reshape(-1, 1)))
        P1 = K @ P1

        # Calculate the fundamental matrix for the first frame
        first_points_realworld = points_real_world_all[0]
        first_points_mirror = points_mirror_all[0]
        F, _ = self.ransac_fundamental_matrix_epipolar(first_points_realworld, first_points_mirror)

        if F is None:
            raise ValueError("RANSAC failed to find a valid fundamental matrix for the first frame.")
        
        # Get the essential matrix for the first frame
        essential_matrix = self.get_essential_matrix(F, K)
        
        # Decompose the essential matrix to get R and t
        R1, R2, t1, _ = self.decompose_reflective_essential_matrix(essential_matrix)
        
        # Choose the correct rotation and translation
        R, t = self.choose_correct_R(R1, R2, t1, first_points_realworld, first_points_mirror, K)
        
        if R is None or t is None:
            raise ValueError("Failed to find a valid rotation matrix and translation vector for the first frame.")
        
        # Get the camera projection matrix for the virtual camera
        P2 = self.get_virtual_camera_matrix(R, t, K)
        
        prev_R = R
        prev_t = t

        # Now triangulate points for all frames using the first frame's F, R, and t
        for points_realworld, points_mirror in zip(points1, points2):
            # Triangulate points using the same P2 (from the first frame)
            pts1 = np.array(points_realworld)
            pts2 = np.array(points_mirror)
            # pts1[:,1]=self.height-pts1[:,1]
            # pts2[:,1]=self.height-pts2[:,1]
            points3D = self.triangulate_points_fundamental(pts1, pts2, P1, P2)
            triangulated_points.append(points3D)

        return np.array(triangulated_points), f        

    # def normalize_points(self,points):
    #     """
    #     Normalize points by translating them so that their centroid is at the origin
    #     and scaling them so that the average distance from the origin is sqrt(2).
    #     Returns the normalized points and the normalization matrix.
    #     """
    
    #     points = self._transpose_if_needed(points)

    #     # Compute centroid
    #     centroid = np.mean(points, axis=0)
        
    #     # Shift points to the centroid
    #     shifted_points = points - centroid
        
    #     # Compute the average distance of the points from the origin
    #     mean_distance = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))
        
    #     # Compute the scaling factor so that the average distance is sqrt(2)
    #     scale = np.sqrt(2) / mean_distance
        
    #     # Construct the normalization matrix
    #     T = np.array([[scale, 0, -scale * centroid[0]],
    #                 [0, scale, -scale * centroid[1]],
    #                 [0, 0, 1]])
        
    #     # Apply normalization to the points
    #     ones_column = np.ones((points.shape[0], 1))
        
    #     points_homogeneous = np.hstack((points, ones_column))
    #     normalized_points = (T @ points_homogeneous.T).T
        
    #     return normalized_points[:, :2], T
    
    def ransac_fundamental_matrix_epipolar(self, points1, points2, threshold=0.01, iterations=150):
        """
        Applies RANSAC to estimate the reflective fundamental matrix using epipolar distance.
        
        Parameters:
        - points1: List of points from image 1
        - points2: List of corresponding points from image 2
        - threshold: Epipolar distance threshold to determine inliers
        - iterations: Number of RANSAC iterations
        
        Returns:
        - The best fundamental matrix estimated using RANSAC
        - The inlier mask (boolean array indicating inliers)
        """
        best_F = None
        best_inliers = []
        max_inliers = 0

        num_points = len(points1)
        iterations=self.n_iter
        threshold=self.threshold
        for i in range(iterations):
            # Randomly select 8 pairs of points
            indices = np.random.choice(num_points, 8, replace=False)
            subset1 = np.array([points1[i] for i in indices])
            subset2 = np.array([points2[i] for i in indices])

            # Compute the fundamental matrix using the selected points
           
            F=self.find_fundamental_matrix_CV(subset1,subset2)
            if F is None:
                continue
            # Compute inliers based on epipolar distance
            inliers = []
            for p1, p2 in zip(points1, points2):
                # Convert points to homogeneous coordinates
                p1_h = np.append(p1, 1)
                p2_h = np.append(p2, 1)

                # Compute epipolar line l' = F * p1
                epipolar_line = np.dot(F, p1_h)

                # Compute the epipolar distance
                numerator = np.abs(np.dot(p2_h, epipolar_line))
                denominator = np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2)
                epipolar_distance = numerator / denominator

                # Count as inlier if the epipolar distance is below the threshold
                if epipolar_distance < threshold:
                    inliers.append(True)
                else:
                    inliers.append(False)

            # Check if this model has more inliers than previous models
            num_inliers = np.sum(inliers)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_inliers = inliers
                best_F = F

        # Recompute the fundamental matrix using all inliers
        inlier_points1 = np.array([p for i, p in enumerate(points1) if best_inliers[i]])
        inlier_points2 = np.array([p for i, p in enumerate(points2) if best_inliers[i]])

        best_F = self.find_fundamental_matrix_CV(inlier_points1, inlier_points2)

        return best_F, best_inliers
    
    
 

    def ransac_fundamental_matrix_sampson(self, points1, points2, n_iter=None, t=None):
        """
        Perform RANSAC to estimate the best fundamental matrix using the Sampson error.
        After RANSAC, refine the result using Levenberg-Marquardt optimization.
        """
        n_iter = n_iter if n_iter is not None else self.n_iter
        t = t if t is not None else self.threshold
        num_points = points1.shape[0]
        max_inliers = 0
        best_F = None
        best_inliers = []
        
        # Helper function to compute Sampson error
        def sampson_error(F, points1, points2):
            """
            Compute the Sampson error for each point correspondence.
            F: Fundamental matrix (3x3)
            points1: Points from the first image (Nx2)
            points2: Corresponding points from the second image (Nx2)
            Returns: Residuals (Nx1) representing the Sampson error for each correspondence.
            """
            # Convert points to homogeneous coordinates
            ones = np.ones((points1.shape[0], 1))
            points1_hom = np.hstack([points1, ones])  # Nx3
            points2_hom = np.hstack([points2, ones])  # Nx3

            # Epipolar lines
            lines1 = points2_hom @ F.T   # l1 = F' * p2
            lines2 = points1_hom @ F     # l2 = F * p1

            # Compute Sampson error for each correspondence
            num = np.sum(points2_hom * (F @ points1_hom.T).T, axis=1) ** 2
            denom = (lines1[:, 0] ** 2 + lines1[:, 1] ** 2 + lines2[:, 0] ** 2 + lines2[:, 1] ** 2)
            epsilon = 1e-8
            sampson_err = num / (denom + epsilon)

            return sampson_err

        for _ in range(n_iter):
            # Randomly select 8 point correspondences for the 8-point algorithm
            idx = np.random.choice(num_points, 8, replace=False)
            
            sample_points1 = points1[idx]
            sample_points2 = points2[idx]

            # Estimate the fundamental matrix F using the selected points
            F = self.find_fundamental_matrix_CV(sample_points1, sample_points2)

            # Calculate the Sampson error for all correspondences
            residuals = sampson_error(F, points1, points2)

            # Count inliers
            inliers = np.where(residuals < t)[0]
            num_inliers = len(inliers)

            # Update the best fundamental matrix if the current one has more inliers
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_F = F
                best_inliers = inliers

        # Recompute the fundamental matrix using all inliers
        if best_inliers.size > 0:
            best_F = self.find_fundamental_matrix_CV(points1[best_inliers], points2[best_inliers])

            if best_inliers.size >= 9:  # Ensure there are enough inliers for LM optimization
                # Refine with Levenberg-Marquardt using all inliers
                def residual_function(F_vec):
                    F = F_vec.reshape(3, 3)
                    return sampson_error(F, points1[best_inliers], points2[best_inliers])

                F_initial = best_F.flatten()
                
                # Run Levenberg-Marquardt optimization
                result = opt.least_squares(residual_function, F_initial, method='lm')
                best_F = result.x.reshape(3, 3)

        return best_F, best_inliers
    
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
        
    #     # Estimate the fundamental matrix using RANSAC
    #     F, _ = self.ransac_fundamental_matrix(pts1, pts2)
        
    #     # Iterate over all focal lengths
    #     for f in self.focal_lengths:
    #         print(f)
            
    #         # Create the intrinsic matrix for the current focal length
    #         K1 = np.array([[f, 0, self.width / 2],
    #                     [0, f, self.height / 2],
    #                     [0, 0, 1]])
            
    #         # Compute the essential matrix from the fundamental matrix
    #         E = self.get_essential_matrix(F, K1)
            
    #         # Decompose the essential matrix to get possible rotations and translations
    #         R1, R2, t_F, _ = self.decompose_reflective_essential_matrix(E)
            
    #         # Choose the correct R and t based on the triangulation validity
    #         R, T = self.choose_correct_R(R1, R2, t_F, pts1, pts2, K1)
    #         print(R,T)
    #         if R is None:
    #             continue
            
    #         # Compute the projection matrices for the real and virtual cameras
    #         P2 = self.get_virtual_camera_matrix(R, T, K1)
    #         P1 = np.hstack((np.identity(3), np.zeros((3, 1))))
    #         P1 = K1 @ P1
            
    #         # Triangulate the 3D points
    #         points3D = self.triangulate_points_fundamental(pts1, pts2, P1, P2)
            
    #         left_distances = []
    #         right_distances = []
            
    #         # Calculate distances for left limbs
    #         for limb, (start_label, end_label) in self.left_limbs.items():
    #             start_point = points3D[:3, self.labels.index(start_label)]
    #             end_point = points3D[:3, self.labels.index(end_label)]
    #             distance = np.linalg.norm(end_point - start_point)
    #             left_distances.append(distance)
            
    #         # Calculate distances for right limbs
    #         for limb, (start_label, end_label) in self.right_limbs.items():
    #             start_point = points3D[:3, self.labels.index(start_label)]
    #             end_point = points3D[:3, self.labels.index(end_label)]
    #             distance = np.linalg.norm(end_point - start_point)
    #             right_distances.append(distance)
            
    #         # Calculate the score by comparing left and right limb lengths
    #         limb_scores = []
    #         for left_limb, right_limb in zip(left_distances, right_distances):
    #             limb_score = min(left_limb, right_limb) / max(right_limb, left_limb)
    #             limb_scores.append(limb_score)
            
    #         # Calculate the mean score for this focal length
    #         score = np.mean(limb_scores)
    #         print(f"Score for focal length {f}: {score}\n")
            
    #         # Update the best focal length if this score is higher
    #         if score > max_score:
    #             max_score = score
    #             best_focal_length = f
        
    #     return best_focal_length
    