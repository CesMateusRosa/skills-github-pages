import numpy as np
from scipy.signal import savgol_filter,butter, filtfilt, savgol_filter, gaussian
from scipy.signal.windows import gaussian

class PoseFiltering:
    def __init__(self, window_size,jump_threshold=10,poly_order=3,poly_order_sg=2):
        self.window_size = window_size
        self.poly_order = poly_order
        self.poly_order_sg = poly_order_sg
        self.jump_threshold = jump_threshold
       
    
    def euclidean_distance(self, point1, point2):
        """ Calculate the Euclidean distance between two 2D points """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def weighted_convolve(self, keypoints):
        """ Apply weighted moving average filter with lower weights for erratic points """
        filtered_keypoints = np.zeros_like(keypoints)
        
        for i in range(len(keypoints)):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(len(keypoints), i + self.window_size // 2 + 1)
            window_points = keypoints[window_start:window_end]
            
            # Initialize weights for the window (default 1)
            weights = np.ones(len(window_points))
            
            # Calculate distances from the current point to the rest of the points in the window
            for j in range(1, len(window_points)):
                distance = self.euclidean_distance(window_points[j], window_points[j - 1])
                if distance > self.jump_threshold:
                    # Reduce weight if a large jump is detected
                    weights[j] *= 0.2  # Lower weight for erratic points
            
            # Normalize weights
            weights /= np.sum(weights)
            
            # Apply weighted average
            filtered_keypoints[i] = np.average(window_points, axis=0, weights=weights)
        
        return filtered_keypoints

    def weighted_moving_average(self, keypoints):
        """ Smooth keypoints using weighted convolution """
        keypoints = np.array(keypoints)
        smoothed_keypoints = self.weighted_convolve(keypoints)
        return smoothed_keypoints.tolist()
    
    def moving_average(self, arr):
        """ Apply a moving average filter along the specified axis """
        window = np.ones(self.window_size) / self.window_size
        filtered_data = np.apply_along_axis(
            lambda m: np.convolve(m, window, mode='valid'),
            axis=0, arr=arr
        )
        return filtered_data.tolist()

    def savgol_filter_along_axis(self, arr):
        """ Apply the Savitzky–Golay filter along the specified axis """
        if self.window_size % 2 == 0:
            self.window_size += 1
        if self.window_size <= self.poly_order_sg:
            raise ValueError("window_size must be greater than poly_order")
        
        filtered_data = np.apply_along_axis(
            lambda m: savgol_filter(m, self.window_size, self.poly_order, mode='interp'),
            axis=0, arr=arr
        )
        return filtered_data.tolist()

    def gaussian_window(self,std_dev=0.1):
        """ Create a Gaussian window """
        window = gaussian(self.window_size, std_dev)
        window /= window.sum()  # Normalize to ensure the sum is 1
        return window

    def weighted_gaussian_convolve(self, keypoints):
        """ Apply Gaussian filter with adjusted weights for erratic movements """
        filtered_keypoints = np.zeros_like(keypoints)
        gaussian_weights = self.gaussian_window()

        for i in range(len(keypoints)):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(len(keypoints), i + self.window_size // 2 + 1)
            window_points = keypoints[window_start:window_end]
            
            # Adjust Gaussian weights based on erratic movement detection
            adjusted_weights = np.copy(gaussian_weights[:len(window_points)])
            
            for j in range(1, len(window_points)):
                distance = self.euclidean_distance(window_points[j], window_points[j - 1])
                if distance > self.jump_threshold:
                    # Reduce weight if a large jump is detected
                    adjusted_weights[j] *= 0.2  # Decrease weight for erratic points
            
            # Normalize adjusted weights
            adjusted_weights /= np.sum(adjusted_weights)
            
            # Apply weighted average using the adjusted Gaussian weights
            filtered_keypoints[i] = np.average(window_points, axis=0, weights=adjusted_weights)
        
        return filtered_keypoints

    def gauassian_filter(self, arr):
        """ Apply Gaussian filter along the specified axis """
        keypoints = np.array(arr)
        smoothed_keypoints = self.weighted_gaussian_convolve(keypoints)
        return smoothed_keypoints.tolist()
    
    def butterworth_filter(self, data, cutoff_freq=2, filter_order=2, sampling_rate=30):
        """ Apply a low-pass Butterworth filter to the keypoint data """
        # Calculate the Nyquist frequency
        nyquist_freq = 0.5 * sampling_rate
        
        # Normalize the cutoff frequency to be in the range [0, 1] (Wn)
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # Design Butterworth filter
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        
        # Apply filter to each keypoint across frames
        filtered_data = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=0, arr=data)
        
        return filtered_data.tolist()