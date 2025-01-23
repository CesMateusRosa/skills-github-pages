import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import importlib
import Base_Pose_Estimator
importlib.reload(Base_Pose_Estimator)
from Base_Pose_Estimator import BasePoseEstimator, LimbAssignment


class PoseVisualizer(BasePoseEstimator):
    
    def __init__(self, keypoint_labels, connections):
        super().__init__(keypoint_labels, connections)

    def _get_color_for_side(self, label):
        """
        Determine color based on whether the label is for the left or right side.
        """
        if label and 'left' in label:
            return 'red', 'pink'  # Colors for left side
        elif label and 'right' in label:
            return 'blue', 'lightblue'  # Colors for right side
        else:
            return 'gray', 'gray'  # Default color if no side is identified

    def _transpose_if_needed(self, points):
        """
        Ensure points are in the correct shape (2, N). If they are in shape (N, 2), transpose them.
        """
        points = np.array(points)
        if points.shape[0] != 2 and points.shape[1] == 2:
            points = points.T
        return points

    def _plot_connections(self, ax, points, color):
        """
        Plot connections between keypoints on the given axis.
        """
        for connection in self.connections:
            idx1 = self.labels.index(connection[0])
            idx2 = self.labels.index(connection[1])
            ax.plot([points[0, idx1], points[0, idx2]], [points[1, idx1], points[1, idx2]], color)

    def plot_interactive_3D(self, points3D):
        """
        Plot a 3D interactive plot using Plotly with the given 3D points.
        """
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='cube',
                dragmode='orbit',
                uirevision=True
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        data = []
        for connection in self.connections:
            trace_lines = go.Scatter3d(
                x=points3D[0, [self.labels.index(connection[0]), self.labels.index(connection[1])]], 
                y=points3D[1, [self.labels.index(connection[0]), self.labels.index(connection[1])]], 
                z=2 * points3D[2, [self.labels.index(connection[0]), self.labels.index(connection[1])]], 
                mode='lines',
                line=dict(color='black', width=2),
            )
            data.append(trace_lines)
        
        fig = go.Figure(data=data, layout=layout)
        fig.show()


    def plot_interactive_3D_compare(self, points3D_1, points3D_2=None, connections_2=None, labels_2=None):
        """
        Plot a 3D interactive plot using Plotly with the given 3D points.
        """
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='cube',
                dragmode='orbit',
                uirevision=True
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        data = []
        # Plot the first set of points
        for connection in self.connections:
            trace_lines = go.Scatter3d(
                x=points3D_1[0, [self.labels.index(connection[0]), self.labels.index(connection[1])]], 
                y=points3D_1[1, [self.labels.index(connection[0]), self.labels.index(connection[1])]], 
                z=points3D_1[2, [self.labels.index(connection[0]), self.labels.index(connection[1])]], 
                mode='lines',
                line=dict(color='black', width=2),
            )
            data.append(trace_lines)
        
        # Plot the second set of points if provided
        if points3D_2 is not None and connections_2 is not None and labels_2 is not None:
            for connection in connections_2:
                trace_lines = go.Scatter3d(
                    x=points3D_2[0, [labels_2.index(connection[0]), labels_2.index(connection[1])]], 
                    y=points3D_2[1, [labels_2.index(connection[0]), labels_2.index(connection[1])]], 
                    z=points3D_2[2, [labels_2.index(connection[0]), labels_2.index(connection[1])]], 
                    mode='lines',
                    line=dict(color='red', width=2),
                )
                data.append(trace_lines)
        
        fig = go.Figure(data=data, layout=layout)
        fig.show()
    def plot_2D_projection_with_vector(self, points1, points2, vector):
        """
        Plot two sets of 2D points in a 3D space along with a vector.
        Points1 are projected at z=0 and Points2 at z=100, with a vector starting at the centroid of points2.
        """
        data = []
        centroid = points2[-1]

        vector_trace = go.Cone(
            x=[centroid[0]], y=[centroid[1]], z=[25],
            u=[50 * vector[0]], v=[50 * vector[1]], w=[50 * vector[2]],
            sizemode="scaled", sizeref=2, anchor="tail", showscale=False, name='Vector'
        )
        data.append(vector_trace)

        for point in points1:
            trace = go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[0],
                mode='markers', marker=dict(size=5, color='blue'), name='Points1'
            )
            data.append(trace)

        for point in points2:
            trace = go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[100],
                mode='markers', marker=dict(size=5, color='red'), name='Points2'
            )
            data.append(trace)

        layout = go.Layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def plot_matplotlib_3D_mirrored_plot(self, points3D_1, points3D_2):
        """
        Plot two sets of 3D points using Matplotlib, with one set mirrored.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points3D_1[0, :], points3D_1[1, :], points3D_1[2, :], color='blue')
        ax.scatter(points3D_2[0, :], points3D_2[1, :], points3D_2[2, :], color='red')

        for connection in self.connections:
            index_1 = self.labels.index(connection[0])
            index_2 = self.labels.index(connection[1])
            ax.plot(points3D_1[0, [index_1, index_2]], points3D_1[1, [index_1, index_2]], points3D_1[2, [index_1, index_2]], color='blue')
            ax.plot(points3D_2[0, [index_1, index_2]], points3D_2[1, [index_1, index_2]], points3D_2[2, [index_1, index_2]], color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot_interactive_mirrorfig(self, points3D1, points3D2):
        """
        Plot two sets of 3D points interactively using Plotly, with one set mirrored.
        """
        min_range_x = np.min([points3D1[0].min(), points3D2[0].min()])
        max_range_x = np.max([points3D1[0].max(), points3D2[0].max()])
        min_range_y = np.min([points3D1[1].min(), points3D2[1].min()])
        max_range_y = np.max([points3D1[1].max(), points3D2[1].max()])
        min_range_z = np.min([points3D1[2].min(), points3D2[2].min()])
        max_range_z = np.max([points3D1[2].max(), points3D2[2].max()])
        range_ = max(max_range_x - min_range_x, max_range_y - min_range_y, max_range_z - min_range_z)

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X', range=[min_range_x, min_range_x + range_]),
                yaxis=dict(title='Y', range=[min_range_y, min_range_y + range_]),
                zaxis=dict(title='Z', range=[min_range_z, min_range_z + range_]),
                aspectmode='cube',
                dragmode='orbit',
                uirevision=True
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        data = []
        for pair in self.connections:
            trace_lines_body = go.Scatter3d(
                x=[points3D1[0, pair[0]], points3D1[0, pair[1]]],
                y=[points3D1[1, pair[0]], points3D1[1, pair[1]]],
                z=[points3D1[2, pair[0]], points3D1[2, pair[1]]],
                mode='lines',
                line=dict(color='blue', width=2),
            )
            trace_lines_mirror = go.Scatter3d(
                x=[points3D2[0, pair[0]], points3D2[0, pair[1]]],
                y=[points3D2[1, pair[0]], points3D2[1, pair[1]]],
                z=[points3D2[2, pair[0]], points3D2[2, pair[1]]],
                mode='lines',
                line=dict(color='red', width=2),
            )
            data.append(trace_lines_body)
            data.append(trace_lines_mirror)

        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def plot_matplotlib_2D_plot(self, points2D,image=None, color='k-'):
        """
        Plot a 2D plot using Matplotlib with the given 2D points.It can be plotted against the reference image
        """
        points2D = self._transpose_if_needed(points2D)
        fig, ax = plt.subplots()
        if image is not None:
            ax.imshow(image)
        self._plot_connections(ax, points2D, color)
        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    def plot_matplotlib_2D_plot_2x(self, points2D_1, points2D_2, image=None):
        """
        Plot two sets of 2D points using Matplotlib, with different colors for each set. It can be plotted against the reference imagge
        """
        points2D_1 = self._transpose_if_needed(points2D_1)
        points2D_2 = self._transpose_if_needed(points2D_2)
        fig, ax = plt.subplots()
        if image is not None:
            ax.imshow(image)
        self._plot_connections(ax, points2D_1, 'k-')
        self._plot_connections(ax, points2D_2, 'r-')
        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()



    def plot_projected_points(self, points2D, points2D_2):
        """
        Plot two sets of 2D points with different colors based on their labels.
        """
        points2D = self._transpose_if_needed(points2D)
        points2D_2 = self._transpose_if_needed(points2D_2)
        fig, ax = plt.subplots()
        for connection in self.connections:
            idx1 = self.labels.index(connection[0])
            idx2 = self.labels.index(connection[1])
            color = 'red' if 'L' in connection[0] or 'left' in connection[0].lower() else 'blue'
            color_2 = 'pink' if 'R' in connection[0] else 'lightblue'
            ax.plot([points2D[0, idx1], points2D[0, idx2]], [points2D[1, idx1], points2D[1, idx2]], color=color)
            ax.plot([points2D_2[0, idx1], points2D_2[0, idx2]], [points2D_2[1, idx1], points2D_2[1, idx2]], color=color_2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

 
