import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from Base_Pose_Estimator import BasePoseEstimator
from Base_Pose_Estimator import LimbAssignment
class create_animation(BasePoseEstimator):
    
    def __init__(self, keypoint_labels, connections):
        super().__init__(keypoint_labels, connections)
        self.limb_assignment = LimbAssignment(self.labels, self.connections)
        self.left_limbs = {k: v for k, v in self.limb_assignment.left_limbs.items() if not k.startswith('left_body_part_')}
        self.right_limbs = {k: v for k, v in self.limb_assignment.right_limbs.items() if not k.startswith('right_body_part_')}
    
    
    def create_animation_2D(self,points_realworld_all_frames, points_mirror_all_frames, interval=50):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Initialize lists to store all x and y coordinates
        all_x, all_y = [], []

        # Gather all x and y coordinates from both persons' frames for axis limits
        for frames in [points_realworld_all_frames, points_mirror_all_frames]:
            for frame_data in frames:
                for x, y in frame_data:
                    if x != 0 and y != 0:  # Only consider non-zero coordinates
                        all_x.append(x)
                        all_y.append(y)

        # Calculate the min and max values for setting axis limits
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)

        # Set the axis limits with some padding
        ax.set_xlim([xmin - 10, xmax + 10])
        ax.set_ylim([ymin - 10, ymax + 10])

        # Invert the y-axis
        ax.invert_yaxis()

        # Define colors for right and left sides
        right_color_real = 'red'
        left_color_real = 'blue'
        right_color_mirror = 'lightcoral'
        left_color_mirror = 'lightblue'

        # Create plots for person 1
        body_part_plots_person1 = [ax.plot([], [], 'o', color=right_color_real if i % 2 == 0 else left_color_real)[0]
                                for i in range(len(points_realworld_all_frames[0]))]
        connection_plots_person1 = [ax.plot([], [], 'b-')[0] for _ in self.connections]

        # Create plots for person 2
        body_part_plots_person2 = [ax.plot([], [], 'o', color=right_color_mirror if i % 2 == 0 else left_color_mirror)[0]
                                for i in range(len(points_mirror_all_frames[0]))]
        connection_plots_person2 = [ax.plot([], [], 'g-')[0] for _ in self.connections]

        def update(frame):
            # Remove the previous frame id text if it exists
            if hasattr(update, 'frame_id_text') and update.frame_id_text:
                update.frame_id_text.remove()

            # Display the frame id in the top right corner
            update.frame_id_text = ax.text(0.95, 0.01, f'Frame: {frame}',
                                        verticalalignment='bottom', horizontalalignment='right',
                                        transform=ax.transAxes,
                                        color='green', fontsize=15)

            # Update the body part plots for person 1
            for i, plot in enumerate(body_part_plots_person1):
                x, y = points_realworld_all_frames[frame][i]
                if x != 0 and y != 0:
                    color = right_color_real if i % 2 == 0 else left_color_real
                    plot.set_data([x], [y])  # Wrapping x and y as lists
                    plot.set_color(color)
                else:
                    plot.set_data([], [])

            # Update the connection plots for person 1
            for i, (start_part, end_part) in enumerate(self.connections):
                start_x, start_y = points_realworld_all_frames[frame][self.labels.index(start_part)]
                end_x, end_y = points_realworld_all_frames[frame][self.labels.index(end_part)]
                if (start_x != 0 and start_y != 0) and (end_x != 0 and end_y != 0):
                    connection_plots_person1[i].set_data([start_x, end_x], [start_y, end_y])  # Wrapping both as lists
                else:
                    connection_plots_person1[i].set_data([], [])

            # Update the body part plots for person 2
            for i, plot in enumerate(body_part_plots_person2):
                x, y = points_mirror_all_frames[frame][i]
                if x != 0 and y != 0:
                    color = right_color_mirror if i % 2 == 0 else left_color_mirror
                    plot.set_data([x], [y])  # Wrapping x and y as lists
                    plot.set_color(color)
                else:
                    plot.set_data([], [])

            # Update the connection plots for person 2
            for i, (start_part, end_part) in enumerate(self.connections):
                start_x, start_y = points_mirror_all_frames[frame][self.labels.index(start_part)]
                end_x, end_y = points_mirror_all_frames[frame][self.labels.index(end_part)]
                if (start_x != 0 and start_y != 0) and (end_x != 0 and end_y != 0):
                    connection_plots_person2[i].set_data([start_x, end_x], [start_y, end_y])  # Wrapping both as lists
                else:
                    connection_plots_person2[i].set_data([], [])

        # Create the animation
        anim = FuncAnimation(fig, update, frames=range(len(points_realworld_all_frames)), interval=interval)

        # Show the plot
        return anim   
        
    def animate_pose_3D(self,pose_3d, interval=50):
        # Ensure pose_3d is a numpy array
        pose_3d = np.array(pose_3d)
        
        #Manter a escala para toda a sequência
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        lines = []  # Initialize the lines list outside the loop

        for start, end in self.connections:
            start_idx, end_idx = self.labels.index(start), self.labels.index(end)
            # Append line objects for each connection
            line = ax.plot([pose_3d[0, 0, start_idx], pose_3d[0, 0, end_idx]],
                        [pose_3d[0, 1, start_idx], pose_3d[0, 1, end_idx]],
                        [pose_3d[0, 2, start_idx], pose_3d[0, 2, end_idx]],
                        linestyle="-", marker="o")[0]
            lines.append(line)  # Append the created line object to the lines list

        def update_graph(num, pose_3d, lines):
            for line, (start, end) in zip(lines, self.connections):
                start_idx, end_idx = self.labels.index(start), self.labels.index(end)
                start_point = pose_3d[num, :3, start_idx]
                end_point = pose_3d[num, :3, end_idx]
                line.set_data([start_point[0], end_point[0]], [start_point[1], end_point[1]])
                line.set_3d_properties([start_point[2], end_point[2]])
            return lines
        # Setting the axes properties
        ax.set_xlim3d([np.min(pose_3d[:, 0]), np.max(pose_3d[:, 0])])
        ax.set_ylim3d([np.min(pose_3d[:, 1]), np.max(pose_3d[:, 1])])
        ax.set_zlim3d([np.min(pose_3d[:, 2]), np.max(pose_3d[:, 2])])

        # Creating the animation
        anim = animation.FuncAnimation(fig, update_graph, frames=len(pose_3d), fargs=(pose_3d, lines),
                                    interval=interval, blit=False)

        return anim 
    
    def animate_pose_3D_compare(self, pose_3d_1, pose_3d_2, connections_2, labels_2, interval=50):
        # Ensure pose_3d_1 and pose_3d_2 are numpy arrays
        pose_3d_1 = np.array(pose_3d_1)
        pose_3d_2 = np.array(pose_3d_2)
        
        # Print shapes for debugging
        print("pose_3d_1 shape:", pose_3d_1.shape)
        print("pose_3d_2 shape:", pose_3d_2.shape)
        
        # Ensure both pose arrays have the same number of frames
        min_frames = min(pose_3d_1.shape[0], pose_3d_2.shape[0])
        pose_3d_1 = pose_3d_1[:min_frames]
        pose_3d_2 = pose_3d_2[:min_frames]
        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        lines_1 = []  # Initialize the lines list for the first pose
        lines_2 = []  # Initialize the lines list for the second pose

        # Plotting connections for the first pose
        for start, end in self.connections:
            start_idx, end_idx = self.labels.index(start), self.labels.index(end)
            line_1 = ax.plot([pose_3d_1[0, 0, start_idx], pose_3d_1[0, 0, end_idx]],
                            [pose_3d_1[0, 1, start_idx], pose_3d_1[0, 1, end_idx]],
                            [pose_3d_1[0, 2, start_idx], pose_3d_1[0, 2, end_idx]],
                            linestyle="-", marker="o", color="blue")[0]
            lines_1.append(line_1)

        # Plotting connections for the second pose
        for start, end in connections_2:
            start_idx, end_idx = labels_2.index(start), labels_2.index(end)
            line_2 = ax.plot([pose_3d_2[0, 0, start_idx], pose_3d_2[0, 0, end_idx]],
                            [pose_3d_2[0, 1, start_idx], pose_3d_2[0, 1, end_idx]],
                            [pose_3d_2[0, 2, start_idx], pose_3d_2[0, 2, end_idx]],
                            linestyle="-", marker="o", color="red")[0]
            lines_2.append(line_2)

        def update_graph(num, pose_3d_1, pose_3d_2, lines_1, lines_2):
            # Update lines for the first pose
            for line, (start, end) in zip(lines_1, self.connections):
                start_idx, end_idx = self.labels.index(start), self.labels.index(end)
                line.set_data([pose_3d_1[num, 0, start_idx], pose_3d_1[num, 0, end_idx]],
                            [pose_3d_1[num, 1, start_idx], pose_3d_1[num, 1, end_idx]])
                line.set_3d_properties([pose_3d_1[num, 2, start_idx], pose_3d_1[num, 2, end_idx]])

            # Update lines for the second pose
            for line, (start, end) in zip(lines_2, connections_2):
                start_idx, end_idx = labels_2.index(start), labels_2.index(end)
                line.set_data([pose_3d_2[num, 0, start_idx], pose_3d_2[num, 0, end_idx]],
                            [pose_3d_2[num, 1, start_idx], pose_3d_2[num, 1, end_idx]])
                line.set_3d_properties([pose_3d_2[num, 2, start_idx], pose_3d_2[num, 2, end_idx]])

            return lines_1 + lines_2
        
        # Setting the axes properties
        ax.set_xlim3d([np.min(pose_3d_1[:, 0]), np.max(pose_3d_1[:, 0])])
        ax.set_ylim3d([np.min(pose_3d_1[:, 1]), np.max(pose_3d_1[:, 1])])
        ax.set_zlim3d([np.min(pose_3d_1[:, 2]), np.max(pose_3d_1[:, 2])])

        # Creating the animation
        anim = animation.FuncAnimation(fig, update_graph, frames=min_frames, fargs=(pose_3d_1, pose_3d_2, lines_1, lines_2),
                                    interval=interval, blit=False)

        return anim

    def animate_pose_3D_left_right(self, pose_3d, interval=50):
        # Ensure pose_3d is a numpy array
        pose_3d = np.array(pose_3d)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        lines = []  # Initialize the lines list outside the loop

        # Define colors for left and right limbs
        left_color = 'blue'
        right_color = 'red'

        for start, end in self.connections:
            start_idx, end_idx = self.labels.index(start), self.labels.index(end)

            # Determine color based on left or right limbs
            if start.startswith('left') or end.startswith('left'):
                color = left_color
            elif start.startswith('right') or end.startswith('right'):
                color = right_color
            else:
                color = 'green'  # default for any other part

            # Append line objects for each connection
            line = ax.plot([pose_3d[0, 0, start_idx], pose_3d[0, 0, end_idx]],
                        [pose_3d[0, 1, start_idx], pose_3d[0, 1, end_idx]],
                        [pose_3d[0, 2, start_idx], pose_3d[0, 2, end_idx]],
                        linestyle="-", marker="o", color=color)[0]
            lines.append(line)  # Append the created line object to the lines list

        def update_graph(num, pose_3d, lines):
            for line, (start, end) in zip(lines, self.connections):
                start_idx, end_idx = self.labels.index(start), self.labels.index(end)
                start_point = pose_3d[num, :3, start_idx]
                end_point = pose_3d[num, :3, end_idx]
                line.set_data([start_point[0], end_point[0]], [start_point[1], end_point[1]])
                line.set_3d_properties([start_point[2], end_point[2]])
            return lines

        # Setting the axes properties
        ax.set_xlim3d([np.min(pose_3d[:, 0]), np.max(pose_3d[:, 0])])
        ax.set_ylim3d([np.min(pose_3d[:, 1]), np.max(pose_3d[:, 1])])
        ax.set_zlim3d([np.min(pose_3d[:, 2]), np.max(pose_3d[:, 2])])

        # Creating the animation
        anim = animation.FuncAnimation(fig, update_graph, frames=len(pose_3d), fargs=(pose_3d, lines),
                                    interval=interval, blit=False)

        return anim
        


   