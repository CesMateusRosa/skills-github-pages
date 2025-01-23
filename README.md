<header>

<!--
  <<< Author notes: Course header >>>
  Include a 1280×640 image, course title in sentence case, and a concise description in emphasis.
  In your repository settings: enable template repository, add your 1280×640 social image, auto delete head branches.
  Add your open source license, GitHub uses MIT license.
-->

# Leveraging Mirror Geometry for Monocular 3D Reconstruction of Human Pose 
A project for avatar gamification in  Rehabilitation Therapy

</header>
This project is a 3D human pose estimation system that uses mirror geometry to overcome the inherent depth ambiguity in monocular pose estimation. By simulating a second view through the reflection, the system enables multi-view triangulation, achieving  accurate 3D human skeleton reconstruction.

<br>
<br>
<p align="center">
  <img src="Example.gif" alt="Example Animation" width="40%">
</p>

<br>



## Reconstruction Pipeline

This project introduces and compares two methods for mirror-based reconstruction: (1) estimating the mirror plane's normal vector and (2) applying the fundamental matrix approach in a mirrored configuration. Both methods leverage a deep 2D pose estimator, which provides 2D keypoint estimates that enable the inference of 3D mirror geometry. 

<br>
<br>
<p align="center">
  <img src="3D_reconstruction_pipeline.png" alt="Example Animation" width="40%">
</p>

<br>
This pipeline can be integrated with any 2D pose estimator, by providing the skeleton model and limb connections to the 3D estimator.

# Novel Keypoint Occlusion Method 
A novel method to estimate occluded points leveraging mirror geometry is employed. This method uses epipolar geometry constraints to establish reasonable guesses for the missing point. 

<br>
<br>
<p align="center">
  <img src="Occlusion Estimation.png" alt="Example Animation" width="40%">
</p>

<br>

# Main References
R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision. Cambridge University
Press, 2003.
C. Liu, Y. Li, K. Ma, D. Zhang, P. Bao, and Y. Mu, “Learning 3D human pose estimation from
catadioptric videos,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021.
</footer>
