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

![](Example.gif)
## Reconstruction Pipeline

This project introduces and compares two methods for mirror-based reconstruction: (1) estimating the mirror plane's normal vector and (2) applying the fundamental matrix approach in a mirrored configuration. Both methods leverage a deep 2D pose estimator, which provides 2D keypoint estimates that enable the inference of 3D mirror geometry.

![](3D_reconstruction_pipeline.png)

&copy; 2023 GitHub &bull; [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) &bull; [MIT License](https://gh.io/mit)

</footer>
