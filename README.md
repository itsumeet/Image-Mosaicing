# ImageMosaicing
Real time stitching of frames captured from a camera(U.A.V) in this case.

The above code implements ->
1. Creation of blank canvas for stitching images
2. Feature Computation from ORB Classes
3. Feature Matching using BFMatcher (Brute-Force Algorithm)
4. Extraction of correspondences and their homographies
5. Warping of images from computed Homograph
