This archive contains the EgoCart dataset (http://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/) for large-scale, image-based, indoor localization in a reatail store. Images and depth maps have been collected using cameras placed on shopping carts. The dataset contains 19531 image-depth pairs along with the related 3DOF positions and class labels. Please see http://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/ and the related publications for more information.

The dataset is divided into training and test set. For each of the sets, we provide:
 - a directory containing all RGB images;
 - a directory containing all depth maps;
 - a text file specifying RGB-depth pairs and associated labels. Each row of the text file represents a sample and has the following format:
	rgb_image_filename depth_image_filename x y u v c
   where:
	- (rgb_image_filename, depth_image_filename) is the RGB-depth pair;
	- (x,y) represents the position of the RGB-depth pair in the store (in meters);
	- (u,v) is a unit vector representing the orientation of the camera in the store;
	- c is the class label (1-16)