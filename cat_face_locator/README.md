This was a quick subproject to locate cat faces in images via a neural network.
It was implemented in order to increase the available number of cat images in the training set (as 10k is probably very limiting for the GANs).
The network worked well, but not well enough. Sometimes it viewed stuff as cat faces that were clearly not cat faces.
It was especially not able to accurately detect the angle of the eyes (relative to the x axis).
Therefore, rotations could not be removed reliably. (Might be no longer a problem with spatial transformers in D.)

The code is very messy. Most parts could and should be replaced by the classes in `dataset/dataset.py`.
I will probably extract the code into its own project and clean it up at some point in the future.
