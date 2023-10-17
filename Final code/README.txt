The files  to be run in a Jupyter Notebook. Store separate systems in separate folders.

To import data from different files, only the FileName variable needs to be changed to the appropriate name. This will also save all data to that file.

model.py : stores functions such as the vector field and eigenbasis coordinates used in multiple notebooks

HyperbolicTrajectory.ipynb: set inital model and forcing parameters, finds and saves the hyperbolic trajectories corresponding to capsize

GenerateCentreManifold.ipynb: Requires as input the hyperbolic trajectories to be saved in the same folder. Outputs a collection of points on the centre manifold along with their 
coordinates from the immersion stored as an array

GenerateStableManifold.ipynb: Requires as input the hyperbolic trajectories to be saved in the same folder. Outputs a collection of points on the stable manifold along with their 
coordinates from the immersion stored as an array

CentreManifoldPlots.ipynb: requires as input the hyperbolic trajectories and output from GenerateCentreManifold.ipynb. Produces the plots to visualise the centre manifold.

StableManifoldPlots.ipynb: requires as input the hyperbolic trajectories and output from GenerateStableManifold.ipynb. Produces a discretisation used to visualise the stable
manifold in the .m file.

StableManifoldImmersion.py stores functions that takes as input the output of the GenerateStableManifold.ipynb file and outputs the immersion for the stable manifold.

SimulationsToCapsize.ipynb: requires the hyperbolic trajectories and stable manifold data and to be stored in same file as the StableManifoldImmersion.py file. Produces the 
visualisations of stable manifold and points moving to capsize as well as computing the integrity measure of the upright state.

