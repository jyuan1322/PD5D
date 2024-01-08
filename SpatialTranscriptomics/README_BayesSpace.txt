# Script to detect 6 cortex layers + WM in Visium 10X ST data using BayesSpace

# To run the script
Rscript ./run_bayesspace.R -i /path/to/outs -o /path/to/layers_output.png -t sample_name

# This script is based on the DLPFC example from the BayesSpace website:
# https://edward130603.github.io/BayesSpace/articles/maynard_DLPFC.html

# For now, parameters are kept the same as the tutorial
# - Fixed k=7 clusters to discover
# Preprocessing: 
# -- Extract 2000 HVGs using scran
# -- Perform PCA with 15 PCs – cluster on the PC space

# Note: I rotate all BayesSpace plots 180 degrees to align with the sample images
