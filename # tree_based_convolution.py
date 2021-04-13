# tree-based convolution (TBCNN)

# Our objective is apply a set of fixed-depth feature detectors sliding over the entire tree

# vec(·) has a feature_size N_f

# Set a fixed-depth window (wtih n nodes)

# N_c is the number of feature detectors

# bias_conv has a size of N_f
# weight_conv has a size of N_c x N_f

#eta_top parameters
#d_i: the depth of the node i in the sliding window
#d: the depth of the window

#eta_right parameters
#p_i : the position of the node i
#n: the total number of p’s siblings (nodes on the same hierarchical level under the same parent node), including p itself
