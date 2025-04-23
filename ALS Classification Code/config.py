
###################################
######### Path Configure ##########
###################################

# Path to metadata which contains at least sample name column and corresponding label column
META_PATH = '../dataset/metadata/mouse_sample_names_sra.tsv'

# Path to spatial transcriptomics imaging data
IMG_PATH = '../dataset/image/'

# Path to spatial transcriptomics gene counts data
CM_PATH = '../dataset/cm/'

ATM_PATH = None

# Path to folder that save the tiles
TILE_PATH = '../dataset/tile/'

# Path to save intermediate output and final result
DATASET_PATH = '../dataset/'

####################################
######### Image Configure #########
####################################

# Tile size (DO NOT CHANGE)
SIZE = 299, 299
# Color channel (RGB)
N_CHANNEL = 3

NORM_METHOD = 'vahadane'

##########################################
######### Count Matrix Configure #########
##########################################

# Threshold for remove low abundant gene, genes expressed in less than THRESHOLD_GENE of total number spots  will be removed
THRESHOLD_GENE = 0.01

# Threshold for remove low quality spots, spots with less than THRESHOLD_SPOT of total genes expressed will be removed
THRESHOLD_SPOT = 0.01

# Minimum gene count value for counting whether expressed or not
# which are used for removing low abundant gene and low quality gene.
# genes with count value higher than MIN_EXP are considered as expressed genes
MIN_EXP = 1

######################################
######### Metadata configure #########
######################################

# Specify column name of sample name column, label column and
# condition column (used for subset if provided otherwise leave it to None) in metadata file
SAMPLE_COLUMN = 'sample_name'
LABEL_COLUMN = 'age'
CONDITION_COLUMN = 'breed'

# Subset dataset that all samples have certain CONDITION in CONDITION_COLUMN
CONDITION = 'B6SJLSOD1-G93A'
ADDITIONAL_COLUMN = 2 if CONDITION_COLUMN else 1

# Set random seed for reproducibility
seed = 37
