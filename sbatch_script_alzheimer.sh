#!/bin/bash
###
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# First argument should be the Python script to run
# Second argument should be the directory containing configuration files
#
###
# Example usage:
#
# ./py-sbatch.sh myscript.py /path/to/config/files/
#
###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=4
NUM_GPUS=1
JOB_NAME="alzheimer_downstream_task"
MAIL_USER="your-mail" #replace this field with your mail
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniforge3
CONDA_ENV=alz_proj

# Extract config filename
for arg in "$@"; do
    if [[ $arg == *"configs"* ]]; then
        CONFIG_PATH=$arg
        break
    fi
done

# Create output filename
# Remove test_configs/ from the start and .yml from the end
OUTPUT_NAME=$(echo "$CONFIG_PATH" | sed 's|configs/||' | sed 's|\.yml$||')
# Replace remaining / with _
OUTPUT_NAME=$(echo "$OUTPUT_NAME" | sed 's|/|_|g')
OUTPUT_NAME="${OUTPUT_NAME}.txt"

# Debug prints
echo "Config path: $CONFIG_PATH"
echo "Output path: $OUTPUT_NAME"

sbatch \
    -N $NUM_NODES \
    -c $NUM_CORES \
    --gres=gpu:$NUM_GPUS \
    --job-name $JOB_NAME \
    --mail-user $MAIL_USER \
    --mail-type $MAIL_TYPE \
    -o "$OUTPUT_NAME"  \
    -w gipdeep1 \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
# Run python with the args to the script
python $@
echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF