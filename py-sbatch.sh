#!/bin/bash

###
# 
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=4
NUM_GPUS=1
JOB_NAME=
MAIL_USER=
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
# 	# -x nlp-2080-1,nlp-2080-2,nlp-a40-1 \
CONDA_HOME=$HOME/miniconda3
#CONDA_ENV=gen_eval
CONDA_ENV=
sbatch \
	-p nlp\
	-A nlp\
	-x nlp-2080-1,nlp-2080-2,nlp-a40-1 \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	--mem=64G \
	-o 'slurm-%N-%j.out' \
	-e 'slurm-%N-%j.err' \
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
