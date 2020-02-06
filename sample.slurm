#!/bin/bash

#SBATCH --job-name=in5550
#SBATCH --account=nn9447k
#SBATCH --mail-type=FAIL
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --ntasks-per-node=2

source ${HOME}/.bashrc

#
# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
#
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load IN5550
module --force purge
module use -a /cluster/shared/nlpl/software/modules/etc
module add nlpl-in5550/202001/3.7

echo "submission directory: ${SUBMITDIR}"
ulimit -a
module list

# by default, pass on any remaining command-line options
python3 main.py ${@}
