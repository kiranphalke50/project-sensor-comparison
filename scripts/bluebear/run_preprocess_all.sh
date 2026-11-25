#!/usr/bin/env bash
#SBATCH --qos=bbdefault
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --account=kowalcau-opm-recordings

set -e

module purge
module load bear-apps/2022a
module load MNE-Python/1.3.1-foss-2022a

# define a environment variable to signify that the script is running on BlueBEAR
export BLUEBEAR=1

# run concat script for all subjects/runs
python /rds/projects/k/kowalcau-tms-opm-meg/project-sensor_comparison/code/project-sensor-comparison/scripts/nmor_data_analysis/preprocess_nmor_raw_data.py
