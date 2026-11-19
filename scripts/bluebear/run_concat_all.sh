#!/usr/bin/env bash
#SBATCH --qos=bbdefault
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --account=kowalcau-opm-recordings

set -e

module purge
module load bear-apps/2022a
module load MNE-Python/1.3.1-foss-2022a

# run concat script for all subjects/runs
python /rds/projects/k/kowalcau-tms-opm-meg/project-sensor_comparison/code/project-sensor-comparison/scripts/gen2_analysis_harrys_script/concat.py
