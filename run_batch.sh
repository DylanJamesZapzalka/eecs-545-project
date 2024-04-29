#!/bin/bash
script_dir=/home/bswang/Courses/EECS545
sbatch --dependency=singleton ${script_dir}/run.sh 0 && \
sbatch --dependency=singleton ${script_dir}/run.sh 0 && \
sbatch --dependency=singleton ${script_dir}/run.sh 0 && \
sbatch --dependency=singleton ${script_dir}/train.sh 1 && \
sbatch --dependency=singleton ${script_dir}/run.sh 1 && \
sbatch --dependency=singleton ${script_dir}/run.sh 1 && \
sbatch --dependency=singleton ${script_dir}/run.sh 1 && \
sbatch --dependency=singleton ${script_dir}/train.sh 2 && \
sbatch --dependency=singleton ${script_dir}/run.sh 2 && \
sbatch --dependency=singleton ${script_dir}/run.sh 2 && \
sbatch --dependency=singleton ${script_dir}/run.sh 2
# bash /home/bswang/Courses/EECS545/run.sh 1 && \
# bash /home/bswang/Courses/EECS545/run.sh 1 && \
# bash /home/bswang/Courses/EECS545/train.sh 2