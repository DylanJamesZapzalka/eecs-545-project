#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=job3
#SBATCH --account=stellayu0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 16000
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=results/jobA.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
conda init
conda activate dpo_kd
python3 phi2_query.py --dataset summarization --trials 4 --input_dir /home/bswang/Courses/EECS545/results/code_summarization/online_lr=0.0005_e=$1 --trial_num $2