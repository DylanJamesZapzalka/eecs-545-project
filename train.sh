#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=eecs545w24_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=2:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 10000
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=results/test.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
conda init
conda activate dpo_kd
python3 fine-tune-phi-qlora-dpo.py --save_root /home/bswang/Courses/EECS545/results --dataset summarization --model_name test_model --lr 5e-4 --num_epochs 1 --curr_epoch $1