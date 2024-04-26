#!/bin/sh
#SBATCH -p normal
##SBATCH -t 1-12
#SBATCH -N 1 -n 48
#SBATCH -J test
#SBATCH -e err
ulimit -a
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

#source /public/home/jlyang/mahuan/anaconda3/etc/profile.d/conda.sh
#source /public/home/jlyang/mahuan/anaconda3/bin/activate quantum
source /public/home/jlyang/quantum/anaconda3/bin/activate vqechem

python test.py > runlog

