#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc
 
# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:1
#SBATCH --time 1-23:50:00
#SBATCH --cpus-per-task 9
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-type FAIL,END

rm slurm*.out
conda activate tf2gpu
python3 MIA.py