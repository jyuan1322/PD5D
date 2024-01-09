#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH --mem=10G
#SBATCH -p medium
#SBATCH -e logs/Rscript-%J.log
for Ident in $(tail -n +2 Files/SubclusterIdents.txt); do
    sbatch --export=clusterident=$Ident MousePFC_snRNA-seq_Workflow_Part4-2.sh
done
