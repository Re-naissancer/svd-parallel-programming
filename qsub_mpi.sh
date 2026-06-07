#!/bin/sh
#PBS -N svd_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)

EXE=${EXE:-main_mpi}
NP=${NP:-8}
SEED=${SEED:-20260410}
STEPS=${STEPS:-6000}

for node in $NODES; do
    scp master_ubss1:/home/${USER}/svd/${EXE} ${node}:/home/${USER}/${EXE} 1>&2
done

export SVD_MPI_LOCAL_STEPS=${STEPS}

/usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/${EXE} ${SEED}
