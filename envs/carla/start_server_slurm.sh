#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 4000 # memory pool for each core (4GB)
#SBATCH -t 0-00:01 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -o log/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J carla_server # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Start CARLA Simulator
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yzhang/dl-tools/libomp5/usr/lib/x86_64-linux-gnu
sh /home/yzhang/dl-tools/carla-0.9.12/CarlaUE4.sh -graphicsadapter=0 -RenderOffScreen -carla-rpc-port=20000 -quality-level=low -benchmark -fps 20 -nosound &
sleep 10
echo "Carla Server Starts at port 20000"

# Test Client
python carla_client_test.py
echo "Carla Client Connected"

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";

