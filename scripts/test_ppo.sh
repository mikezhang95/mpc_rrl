
# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/dl-tools/libomp5/usr/lib/x86_64-linux-gnu

# runs 
base_dir="outputs/carla_mpc-default"
# base_dir="outputs/carla_mpc-online"
# base_dir="outputs/carla_mpc-rnn_ppo-dr"

# perturbation
perturb_param_list="moi final_ratio tire_friction damping_rate drag_coefficient"
perturb_param_list=($perturb_param_list)
length=${#perturb_param_list[@]} 
 
# devices
seed=12345
port=20000
cuda_id=0

exp_dir=$base_dir/$seed 
for ((i=0; i<${length}; i++));do

    echo "Testing ${perturb_param_list[$i]}"
    export CUDA_VISIBLE_DEVICES=${cuda_id}

    # start carla
    sh $HOME/dl-tools/carla-0.9.12/CarlaUE4.sh -graphicsadapter=$cuda_id -RenderOffScreen -carla-rpc-port=$port -quality-level=low -benchmark -fps 20 -nosound &
    sleep 60
    echo "Carla Server Starts on ${port}"

    # run test
    python test_ppo.py \
           --experiments_dir ${exp_dir} \
           --agent_dir ${exp_dir} \
           --num_steps 100 \
           --perturb_param ${perturb_param_list[$i]} \
           --port $port &

    port=$(($port+2))
    cuda_id=$(($cuda_id+1))
done



