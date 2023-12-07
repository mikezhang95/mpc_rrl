

# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/dl-tools/libomp5/usr/lib/x86_64-linux-gnu

overrides=carla_mpc

agent=rnn_ppo
alias=dr
num_train_env=2

cuda_id=0
port=20000

# for seed in 12345 23451 34512 45123 51234; do
for seed in 12345; do

    # set up cuda
    export CUDA_VISIBLE_DEVICES=${cuda_id}

    # start carla
    for i in $(seq 1 $num_train_env)
    do
        new_port=$(($port+2*i-2))
        sh $HOME/dl-tools/carla-0.9.12/CarlaUE4.sh -graphicsadapter=${cuda_id} -RenderOffScreen -carla-rpc-port=${new_port} -quality-level=low -benchmark -fps 20 &
        echo "Carla Server Starts on ${new_port}"
    done
    sleep 60

    # train
    python -u train_ppo.py \
        overrides=${overrides} \
        agent=${agent} \
        env.params.port=${port} \
        env.params.domain_random=False \
        agent.params.system_id_coef=1e-3 \
        agent.params.n_steps=512 \
        agent.params.batch_size=512 \
	agent.params.ent_coef=0.01 \
	agent.params.gae_lambda=0.98 \
        actor_lr=1e-4 \
        seed=${seed} \
        num_train_env=${num_train_env} \
        experiment=${agent}-${alias} 

    port=$(($port+2))
    cuda_id=$(($cuda_id+1))
done


# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
