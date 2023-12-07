
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICS=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yzhang/dl-tools/libomp5/usr/lib/x86_64-linux-gnu

sh /home/yzhang/dl-tools/carla-0.9.12/CarlaUE4.sh -graphicsadapter=2 -RenderOffScreen -carla-rpc-port=2000 -quality-level=low -benchmark -fps 20 &

