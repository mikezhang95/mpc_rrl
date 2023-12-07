
ps aux | grep carla | awk '{print $2}' | xargs kill -9
