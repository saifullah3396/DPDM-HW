srun --gpus-per-task=0 \
  --container-image=/netscratch/$USER/envs/TORCH_FUSION_v4.sqsh \
  --container-workdir=/home/$USER/projects/docmm \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/:/home/$USER/ \
  --container-save=/netscratch/$USER/envs/TORCH_FUSION_v5.sqsh \
  --partition V100-16GB \
  --pty /bin/bash
