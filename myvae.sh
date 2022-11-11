#!/bin/bash
#SBATCH --time=0-02:59
#SBATCH --account=def-ayassine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G

module load python/3 
module load cuda cudnn 
cd ~/projects/def-ayassine/shamisa/VAE-NILM
source vae-venv/bin/activate 

ssh shamisa@10.29.77.39 "ssh -f -N -L 3128:localhost:3128 m5naseri@0.tcp.ngrok.io -p 10656" &
sleep 3
ssh -f -N -L 3128:localhost:3128 shamisa@10.29.77.39
sleep 3
export http_proxy=http://127.0.0.1:3128
export https_proxy=http://127.0.0.1:3128
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128
curl https://ipinfo.io/ip

wandb online
wandb login b76283bc6c04e2ce6611147c4d328f71af8c71ba
wandb agent shamisa-kp/vae_myown/wvc4yzmz

