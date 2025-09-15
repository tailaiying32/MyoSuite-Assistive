cd /home/tty6/Myosuite-Assistive
conda activate myosuite
export PYTHONPATH=/home/tty6/MyoSuite-Assistive:$PYTHONPATH
SEED=$RANDOM
python -m deprl.main \
    -m env_parameters.goal="[0.9,0,1.8]","[0.6, 0, 1.5]" \
    env_parameters.reward_scale=1.0,2.0 \
    tonic.seed=$SEED

