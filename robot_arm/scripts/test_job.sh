cd /home/tty6/Myosuite-Assistive
conda activate myosuite
for i in {1..3}; do
    SEED=$RANDOM
    python -m deprl.main robot_arm/train.yaml -m reward.reward_scale=1.0,2.0 tonic.seed=$SEED
done
