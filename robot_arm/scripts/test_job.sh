. /home/tty6/miniforge3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
cd /home/tty6/MyoSuite-Assistive
conda activate myosuite
export PYTHONPATH=/home/tty6/MyoSuite-Assistive:$PYTHONPATH
find /dev/shm -maxdepth 1 -user "$USER" -type f -print -delete || true
ipcs -s | awk -v u="$USER" '$3==u {print $2}' | xargs -r -n1 ipcrm -s
python -m deprl.main