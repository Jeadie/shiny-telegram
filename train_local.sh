
mkdir ./jobs 2> /dev/null

python -m initialiser \
	--env="Pendulum-v0" \
	--logdir="./jobs/" \
	--actors=12 \
       	--generations=200 \
       	--episodes=6 \
       	--duration=8 \
       	--actor-type="PendulumDNNActor" \

