
mkdir ./jobs 2> /dev/null

python -m initialiser \
	--env="Pendulum-v0" \
	--logdir="./jobs/" \
	--actors=2 \
       	--generations=1 \
       	--episodes=1 \
       	--duration=1 \
       	--actor-type="PendulumDNNActor" \

