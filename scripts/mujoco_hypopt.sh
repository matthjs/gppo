#!/bin/bash

# TODO: EDIT/FIX IT

# Define environments and their corresponding parameters
# Format: ENVIRONMENT NUM_ENVS NUM_RUNS NUM_EPISODES
ENV_CONFIGS=(
    "Hopper-v5 1 1 1000"
    "HalfCheetah-v5 2 3 1500"
    "Walker2d-v5 4 2 2000"
)

# Define agents
AGENTS=("GPPO" "SB_PPO")

# Loop over environments first
for ENV_LINE in "${ENV_CONFIGS[@]}"; do
    read -r ENV N_ENVS NUM_RUNS NUM_EPISODES <<< "$ENV_LINE"

    # Loop over agents
    for AGENT in "${AGENTS[@]}"; do
        echo "Running main.py with environment=$ENV, agent=$AGENT, n_envs=$N_ENVS, num_runs=$NUM_RUNS, num_episodes=$NUM_EPISODES"

        python main.py \
            mode=hpo_gppo \
            environment=$ENV \
            agent=$AGENT \
            n_envs=$N_ENVS \
            num_runs=$NUM_RUNS \
            num_episodes=$NUM_EPISODES \
            hydra.run.dir=outputs/${ENV}_${AGENT}_${N_ENVS}env_${NUM_RUNS}runs
    done
done
