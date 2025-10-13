#!/bin/bash

# Define environments and their corresponding parameters
# Format: ENVIRONMENT NUM_ENVS NUM_RUNS NUM_EPISODES
ENV_CONFIGS=(
    "Humanoid-v5 8 1 15000"
    "Walker2d-v5 8 1 15000"
    "Ant-v5 8 1 15000"
    "HalfCheetah-v5 8 1 15000"
    "Hopper-v5 8 1 15000"
)

EXP_ID="mujoco_experiment_eurips"
AGENT_PREFIX="gppo"

# Loop over environments
for ENV_LINE in "${ENV_CONFIGS[@]}"; do
    read -r ENV N_ENVS NUM_RUNS NUM_EPISODES <<< "$ENV_LINE"

    # Construct agent name based on prefix and environment
    AGENT="${AGENT_PREFIX}_${ENV}"

    echo "Running main.py with environment=$ENV, agent=$AGENT, n_envs=$N_ENVS, num_runs=$NUM_RUNS, num_episodes=$NUM_EPISODES"

    time python main.py \
        exp_id=$EXP_ID \
        environment=$ENV \
        agent=$AGENT \
        n_envs=$N_ENVS \
        num_runs=$NUM_RUNS \
        num_episodes=$NUM_EPISODES \
        # hydra.run.dir=outputs/${ENV}_${AGENT}_${N_ENVS}env_${NUM_RUNS}runs
done
