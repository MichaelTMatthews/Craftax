import numpy as np


def main():
    cx_classic_sps = 70000
    cx_extended_sps = 35000

    num_repeats_main = 10
    num_env_steps = 1e9

    algorithm_speeds = {
        "PPO": 1.0,
        "PPO-LSTM": 2.0,
        "ICM": 2.0,
        "RND": 2.0,
        "E3B": 2.0,
    }

    steps_main = (
        num_repeats_main
        * num_env_steps
        * np.sum([v for v in algorithm_speeds.values()])
    )
    seconds_main = steps_main / cx_extended_sps
    gpu_days_main = seconds_main / 3600 / 24
    print(gpu_days_main)

    hyp_to_tune = {
        "lr": 5,
        "num_envs": 5,
        "num_minibatches": 5,
        "gamma": 5,
        "gae_lambda": 5,
        # "clip_eps": 5,
        "ent_coef": 2,
        # "vf_coef": 5,
        # "anneal_lr": 2,
        "layer_size": 5,
        "recurrent_state_size": 5,
        "icm_intrinsic_reward_scale": 5,
        "rnd_intrinsic_reward_scale": 5,
        "e3b_intrinsic_reward_scale": 5,
        "e3b_ridge_regulariser": 3,
    }

    hyp_repeats = 2

    steps_hyp = hyp_repeats * num_env_steps * np.sum([v for v in hyp_to_tune.values()])
    gpu_days_hyp = steps_hyp / cx_extended_sps / 3600 / 24

    print(gpu_days_hyp)


if __name__ == "__main__":
    main()
