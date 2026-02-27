"""Entry point for Self-Play Teaching (SPT) GRPO training.

Mirrors verl.trainer.main_ppo but uses SPTTaskRunner to inject
Self-Play Teaching into the training loop.
"""

import os

import hydra
import ray

from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from .spt_trainer import SPTTaskRunner

# Point Hydra to verl's config directory
import verl.trainer.main_ppo as _main_ppo_mod
_VERL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(_main_ppo_mod.__file__)), "config")


@hydra.main(config_path=_VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SPTTaskRunner))


if __name__ == "__main__":
    main()
