"""Custom AgentLoop overrides for per-batch prompt length support.

verl's AgentLoopWorkerBase pads all prompts to a single global
``prompt_length``.  For self-teach, A₂ prompts (which contain Q + A₁ + F)
need a larger budget than A₁/F prompts.

This module provides a worker subclass that checks for a per-sample
``max_prompt_length`` key in kwargs (populated from ``non_tensor_batch``)
and temporarily swaps the config value during post-processing.

Usage — add to your run script:
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=\
        src.self_teach.agent_loop_overrides.FlexPromptAgentLoopManager
"""

import ray
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorkerBase,
    _InternalAgentLoopOutput,
)


@ray.remote
class FlexPromptAgentLoopWorker(AgentLoopWorkerBase):
    """AgentLoopWorker that supports per-batch prompt padding length."""

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        override = kwargs.pop("max_prompt_length", None)
        if override is not None:
            original = self.config.actor_rollout_ref.rollout.prompt_length
            self.config.actor_rollout_ref.rollout.prompt_length = int(override)
            try:
                return await super()._agent_loop_postprocess(output, **kwargs)
            finally:
                self.config.actor_rollout_ref.rollout.prompt_length = original
        return await super()._agent_loop_postprocess(output, **kwargs)


class FlexPromptAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that uses FlexPromptAgentLoopWorker."""

    agent_loop_workers_class = FlexPromptAgentLoopWorker
