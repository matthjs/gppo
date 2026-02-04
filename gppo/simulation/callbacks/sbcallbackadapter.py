from stable_baselines3.common.callbacks import BaseCallback
from gppo.simulation.callbacks.abstractcallback import AbstractCallback
from gppo.simulation.callbacks.complexitycallback import TimeBudgetCallback
from gppo.simulation.simulatorldata import SimulatorRLData


class SB3CallbackAdapter(BaseCallback):
    """
    Adapter to wrap an AbstractCallback and integrate it with Stable Baselines3 training.

    Parameters
    ----------
    callback: AbstractCallback
        The custom callback to wrap.
    data: SimulatorRLData
        Data object with experiment/environment information for the callback.
    verbose: int
        Verbosity level (0 = silent, 1 = info, 2 = debug).
    """

    def __init__(self, callback: AbstractCallback, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback

    def _on_training_start(self) -> None:
        self.callback.on_training_start()

    def _on_step(self) -> bool:
        # SB3 step info: self.locals contains action, reward, next_obs, dones
        action = self.locals.get("actions", None)
        reward = self.locals.get("rewards", None)
        next_obs = self.locals.get("new_obs", None)
        done = self.locals.get("dones", None)

        # Call the wrapped callback
        continue_training = self.callback.on_step(action, reward, next_obs, done)
        return continue_training
    
    def on_rollout_start(self) -> None:
        self.callback.on_rollout_start()
        # A bit hacky but we will haave to do this for now.
        if isinstance(self.callback, TimeBudgetCallback):
            self.callback.on_learn({})
            self.callback.on_training_end()

    # def on_rollout_end(self) -> None:
    #     """Mark the start of a training update (when rollout buffer is full)."""
    #    self.callback.on_rollout_end()

    def on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (after collecting experiences).
        Can map to episode end if needed.
        """
        # Collect learning info from SB3
        self.callback.on_rollout_end()
            # self.callback.on_learn(learning_info)

    def _on_training_end(self) -> bool:
        # Called after every gradient update (learning step)
        # Do your "after update" logic here
        # print("Learning update finished!")
        # self.callback.on_learn({})
        # if isinstance(self.callback, TimeBudgetCallback):
            # A bit hacky but we will haave to do this for now.
        #    self.callback.on_training_end()
        return True

    # def _on_training_end(self) -> None:
    #    self.callback.on_training_end()

    def _on_rollout_start(self) -> None:
        self.callback.on_update_start()

    def _on_rollout_end_update(self) -> None:
        self.callback.on_update_end()
