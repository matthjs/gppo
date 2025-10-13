from stable_baselines3.common.callbacks import BaseCallback
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData


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

    def on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (after collecting experiences).
        Can map to episode end if needed.
        """
        # Collect learning info from SB3
        learning_info = {} # TODO: For now just make this empty # self.locals.get("infos", [{}])[0]  # SB3 puts info dict in 'infos'
        if isinstance(learning_info, dict) and learning_info:
            self.callback.on_rollout_end()
            # self.callback.on_learn(learning_info)

    def _on_training_end(self) -> None:
        self.callback.on_training_end()

    def _on_rollout_start(self) -> None:
        self.callback.on_update_start()

    def _on_rollout_end_update(self) -> None:
        self.callback.on_update_end()
