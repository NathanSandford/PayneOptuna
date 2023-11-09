from typing import List, Optional, Union
from pathlib import Path
import numbers
import torch
import pytorch_lightning as pl
import optuna


class MetricsCallback(pl.Callback):
    """
    PyTorch Lightning callback to log training metrics.

    :param list[str] metrics_to_track: List of metrics to log
    """

    def __init__(self, metrics_to_track: List[str]) -> None:
        super().__init__()
        self.metrics = {metric: [] for metric in metrics_to_track}

    def on_validation_end(self, trainer, pl_module):
        for metric in self.metrics.keys():
            if metric in trainer.callback_metrics.keys():
                self.metrics[metric].append(trainer.callback_metrics[metric])


class CheckpointCallback(pl.callbacks.ModelCheckpoint):
    """
    PyTorch Lightning callback to save model checkpoints during training.
    This is only slightly updated from pl.callbacks.ModelCheckpoint to improve the clarity in the print statements.

    :param Union[str,Path] dirpath: Path to checkpoint directory
    :param str filename: Formatting for checkpoint file names
    :param str monitor: Metric to monitor for improvements. Most likely "val-loss" for the Payne.
    :param str mode: "min" or "max" depending on metric. Most likely "min" if monitor = "val-loss".
    :param int every_n_epochs: Number of epochs between checkpoints. Default = 1.
    :param Optional[int] trial_number: Trial number of model (i.e., when tuning w/ Optuna). Default = None.
    :param bool verbose: If true, prints updates after each validation epoch. Default = True.
    """

    def __init__(
        self,
        dirpath: Union[str, Path],
        filename: str,
        monitor: str,
        mode: str,
        every_n_epochs: int = 1,
        trial_number: Optional[int] = None,
        verbose: bool = True,
    ):
        super(CheckpointCallback, self).__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            every_n_epochs=every_n_epochs,
            verbose=verbose,
        )
        self.trial_number = trial_number

    def _save_top_k_checkpoints(self, trainer, pl_module, metrics):
        current = metrics.get(self.monitor)
        epoch = metrics.get("epoch")
        step = metrics.get("step")

        if self.check_monitor_top_k(trainer, current):
            self._update_best_and_save(
                current, epoch, step, trainer, pl_module, metrics
            )
        elif self.verbose:
            trial_txt = (
                f"Trial {self.trial_number:d}, "
                if self.trial_number is not None
                else ""
            )
            pl.utilities.rank_zero_info(
                f"{trial_txt}Epoch {epoch:d}, step {step:d}: {self.monitor} ({current:.2f}) was not in " +
                f"top {self.save_top_k} (best: {self.best_model_score:0.2f})"
            )

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        epoch: int,
        step: int,
        trainer,
        pl_module,
        ckpt_name_metrics,
    ):
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

            # do not save nan, replace with +/- inf
            if isinstance(current, torch.Tensor) and torch.isnan(current):
                current = torch.tensor(float('inf' if self.mode == "min" else '-inf'))

        filepath = self._get_metric_interpolated_filepath_name(
            ckpt_name_metrics, epoch, step, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(
                self.best_k_models, key=self.best_k_models.get
            )
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            trial_txt = (
                f"Trial {self.trial_number:d}, "
                if self.trial_number is not None
                else ""
            )
            pl.utilities.rank_zero_info(
                f"{trial_txt}Epoch {epoch:d}, step {step:d}: {self.monitor} reached {current:0.2f}, saving model"
            )
        self._save_model(filepath, trainer, pl_module)

        if del_filepath is not None and filepath != del_filepath:
            self._del_model(del_filepath)


class EarlyStoppingCallback(pl.callbacks.EarlyStopping):
    """
    PyTorch Lightning callback to preemptively stop the model if it has not shown recent improvement.
    This is only slightly updated from pl.callbacks.EarlyStopping to include a print statement and
    to skip the callback at the end of each training epoch.

    :param str monitor: Metric to monitor for improvements. Most likely "val-loss" for the Payne.
    :param str mode: "min" or "max" depending on metric. Most likely "min" if monitor = "val-loss".
    :param int patience: Number of epochs of no improvement after which to stop the training.
    :param bool verbose: If true, prints a message when early stopping is invoked. Default = True.
    """

    def __init__(
        self, monitor: str, mode: str, patience: int, verbose: bool = True
    ) -> None:
        super(EarlyStoppingCallback, self).__init__(
            monitor=monitor, mode=mode, patience=patience, verbose=verbose
        )

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def _run_early_stopping_check(self, trainer, pl_module):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run  # disable early_stopping with fast_dev_run
            or not self._validate_condition_metric(
                logs
            )  # short circuit if metric not present
        ):
            return  # short circuit if metric not present

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        #if current is not None:
        #    if isinstance(current, pl.metrics.metric.Metric):
        #        current = current.compute()
        #    elif isinstance(current, numbers.Number):
        #        current = torch.tensor(
        #            current, device=pl_module.device, dtype=torch.float
        #        )
        #
        #if trainer.use_tpu and pl.utilities.TPU_AVAILABLE:
        #    current = current.cpu()

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                print("Early Stopping!")
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_boolean_decision(trainer.should_stop)


class PruningCallback(pl.callbacks.early_stopping.EarlyStopping):
    """
    PyTorch Lightning callback to prune unpromising trials.
    The pruning algorithm is set separately when the Optuna study is instantiated.
    This is only slightly updated from optuna.integration.PyTorchLightningPruningCallback to include "mode".

    :param optuna.trial.Trial trial: Optuna trial
    :param str monitor: Metric to monitor for improvements. Most likely "val-loss" for the Payne.
    :param str mode: "min" or "max" depending on metric. Most likely "min" if monitor = "val-loss".
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str, mode: str) -> None:
        super(PruningCallback, self).__init__(monitor=monitor, mode=mode)
        self._trial = trial

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
