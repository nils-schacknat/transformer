import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Transformer
from torch.utils.tensorboard import SummaryWriter
from datapipe import TranslationDatapipe
from torchtext.data import metrics
import datetime
from util import strip_tokens_after_eos


class Trainer:
    def __init__(
        self,
        model: Transformer,
        train_pipe: TranslationDatapipe,
        val_pipe: TranslationDatapipe,
        device: str,
        log_dir: str,
        label_smoothing: float,
        warmup_steps: int,
        run_identifier: Optional[str] = None
    ) -> None:
        """
        Trainer class for training and validating a PyTorch transformer model.

        Args:
            model (Transformer): The model to train.
            train_pipe (TranslationDatapipe): DataLoader for training data.
            val_pipe (TranslationDatapipe): DataLoader for validation data.
            device (str): Device to use for training ("cpu" or "cuda").
            log_dir (str): Directory path for logging with TensorBoard.
            label_smoothing (float): Smoothing factor for label smoothing in the loss calculation.
            warmup_steps (int): Number of warm-up steps for the learning rate scheduler.
            run_identifier (Optional[str]): Identifier for the current run (default: None).
        """
        self.model = model
        self.train_pipe = train_pipe
        self.val_pipe = val_pipe
        self.device = device

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=train_pipe.target_tokenizer.pad_id(), label_smoothing=label_smoothing
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.model.model_dim**(-.5), betas=(0.9, 0.98), eps=1e-9
        )

        def lrate(step_num: int) -> float:
            step_num += 1
            return min(step_num**(-.5), step_num * warmup_steps**(-1.5))
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lrate)

        self.metrics = {
            "train_loss": [],
            "bleu_score": [],
        }
        self.current_bleu_score = torch.nan
        if run_identifier is None:
            now = datetime.datetime.now()
            run_identifier = now.strftime("%Y-%m-%d-%H%M")

        self.run_dir = os.path.join(log_dir, run_identifier)
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = SummaryWriter(self.run_dir)
        self.training_step = 1

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Performs a single training step.

        Args:
            inputs (torch.Tensor): Input sequences.
            targets (torch.Tensor): Target sequences.
        """
        self.model.train()

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        src_key_padding_mask = inputs == self.train_pipe.source_tokenizer.pad_id()

        enocded_inputs = self.model.encode_source(
            src_sequence=inputs, src_key_padding_mask=src_key_padding_mask
        )
        logits = self.model(
            src_encoding=enocded_inputs,
            tgt_sequence=targets[..., :-1],
            src_key_padding_mask=src_key_padding_mask,
        )
        torch.autograd.set_detect_anomaly(True)
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), targets[..., 1:].reshape(-1))
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        self.training_step += 1

        # Metrics
        train_loss = loss.item()

        self.metrics["train_loss"].append(train_loss)
        self.writer.add_scalar("loss", train_loss, self.training_step)
        self.writer.flush()

    def validate(self, num_validation_steps: int) -> None:
        """
        Validate by computing the BLEU score on the validation set.
        """
        self.model.eval()
        bleu_score_list = []

        with torch.no_grad():
            pbar = tqdm(total=num_validation_steps, ncols=120, desc="Evaluating", leave=False)
            pbar.update(1)
            for i, (inputs, targets) in enumerate(self.val_pipe):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                src_key_padding_mask = inputs == self.train_pipe.source_tokenizer.pad_id()
                outputs = self.model.generate(src_sequence=inputs, src_key_padding_mask=src_key_padding_mask, max_len=2)

                targets_processed = strip_tokens_after_eos(targets, val_pipe.target_tokenizer.eos_id())
                targets_processed = [val_pipe.target_tokenizer.id_to_piece(s) for s in targets_processed]
                outputs_processed = strip_tokens_after_eos(outputs, val_pipe.target_tokenizer.eos_id())
                outputs_processed = [val_pipe.target_tokenizer.id_to_piece(s) for s in outputs_processed]

                # Calculate the BLEU score
                bleu_score = metrics.bleu_score(outputs_processed, targets_processed)
                self.current_bleu_score = bleu_score

                bleu_score_list.append(bleu_score)
                pbar.update(1)
                if i == num_validation_steps - 1:
                    break

        bleu_score = np.mean(bleu_score_list)
        self.metrics["bleu_score"].append((bleu_score, self.training_step))

        self.writer.add_scalar("bleu", bleu_score, self.training_step)

    def train(self, num_training_steps: int, num_validation_steps: int, num_validation_runs: int) -> None:
        """
        Trains the model for the specified number of training steps.

        Args:
            num_training_steps (int): Number of training steps/batches.
            num_validation_steps (int): Number of validation steps/batches.
            num_validation_runs (int): Number of validation runs during training.
        """
        pbar = tqdm(total=num_training_steps, ncols=120, desc="Training")
        pbar.update(1)
        while self.training_step < num_training_steps:
            for inputs, targets in self.train_pipe:

                if self.training_step > num_training_steps:
                    break

                self.train_step(inputs=inputs, targets=targets)
                pbar.set_postfix({"loss": self.metrics["train_loss"][-1], "bleu_score": self.current_bleu_score})
                pbar.update(1)

                if self.training_step % (num_training_steps // num_validation_runs) == 0:
                    pbar.set_postfix({"Validating": "..."})
                    self.validate(num_validation_steps)
                    self.save_checkpoint(os.path.join(self.run_dir, f"checkpoint_{self.training_step}"))

        self.writer.close()

    def save_checkpoint(self, filepath: str) -> None:
        """
        Saves the current model checkpoint.

        Args:
            filepath (str): File path to save the checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
        }
        torch.save(checkpoint, filepath)


if __name__ == "__main__":
    from util import load_config
    import os
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Load config
    config = load_config("config.yaml")

    # Load the datapipe
    datapipe = TranslationDatapipe(**config["datapipe"], **config["tokenizer"])
    train_pipe, val_pipe = datapipe.random_split(config["p_test"])

    # Create the model
    transformer = Transformer(
        **datapipe.tokenizer_params, **config["transformer_params"]
    )

    trainer = Trainer(
        model=transformer,
        train_pipe=train_pipe,
        val_pipe=val_pipe,
        device=device,
        **config["training"]
    )

    os.system(f"tensorboard --logdir={trainer.run_dir} &")
    time.sleep(5)
    trainer.train(**config["num_steps"])
