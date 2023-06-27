from math import sqrt
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Transformer
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import metrics
import datetime
from util import strip_tokens_after_eos
import sentencepiece as spm


class Trainer:
    def __init__(
        self,
        model: Transformer,
        source_tokenizer: spm.SentencePieceProcessor,
        target_tokenizer: spm.SentencePieceProcessor,
        train_pipe: IterableWrapper,
        val_pipe: IterableWrapper,
        max_generation_length: int,
        log_dir: str,
        label_smoothing: float,
        warmup_steps: int,
        run_identifier: Optional[str] = None
    ) -> None:
        """
        Trainer class for training and validating a PyTorch transformer model.

        Args:
            model (Transformer): The model to train.
            source_tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer for the source language.
            target_tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer for the target language.
            train_pipe (TranslationDatapipe): DataLoader for training data.
            val_pipe (TranslationDatapipe): DataLoader for validation data.
            max_generation_length (int): The maximum length for sequence generation during validation.
            log_dir (str): Directory path for logging with TensorBoard.
            label_smoothing (float): Smoothing factor for label smoothing in the loss calculation.
            warmup_steps (int): Number of warm-up steps for the learning rate scheduler.
            run_identifier (Optional[str]): Identifier for the current run (default: None).
        """
        self.model = model
        self.train_pipe = train_pipe
        self.val_pipe = val_pipe
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_generation_length = max_generation_length

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.target_tokenizer.pad_id(), label_smoothing=label_smoothing
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1/sqrt(self.model.model_dim), betas=(0.9, 0.98), eps=1e-9
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

        src_key_padding_mask = inputs == self.source_tokenizer.pad_id()

        enocded_inputs = self.model.encode_source(
            src_sequence=inputs
        )
        logits = self.model(
            src_encoding=enocded_inputs,
            tgt_sequence=targets[..., :-1],
            src_key_padding_mask=src_key_padding_mask,
        )
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
        with torch.no_grad():
            pbar = tqdm(total=num_validation_steps, ncols=100, desc="Evaluating", leave=False)
            pbar.update(1)

            reference_corpus = []
            candidate_corpus = []
            for i, (inputs, targets) in enumerate(self.val_pipe):
                outputs = self.model.generate(src_sequence=inputs, max_len=self.max_generation_length)

                targets_processed = strip_tokens_after_eos(targets, self.target_tokenizer.eos_id())
                targets_processed = [[self.target_tokenizer.id_to_piece(s)] for s in targets_processed]
                outputs_processed = strip_tokens_after_eos(outputs, self.target_tokenizer.eos_id())
                outputs_processed = [self.target_tokenizer.id_to_piece(s) for s in outputs_processed]

                reference_corpus.extend(targets_processed)
                candidate_corpus.extend(outputs_processed)

                pbar.update(1)
                if i == num_validation_steps - 1:
                    break

            # Calculate the BLEU score
            bleu_score = metrics.bleu_score(candidate_corpus, reference_corpus)
            self.current_bleu_score = bleu_score

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
        pbar = tqdm(total=num_training_steps, ncols=100, desc="Training")
        pbar.update(self.training_step)
        while self.training_step < num_training_steps:
            for inputs, targets in self.train_pipe:

                if self.training_step > num_training_steps:
                    break

                try:
                    self.train_step(inputs=inputs, targets=targets)
                    pbar.set_postfix({"loss": self.metrics["train_loss"][-1], "bleu_score": self.current_bleu_score})
                    pbar.update(1)

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("Skipping training step, out of memory error!")

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
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "metrics": self.metrics,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.metrics = checkpoint["metrics"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.training_step = len(self.metrics["train_loss"]) + 1


if __name__ == "__main__":
    from translation_datapipe import create_translation_datapipe_train, create_translation_datapipe_val
    from util import load_config, load_tokenizers, get_tokenizer_params
    import os
    import time

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    torch.set_default_device(device)

    # Load config
    config = load_config("config.yaml")

    # Load the tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(**config["tokenizer"])

    # Load the training and validation pipe
    train_pipe = create_translation_datapipe_train(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        **config["datapipe_train"]
    )
    val_pipe = create_translation_datapipe_val(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_generation_length=config["max_generation_length"],
        **config["datapipe_val"]
    )

    # Create the model
    transformer = Transformer(
        **get_tokenizer_params(source_tokenizer, target_tokenizer),
        **config["transformer_params"]
    )

    trainer = Trainer(
        model=transformer,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        train_pipe=train_pipe,
        val_pipe=val_pipe,
        max_generation_length=config["max_generation_length"],
        **config["training"]
    )

    # os.system(f"tensorboard --logdir={trainer.run_dir} &")
    # time.sleep(5)
    trainer.train(**config["num_steps"])
