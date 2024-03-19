import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.experimental import optimizers

import argparse
import copy
import json
import logging
import math
import os
import sys
import pickle
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import grok.metrics as metrics
from grok.data import (
    DEFAULT_DATA_DIR,
    EOS_TOKEN,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import Transformer
from grok.measure import get_sharpness

DEFAULT_LOG_DIR = "logs"

class TrainableTransformer:
    def __init__(self, hparams: Namespace):
        self.hparams = hparams
        self.prepare_data()

        self.transformer = Transformer(
            hparams.n_layers,
            hparams.n_heads,
            hparams.d_model,
            hparams.dropout,
            hparams.max_context_len,
            len(self.train_dataset.tokenizer),
            hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
        )

        self.margin = jnp.array([0])
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        parser.add_argument(
            "--batchsize",
            type=float,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--weight_noise", type=float, default=0.0)
        parser.add_argument("--non_linearity", type=str, default="relu")
        parser.add_argument("--max_context_len", type=int, default=50)

        parser.add_argument("--math_operator", type=str, default="+")
        parser.add_argument(
            "--operand_length",
            type=int,
            help="for list operations, the length of the lists",
        )

        parser.add_argument("--train_data_pct", type=float, default=5)
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--anneal_lr_steps", type=int, default=100000)
        parser.add_argument("--anneal_lr", dest="anneal_lr", action="store_true")
        parser.set_defaults(anneal_lr=False)

        parser.add_argument("--max_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
        parser.add_argument("--noise_factor", type=float, default=0)

        parser.add_argument(
            "--save_activations", dest="save_activations", action="store_true"
        )
        parser.set_defaults(save_activations=False)
        parser.add_argument("--save_outputs", dest="save_outputs", action="store_true")
        parser.set_defaults(save_outputs=False)

        parser.add_argument(
            "--logdir",
            type=str,
            default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir",
            type=str,
            default=DEFAULT_DATA_DIR,
        )

        return parser

    def prepare_data(self):
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,
            operator=self.hparams.math_operator,
            operand_length=self.hparams.operand_length,
            data_dir=self.hparams.datadir,
        )

    def train_dataloader(self):
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.train_dataset,
            device,
            batchsize_hint=self.hparams.batchsize,
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator

    def val_dataloader(self):
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset,
            device,
            batchsize_hint=-1,
        )
        return iterator

    def test_dataloader(self):
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset, device, batchsize_hint=-1
        )
        return iterator

    def _scheduler_lr(self, step):
        max_lr = self.hparams.max_lr
        min_lr = self.hparams.max_lr / 10
        warmup_steps = self.hparams.warmup_steps
        if not self.hparams.anneal_lr:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step <= self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + jnp.cos(jnp.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self):
        optimizer = CustomAdamW(
            self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-8,
            lr=1,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
            weight_decay_form=self.hparams.weight_decay_kind,
        )
        schedulers = [
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=self._scheduler_lr),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def _accuracy(self, y_hat, y):
        y_hat = jnp.argmax(y_hat, axis=-2)
        row_accuracy = jnp.min((y_hat == y), axis=-1)
        accuracy = row_accuracy.astype(jnp.float32) * 100
        return accuracy

    def _step(
        self,
        batch,
        batch_idx,
        train=True,
        reduction="mean",
        grads=False,
    ):
        x = batch["text"]
        y = batch["target"]
        y_hat, attentions, values = self(
            x=x, save_activations=self.hparams.save_activations
        )
        y_hat = y_hat.transpose(-2, -1)

        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        eq_position_t = jnp.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        y_rhs = y[..., eq_position + 1 :]
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        x_lhs = x[..., : eq_position + 1]

        if train:
            coeff = float(batch["target"].shape[0]) / len(self.train_dataset)
        else:
            coeff = float(batch["target"].shape[0]) / len(self.val_dataset)
        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

        acc = self._accuracy(y_hat_rhs, y_rhs)
        if reduction == "mean":
            acc = acc.mean()

        grad_vec = None
        if grads:
            loss.backward()
            for p in self.parameters():
                p.grad.data.div_(batch["text"].shape[0])
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = jnp.concatenate((grad_vec, p.grad.data.view(-1)))
            return loss, grad_vec
        return loss, acc, coeff, x_lhs, y_hat_rhs, attentions, values

    def _save_inputs(self, outputs, ds):
        logdir = self.hparams.logdir + "/inputs/" + ds
        os.makedirs(logdir, exist_ok=True)
        pickle_file = logdir + f"/{ds}.pt"

        x_lhs = jnp.concatenate([x["x_lhs"] for x in outputs])
        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations,
    ):
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = jnp.concatenate(activations[l][h])

        return activations

    def _save_activations(self, outputs, ds):
        output = {}
        if self.hparams.save_outputs:
            y_hat_rhs = jnp.concatenate([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:
            partial_attentions = list([o["partial_attentions"] for o in outputs])
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = list([o["partial_values"] for o in outputs])
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        if self.hparams.save_outputs or self.hparams.save_activations:
            logdir = self.hparams.logdir + "/outputs/" + ds
            os.makedirs(logdir, exist_ok=True)
            pickle_file = logdir + f"/epoch_{self.current_epoch:010}.pt"
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.training_epoch_start_time = time.time()
            self.fwd_time_in_epoch = 0

        start = time.time()
        loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=True
        )
        self.fwd_time_in_epoch += time.time() - start

        schedulers = self.trainer.lr_schedulers[0]
        if self.current_epoch != self.next_train_epoch_to_log:
            return {"loss": loss}
        lr = schedulers["scheduler"].optimizer.param_groups[0]["lr"]
        output = {
            "loss": loss,
            "partial_train_loss": coeff * loss,
            "partial_train_accuracy": coeff * accuracy,
            "learning_rate": jnp.array([lr]),
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def training_epoch_end(self, outputs):
        epoch_is_to_be_logged = self.current_epoch == self.next_train_epoch_to_log
        if epoch_is_to_be_logged:
            self.next_train_epoch_to_log = max(
                int(1.01 * self.next_train_epoch_to_log),
                self.next_train_epoch_to_log + 1,
            )
            with torch.no_grad():
                loss = jnp.stack([x["partial_train_loss"] for x in outputs]).sum()
                perplexity = jnp.exp(loss)
                accuracy = jnp.stack([x["partial_train_accuracy"] for x in outputs]).sum()

            first_lr = outputs[0]["learning_rate"]

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="train")
                self._save_activations(outputs, ds="train")

            logs = {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_perplexity": perplexity,
                "learning_rate": first_lr,
                "len_train_ds": len(self.train_dataset),
                "len_val_ds": len(self.val_dataset),
                "batches_per_epoch": self.batches_per_epoch,
                "time_per_epoch": time.time() - self.training_epoch_start_time,
                "fwd_time_in_epoch": self.fwd_time_in_epoch,
            }
            for k, v in logs.items():
                self.log(k, v)

    def validation_step(self, batch, batch_idx):
        x = batch["text"]
        y = batch["target"]
        y_hat, _, _ = self(x=x, save_activations=self.hparams.save_activations)
        y_hat = y_hat.transpose(-2, -1)

        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        eq_position_t = jnp.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        y_rhs = y[..., eq_position + 1 :]
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        accuracy = self._accuracy(y_hat_rhs, y_rhs)
        loss = F.cross_entropy(y_hat_rhs, y_rhs)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = jnp.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = jnp.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_accuracy)

