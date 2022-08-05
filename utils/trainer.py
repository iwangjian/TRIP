# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from torch.optim.lr_scheduler import LambdaLR
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler


class Trainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,
            model, 
            train_loader, 
            dev_loader, 
            log_dir, 
            log_steps, 
            validate_steps, 
            num_epochs, 
            lr, 
            warm_up_ratio=0.1, 
            weight_decay=0.01, 
            max_grad_norm=1.0,
            gradient_accumulation_steps=0
        ):

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_ratio = warm_up_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warm_up_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Evaluate every {} batches.".format(self.validate_steps))

        best_model_store_path = os.path.join(self.log_dir, "best_model.bin")
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(tqdm(self.train_loader)):
                self.model.train()
                model_output = self.model(inputs)
                loss = model_output["loss"]
                if self.gradient_accumulation_steps > 0:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.gradient_accumulation_steps > 0:
                    if batch_step > 0 and batch_step % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if batch_step > 0 and batch_step % self.log_steps == 0:
                    log_key = "Batch Step: {}\tloss: {:.3f}"
                    log_value = (batch_step, loss.item(),)
                    for k in model_output.keys():
                        if k != "loss" and "loss" in k:    # show other `xxx_loss` if possible
                            log_key += "\t%s: {:.3f}" % k
                            log_value += (model_output[k].item(),)
                    logging.info(log_key.format(*log_value))
                
                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    logging.info("Evaluation Acc: {:.3f} loss: {:.3f}".format(
                        predicts_dict["avg_acc"], predicts_dict["avg_loss"])
                    )
                    if predicts_dict["avg_acc"] > self.best_metric:
                        self.best_metric = predicts_dict["avg_acc"]
                        logging.info("Epoch {} Batch Step {} -- Best Acc: {:.3f} -- loss: {:.3f}".format(
                            epoch + 1, batch_step, self.best_metric, predicts_dict['avg_loss'])
                        )
                        torch.save(self.model, best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
            
            predicts_dict = self.evaluate(loader=self.dev_loader)
            
            if predicts_dict["avg_acc"] > self.best_metric:
                self.best_metric = predicts_dict["avg_acc"]
                logging.info("Epoch {} Best Avg Acc: {:.3f} -- loss: {:.3f}".format(
                    epoch, self.best_metric, predicts_dict['avg_loss'])
                )
                torch.save(self.model, best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
            
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(self.log_dir, "model_epoch_%d.bin" % (epoch + 1))
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def evaluate(self, loader):
        self.model.eval()
        
        accs = []
        loss = []
        for inputs in tqdm(loader):
            output = self.model(inputs)
            accs.append(output["acc"])
            if self.gradient_accumulation_steps > 0:
                loss.append(float(output["loss"]) / self.gradient_accumulation_steps)
            else:
                loss.append(float(output["loss"]))
        avg_acc = np.mean(accs)
        avg_loss = np.mean(loss)
        
        return_dict = {
            "avg_acc": avg_acc,
            "avg_loss": avg_loss
        }
        return return_dict


class IgniteTrainer(object):
    
    def __init__(self,
            model, 
            train_loader, 
            dev_loader, 
            args
        ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        # parse args
        self.log_dir = args.log_dir
        self.validate_steps = args.validate_steps
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': self.lr}], lr=self.lr, correct_bias=True)
            
        if args.scheduler == "noam":
            # noam decrease the learning rate
            noam_lambda = lambda step: (
                    args.hidden_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
            noam_scheduler = LambdaLR(self.optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
            self.scheduler = LRScheduler(noam_scheduler)
        else:
            # linear decrease the learning rate
            self.scheduler = PiecewiseLinear(self.optimizer, "lr", [(0, self.lr), (self.num_epochs * len(train_loader), 0.0)])

    def update(self, engine, batch):
        self.model.train()
        model_output = self.model(batch)
        loss = model_output["loss"]
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if engine.state.iteration % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return (loss.item(), self.optimizer.param_groups[0]['lr'])

    def inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(batch)
            lm_loss = model_output["lm_loss"]
            acc = model_output["acc"]
            return (lm_loss.item(), acc)

    def run(self):
        trainer = Engine(self.update)
        evaluator = Engine(self.inference)

        # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(self.dev_loader))
        if self.num_epochs < 1:
            trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(self.dev_loader))

        # Evaluation during training
        @trainer.on(Events.ITERATION_STARTED)
        def log_iterations(engine):
            if engine.state.iteration % self.validate_steps == 0:
                evaluator.run(self.dev_loader)
        
        trainer.add_event_handler(Events.ITERATION_STARTED, self.scheduler)

        # Prepare metrics
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss") # update() -> loss
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")   # update() -> lr
        metrics = {"avg_nll": RunningAverage(output_transform=lambda x: x[0])}  # inference() -> lm_loss
        metrics["avg_ppl"] = MetricsLambda(np.math.exp, metrics["avg_nll"])
        metrics["avg_acc"] = RunningAverage(output_transform=lambda x: x[1])    # inference() -> acc
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        # On the main process: add progress bar, tensorboard, checkpoints
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: {}".format(evaluator.state.metrics)))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        tb_logger = TensorboardLogger(log_dir=self.log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                        event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(self.optimizer),
                        event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                        another_engine=trainer),
                        event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        
        # Save model after evaluation
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'model': self.model})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'model': self.model})

        # Run training
        trainer.run(self.train_loader, max_epochs=self.num_epochs)

        # On the main process: save the last checkpoint
        best_model_store_path = os.path.join(tb_logger.writer.logdir, "best_model.bin")
        torch.save(self.model, best_model_store_path)
        logging.info("Saved to [%s]" % best_model_store_path)

        tb_logger.close()