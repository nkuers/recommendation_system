import argparse
import json
from pathlib import Path

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from recbole.model.sequential_recommender.ideagru import IDEAGRU


class IDEATwoStageTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        env_params = []
        rec_params = []
        for name, p in self.model.named_parameters():
            if 'env' in name:
                env_params.append(p)
            else:
                rec_params.append(p)
        self.optimizer_rec = self._build_optimizer(params=rec_params)
        self.optimizer_env = self._build_optimizer(params=env_params)

    def _run_one_epoch(self, train_data, mode='pretrain'):
        self.model.train()
        total = 0.0
        for interaction in train_data:
            interaction = interaction.to(self.device)
            if mode == 'pretrain':
                self.optimizer_rec.zero_grad()
                loss = self.model.calculate_loss(interaction)
                if isinstance(loss, tuple):
                    loss = sum(loss)
                loss.backward()
                self.optimizer_rec.step()
                total += float(loss.item())
            else:
                self.optimizer_env.zero_grad()
                env_loss = self.model.calculate_loss_env(interaction)
                if isinstance(env_loss, tuple):
                    env_loss = sum(env_loss)
                env_loss.backward()
                self.optimizer_env.step()

                self.optimizer_rec.zero_grad()
                rec_loss = self.model.calculate_loss_il(interaction)
                if isinstance(rec_loss, tuple):
                    rec_loss = sum(rec_loss)
                rec_loss.backward()
                self.optimizer_rec.step()
                total += float(rec_loss.item())
        return total / max(len(train_data), 1)

    def fit_two_stage(self, train_data, valid_data, pretrain_epochs, adv_epochs):
        best_valid_score = float('-inf')
        best_valid_result = None
        best_state = None

        for _ in range(pretrain_epochs):
            self._run_one_epoch(train_data, mode='pretrain')
            valid_score, valid_result = self._valid_epoch(valid_data)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_valid_result = valid_result
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        for _ in range(adv_epochs):
            self._run_one_epoch(train_data, mode='adv')
            valid_score, valid_result = self._valid_epoch(valid_data)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_valid_result = valid_result
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_valid_score, best_valid_result

    def fit_two_stage_fair(self, train_data, valid_data, total_epochs, pretrain_epochs, early_stop):
        best_valid_score = float("-inf")
        best_valid_result = None
        best_state = None
        best_epoch = -1
        cur_step = 0
        stop_flag = False

        for epoch_idx in range(total_epochs):
            mode = "pretrain" if epoch_idx < pretrain_epochs else "adv"
            self._run_one_epoch(train_data, mode=mode)
            valid_score, valid_result = self._valid_epoch(valid_data)

            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_valid_result = valid_result
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch_idx
                cur_step = 0
            else:
                cur_step += 1
                if cur_step >= early_stop:
                    stop_flag = True
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_valid_score, best_valid_result, best_epoch, stop_flag


def main():
    parser = argparse.ArgumentParser(description='Run IDEAGRU with two-stage adversarial training.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config_files', required=True, help='comma-separated yaml list')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', required=True)
    parser.add_argument('--paper-mode', action='store_true', help='Use fixed two-stage schedule without early-stop.')
    args = parser.parse_args()

    config_files = [x.strip() for x in args.config_files.split(',') if x.strip()]
    config_dict = {
        'seed': args.seed,
        'device': args.device,
        'epochs': args.epochs,
        'neg_sampling': None,
        'train_neg_sample_args': None,
    }

    config = Config(model=IDEAGRU, dataset=args.dataset, config_file_list=config_files, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    model = IDEAGRU(config, train_data.dataset).to(config['device'])

    total_epochs = int(config['epochs'])
    pretrain_epochs = max(1, total_epochs // 2)
    adv_epochs = max(1, total_epochs - pretrain_epochs)
    early_stop = int(config['early_stop']) if 'early_stop' in config else 10

    trainer = IDEATwoStageTrainer(config, model)
    if args.paper_mode:
        best_valid_score, best_valid_result = trainer.fit_two_stage(train_data, valid_data, pretrain_epochs, adv_epochs)
        best_epoch = total_epochs - 1
        stop_flag = False
        training_mode = 'paper'
    else:
        best_valid_score, best_valid_result, best_epoch, stop_flag = trainer.fit_two_stage_fair(
            train_data=train_data,
            valid_data=valid_data,
            total_epochs=total_epochs,
            pretrain_epochs=pretrain_epochs,
            early_stop=early_stop,
        )
        training_mode = 'fair'
    test_result = trainer.evaluate(test_data, load_best_model=False)

    out = {
        'best_valid_score': float(best_valid_score),
        'best_valid_result': dict(best_valid_result),
        'test_result': dict(test_result),
        'pretrain_epochs': pretrain_epochs,
        'adv_epochs': adv_epochs,
        'early_stop': early_stop,
        'best_epoch': best_epoch,
        'stopped_early': stop_flag,
        'training_mode': training_mode,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
