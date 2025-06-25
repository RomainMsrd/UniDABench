import sys

import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()
       
torch.set_default_dtype(torch.float32)

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super().__init__(args)

        self.results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        if self.uniDA:
            self.results_columns.append('H-score')
            self.results_columns.append("acc_c")
            self.results_columns.append("acc_p")
            self.results_columns.append("acc_mix")
        self.risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]
        self.log_into_wandb = args.log_wandb


    def fit(self):
        if self.log_into_wandb:
            wandb.init(project=f"EUSIPCO_{self.backbone}", name=f"{self.dataset}_{self.da_method}_{self.backbone}", entity="oddteam")
        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)

        # table with risks
        table_risks = pd.DataFrame(columns=self.risks_columns)


        # Trainer
        for sc_id, (src_id, trg_id) in enumerate(self.dataset_configs.scenarios):
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                src_id, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id, sc_id)
                t = torch.cuda.get_device_properties(0).total_memory
                #print(f"Memory usage : {torch.cuda.memory_allocated()/1024/1024/1024:0.3f} GB")
                print(f"Memory usage : {torch.cuda.memory_reserved()/t*100:0.3f} % ")
                
                # initiate the domain adaptation algorithm
                self.initialize_algorithm()

                # Train the domain adaptation algorithm
                self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
                save_dir = os.path.join(self.home_path, self.scenario_log_dir)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.algorithm, os.path.join(save_dir, "last.pt"))

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, self.last_model, self.best_model)
                #self.algorithm.network = self.best_model

                #visualize latent space
                sc = f"{src_id} --> {trg_id}"
                #self.latent_space_tsne(self.home_path, self.scenario_log_dir, sc)
                #self.latent_space_tsne_preds(self.home_path, self.scenario_log_dir, sc)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)
                handlers = self.logger.handlers[:]
                for handler in handlers:
                    self.logger.removeHandler(handler)
                    handler.close()
                self.save_tables_to_file(table_results, 'results_temp')
                self.save_tables_to_file(table_risks, 'risks_temp')

            table_results2 = self.average_run_rusults(table_results)
            table_results2 = self.add_mean_std_table(table_results2, self.results_columns)
            self.save_tables_to_file(table_results2, 'results')
        # Calculate and append mean and std to tables

        table_results = self.average_run_rusults(table_results)
        table_risks = self.average_run_rusults(table_risks)

        table_results = self.add_mean_std_table(table_results, self.results_columns)
        table_risks = self.add_mean_std_table(table_risks, self.risks_columns)

        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

        if self.log_into_wandb:
            table_results = wandb.Table(dataframe=table_results)
            table_risks = wandb.Table(dataframe=table_risks)
            wandb.log({'results': table_results})
            wandb.log({'risks': table_risks})
            wandb.log({'hparams': wandb.Table(
                dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
                allow_mixed_types=True)})
            wandb.finish()

    def test(self):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for sc_id, (src_id, trg_id) in enumerate(self.dataset_configs.scenarios):
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, f"{src_id}_to_{trg_id}_run_{run_id}")
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id, sc_id)

                # Build model
                self.initialize_algorithm()

                # Load checkpoint
                print(self.scenario_log_dir)
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

                scenario = f"{src_id}_to_{trg_id}"

                # --- Last model ---

                self.algorithm = last_chk

                #self.algorithm.network.load_state_dict(last_chk)
                self.evaluate(self.trg_test_dl)
                last_metrics = self.calculate_metrics()
                last_results = self.append_results_to_tables(last_results, scenario, run_id, last_metrics)

                # --- Best model ---
                # self.algorithm.network.load_state_dict(best_chk)
                # self.evaluate(self.trg_test_dl)
                # best_metrics = self.calculate_metrics()
                # best_results = self.append_results_to_tables(best_results, scenario, run_id, best_metrics)

        # Compute mean/std for all relevant columns
        last_results_avg = self.average_run_rusults(last_results)
        #best_results_avg = self.average_run_rusults(best_results)

        last_results_avg = self.add_mean_std_table(last_results_avg, self.results_columns)
        #best_results_avg = self.add_mean_std_table(best_results_avg, self.results_columns)

        # Save tables to file
        self.save_tables_to_file(last_results_avg, 'last_results')
        #self.save_tables_to_file(best_results_avg, 'best_results')

        # Print summary (averaging all metrics over all scenarios and runs)
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
        #summary_best = {metric: np.mean(best_results[metric]) for metric in self.results_columns[2:]}

        for summary_name, summary in [('Last', summary_last)]:#, ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')



