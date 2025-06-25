from trainers.train import Trainer

import argparse
import time
parser = argparse.ArgumentParser()
import pandas

if __name__ == "__main__":
    start_time = time.time()

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')

    # ========  Experiments Name ================
    #parser.add_argument('--save_dir',               default='logs/docker_final_repro2/CNN',         type=str, help='Directory containing all experiments')
    parser.add_argument('--save_dir', default='logs/clean', type=str, help='Directory containing all experiments')

    parser.add_argument('--exp_name', default='EXP1', type=str, help='experiment name')
    #parser.add_argument('--exp_name',               default='NoParam_SRC2_FNO',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='UniJDOT',               type=str, help='UDA (UAN), OVANet, DANCE, PPOT, UniOT, UniJDOT')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default='./data/',                  type=str, help='Path containing dataset')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=10,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default="cuda",                   type=str, help='cpu or cuda')
    parser.add_argument("--uniDA",                  action='store_false', help='Different Label Set between Src and Trg Domain ?')
    parser.add_argument("--generate-private", action='store_false', help='uniDA should be True too ?')
    parser.add_argument("--log_wandb", action='store_true', help='log results using wandb')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()

    gap = (time.time() - start_time)
    print("--- %s seconds ---" % (gap))