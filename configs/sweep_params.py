import numpy as np
sweep_train_hparams = {
        'num_epochs':    {'values': [20]},
        'num_epochs_pr': {'values': [20]},
        'batch_size':    {'values': [32, 64]},
        'weight_decay':  {'values': [1e-4]}
}

sweep_alg_hparams = {
        'UDA': {
            'learning_rate': {'values': [1e-1, 1e-2, 1e-3, 5e-4, 1e-4]},
            'src_cls_loss_wt':  {'values': np.arange(0.05,5+0.05,0.05).tolist()},
            'domain_loss_wt':   {'values': np.arange(0.05,5+0.05,0.05).tolist()},
            'w0': {'values': np.arange(0.05,1+0.05,0.05).tolist()},
        },
        'DANCE': {
            'learning_rate': {'values': [1e-1, 1e-2, 1e-3, 5e-4, 1e-4]},
            'eta': {'values': np.arange(0.05,5+0.05,0.05).tolist()},
            'margin': {'values': np.arange(0.05,1+0.05,0.05).tolist()},
        },
        'OVANet': {
            'learning_rate': {'values': [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
        },
        'UniOT': {
                'learning_rate': {'values': [1e-1, 1e-2, 1e-3, 1e-4]},
                'gamma': {'values': np.arange(0.05,1+0.05,0.05).tolist()},
                'mu': {'values': np.arange(0.05,1+0.05,0.05).tolist()},
                'lam': {'values': np.arange(0.05, 5+0.05, 0.05).tolist()},
                "temp": {'values': [0.05, 0.1, 1]},
                "K": {'values': [5, 10, 15, 20]},
                "MQ_size": {'values': [64, 128, 1000, 2000]}
            },
        'PPOT': {
                'learning_rate': {'values': [1e-1, 1e-2, 1e-3, 1e-4]},
                'tau': {'values': np.arange(0.05,0.9+0.05,0.05).tolist()},
                "tau1": {'values': np.arange(0.05,0.9+0.05,0.05).tolist()},
                "tau2": {'values': np.arange(0.05,2+0.05,0.05).tolist()},
                "alpha": {'values': [0.01, 0.001]},
                "beta": {'values': [0.01, 0.001]},
                "reg": {'values': [0.01, 0.1]},
                "ot" : {'values': np.arange(0.05,5+0.05,0.05).tolist()},
                "p_entropy": {'values': np.arange(0.05,2+0.05,0.05).tolist()},
                "n_entropy": {'values': np.arange(0.05,2+0.05,0.05).tolist()},
                "neg" : {'values': [0.2, 0.25, 0.3, 0.4, 0.5]},
                "thresh": {'values': np.arange(0.05,1+0.05,0.05).tolist()},
            },
            'UniJDOT_THR': {
                'learning_rate': {'values': [0.001]},
                'lamb': {'values': [0.4]},
                "alpha": {'values': [4.65]},
                "src_weight": {'values': [1.5]},
                "K": {'values': [10]},
                "n_batch": {'values': [15]},
                "trg_mem_size": {'values': [64]},
                "joint_decision": {'values': [True]},
                "threshold_method": {'values': ["threshold_yen"]},
                "threshold" : {'values': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 0.99]}
            },
        'UniJDOT': {
                'learning_rate': {'values': [1e-2, 1e-3, 5e-4, 1e-4]},
                'lamb': {'values': np.arange(0.05,5+0.05,0.05).tolist()},
                "alpha": {'values': np.arange(0.05,5+0.05,0.05).tolist()},
                "src_weight": {'values': np.arange(0.05,5+0.05,0.05).tolist()},
                "K": {'values': [5, 10, 15, 20]},
                "n_batch": {'values': [5, 10, 15, 20, 30]},
                "trg_mem_size": {'values': [32, 64, 128]},
                "joint_decision": {'values': [True]},
                "threshold_method": {'values': ["threshold_yen"]}
            }


}