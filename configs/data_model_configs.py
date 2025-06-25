def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]
        


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
                          ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]

        # Uncomment for HP-search
        #self.scenarios = [('10', '4'), ('8', '19'), ('4', '8')]

        self.private_classes = [{'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]}]


        self.generate_private = None  # Is initialized in Trainer given main arguments
        self.da_method = None  # Is initialized in Trainer given main arguments

        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.src_balanced = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # FNO
        self.isFNO = False
        self.fourier_modes = 64 #300


        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # TimesNet
        self.isTimesNet = False
        self.e_layers = 1
        self.d_model = 1
        self.embed = "timeF"
        self.freq = "h"
        self.enc_in = self.input_channels
        self.pred_len = 0
        self.d_ff = 32
        self.num_kernels = 6
        self.top_k = 5

        # TSLANet
        self.patch_size = 8
        self.emb_dim = 128
        self.depth = 2
        self.masking_ratio = 0.4
        self.ICB = True
        self.ASB = True
        self.adaptive_filter = True

class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()
        self.src_balanced = True
        self.sequence_len = 128
        self.scenarios = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"),
                          ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")]
        # Uncomment for HP-search
        #self.scenarios = [('3', '4'), ('6', '7'), ('7', '8')]

        self.private_classes = [{'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]}]
        #self.private_classes = [{'src': [0, 5], 'trg': []}]
        self.generate_private = None #Is initialized in Trainer given main arguments
        #self.scenarios = [("0", "6")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.temp = 0.05

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        #FNO
        self.isFNO = False
        self.fourier_modes = 64

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        # TimesNet
        self.isTimesNet = False
        self.e_layers = 1
        self.d_model = 1
        self.embed = "timeF"
        self.freq = "h"
        self.enc_in = self.input_channels
        self.pred_len = 0
        self.d_ff = 32
        self.num_kernels = 6
        self.top_k = 5

        # TSLANet
        self.patch_size = 8
        self.emb_dim = 128
        self.depth = 2
        self.masking_ratio = 0.4
        self.ICB = True
        self.ASB = True
        self.adaptive_filter = True

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("6", "23"), ("9", "18"), ("12", "16"), ("24", "8"), ("30", "20"),
                          ("13", "3"), ("15", "21"), ("1", "14"), ("17", "29"), ("22", "4")]

        # Uncomment for HP-search
        #self.scenarios = [("2", "11"), ("18", "27"), ("20", "5"), ("7", "13"), ("28", "27")] #HP Research
        self.private_classes = [{'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]}]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.src_balanced = True
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.temp = 0.05

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 5

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # FNO
        self.isFNO = False
        self.fourier_modes = 64

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

        # TimesNet
        self.isTimesNet = False
        self.e_layers = 1
        self.d_model = 1
        self.embed = "timeF"
        self.freq = "h"
        self.enc_in = self.input_channels
        self.pred_len = 0
        self.d_ff = 32
        self.num_kernels = 6
        self.top_k = 5

        #TSLANet
        self.patch_size = 8
        self.emb_dim = 128
        self.depth = 2
        self.masking_ratio = 0.4
        self.ICB = True
        self.ASB = True
        self.adaptive_filter = True