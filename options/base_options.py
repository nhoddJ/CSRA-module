#some codes in this file are from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import argparse

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing.
    #It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # paras1
        parser.add_argument('--use_gpu', type=bool, default=False, help='gpu or cpu')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids')
        parser.add_argument('--is_debug', type=bool, default=True, help='whether the mode for debugging or not')
        parser.add_argument('--n_class', type=int, default=11, help='fianl class number for classification')
        parser.add_argument('--ap_size', type=int, default=49, help='size of average pooling after final convolutional layer')
        parser.add_argument('--global_transform', type=tuple, default=None, help='global transform methods')
        parser.add_argument('--global_transform_xval', type=tuple, default=None, help='global transform methods not usde in validation or test mode')
        parser.add_argument('--global_transform_para', type=tuple, default=None, help='parameters of global transform methods')
        parser.set_defaults(global_transform=(['mask','resize','div']))
        parser.set_defaults(global_transform_xval=([]))
        parser.set_defaults(global_transform_para=(['./data/dataset/region_mask',(224,224), 255]))

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        """
        print('\033[1;35m' + '#' * 20 + ' options ' + '#' * 20 + '\033[0m')
        message = ''
        #message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



