from base_options import BaseOptions

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--mode', type=str, default='test')
        parser.add_argument('--model_load_path', type=str, default='checkpoints', help='dir for model saving')
        parser.add_argument('--save_path', type=str, default='', help='result saving path')

        return parser

