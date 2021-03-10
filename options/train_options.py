
from base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for Adam')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4) for Adam')
        parser.set_defaults(model_save_path='checkpoints')
        parser.set_defaults(epochs=21)
        parser.set_defaults(batch_size=8)

        return parser
