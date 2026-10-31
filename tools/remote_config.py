"""Consume shared explicit connection arguments for retained remote launch scripts."""
import argparse
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--ssh-host')
parser.add_argument('--remote-root')
parser.add_argument('--remote-lock')
parser.add_argument('--training-root')
args, remaining = parser.parse_known_args()
sys.argv[1:] = remaining


class Required:
    def __init__(self, flag):
        self.flag = flag

    def __str__(self):
        raise ValueError(f'Pass {self.flag} explicitly; private machine defaults were removed')

    def __fspath__(self):
        return str(self)

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)


SSH_HOST = args.ssh_host or Required('--ssh-host')
REMOTE_ROOT = args.remote_root or Required('--remote-root')
LOCK = args.remote_lock or Required('--remote-lock')
TRAIN_ROOT = args.training_root or Required('--training-root')
