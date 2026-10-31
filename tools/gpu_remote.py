"""Run a configured worker over SSH without assumptions about a private host."""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='Your configured SSH host')
    parser.add_argument('--python', required=True, help='Remote Python executable path')
    parser.add_argument('--worker', required=True, help='Remote worker script path')
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Worker arguments after --')
    a = parser.parse_args()
    arguments = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    # Windows OpenSSH runs through cmd.exe. subprocess.list2cmdline preserves paths
    # with spaces. Linux remote execution is deliberately outside the tested path.
    command = subprocess.list2cmdline([a.python, a.worker, *arguments])
    subprocess.run(['ssh', a.host, command], check=True)


if __name__ == '__main__':
    main()
