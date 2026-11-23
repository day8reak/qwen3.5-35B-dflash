"""Printable shell commands for the opt-in attention-step trace."""
import os
import shlex
import sys


def note(message):
    if os.environ.get('QWEN35_TRACE_COMMANDS') == '1':
        print(message, file=sys.stderr, flush=True)


def subprocess_event(event, args):
    # CPython emits this before launch, including for subprocesses whose output
    # goes to a compiler log. Do not print the child's environment.
    if event == 'subprocess.Popen':
        _, argv, cwd, _ = args
        if isinstance(argv, (str, bytes)):
            argv = [argv]
        cmd = shlex.join(os.fsdecode(arg) for arg in argv)
        where = shlex.quote(os.fsdecode(cwd) if cwd is not None else os.getcwd())
        print(f'[host] (cd {where} && {cmd})', file=sys.stderr, flush=True)
