import sys


def show_progress(epoch, batch, batch_total, **kwargs):
    message = f'\r{epoch} epoch: [{batch}/{batch_total}'
    for key, item in kwargs.items():
        message += f', {key}: {item}'
    sys.stdout.write(message+']')
    sys.stdout.flush()
