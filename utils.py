import json
import sys
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


def show_progress(epoch, batch, batch_total, **kwargs):
    message = f'\r{epoch} epoch: [{batch}/{batch_total}'
    for key, item in kwargs.items():
        message += f', {key}: {item}'
    sys.stdout.write(message+']')
    sys.stdout.flush()


def save_config(config, filename = None):
    if not isinstance(config, (dict, OrderedDict)):
        raise TypeError('arg config must be a dict or OrderedDict')
    config = OrderedDict(config)

    if filename is None:
        filename = 'config_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '.json'

    with open(filename, 'w') as f:
        json.dump(config, f, indent = 4)
    print(f'Given config has been successfully saved to {filename}.')

    
class ExperimentSaver:
    def __init__(self, logdir = None, parse_args = None):
        if logdir is None:
            self.logdir = Path('logs_' + datetime.now().strftime('%Y-%m-%d-%H-%M'))
        else:
            self.logdir = Path(logdir)
        if not self.logdir.exists():
            self.logdir.mkdir()
            
        self.save_list = []

        if parse_args is not None:
            save_config(vars(parse_args), 'config.json')
            self.append('config.json')

    def append(self, file_or_dir_names):
        if not isinstance(file_or_dir_names, list):
            file_or_dir_names = [file_or_dir_names]
        for name in file_or_dir_names:
            self.save_list.append(Path(name))
        
    def save(self):
        for path in self.save_list:
            path.rename(self.logdir/path)
