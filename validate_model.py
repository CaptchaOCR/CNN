### Fit params

DATA_DIR = "./Captchas"

TRAINED_NN_PATH = './trained_ResNetWrapper_weights.pth'

BATCH_SIZE = 64
NUM_WORKERS = 4


###
from pathlib import Path

import torch
from Infrastructure import CaptchaDataset, ResNetWrapper
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.INFO)


def run():

    # Load in NN from weights
    net = ResNetWrapper.instantiate_with_no_weights()
    net.load_state_dict(torch.load(TRAINED_NN_PATH, 
                                   map_location=torch.device('cpu')))


    # Locate files
    file_locations = [file for file in Path(DATA_DIR).glob('*')]
    logging.info( f'Identified {len(file_locations)} images.' )


    logging.debug('Loading images.')
    # Instantiate testloader
    test_files = file_locations
    testloader = DataLoader(CaptchaDataset.from_dir(test_files), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Run validation
    logging.debug('Starting validation.')
    label_accuracy, char_accuracy = net.validate(testloader)


    return


if __name__ == '__main__':
    run()
