
### Fit params

DATA_DIR = "./Captchas"

BATCH_SIZE = 256
TEST_SIZE = .2
NUM_WORKERS = 8#4 

EPOCHS = 20
LEARNING_RATE = 1e-3

SAVE_PATH = "./ResNetWrapper_weights.pth"


from pathlib import Path
import logging
logging.basicConfig(level=logging.DEBUG)
from sys import getsizeof

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from Infrastructure import CaptchaDataset, unique_characters, captcha_length, ResNetWrapper#, CNN



CUDA = False
if torch.cuda.is_available():
    CUDA = True
logging.warning(f'CUDA available: {torch.cuda.is_available()}')

def run():
    # Locate files
    file_locations = []
    captcha_names = []
    for file in Path(DATA_DIR).glob('*'):
        file_locations.append(str(file))
        captcha_names.append(file.stem)
    logging.info( f'Identified {len(file_locations)} images.' )

    # Split training/test data
    train_files, test_files = train_test_split(file_locations, test_size = TEST_SIZE)
    print(f'Split dataset into 80:20 train/test of sizes {len(train_files)},{len(test_files)}.')

    # Instantiate dataset
    trainset = CaptchaDataset.from_dir(train_files)
    logging.info(f'Loaded imageset into memory: {getsizeof(trainset)}')

    # Instantiate loader
    trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    net = ResNetWrapper()#CNN()

    net.fit(trainloader, 
            criterion = nn.MultiLabelSoftMarginLoss(),
            optimizer = torch.optim.Adam,
            learning_rate = LEARNING_RATE,
            epochs = EPOCHS)

    if SAVE_PATH is not None: torch.save(net.state_dict(), SAVE_PATH)

    logging.info(f'Saved weights to {SAVE_PATH}')


    # Validating accuracy of model
    logging.info(f'Verifying accuracy')

    testloader = DataLoader(CaptchaDataset.from_dir(test_files), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    label_accuracy, char_accuracy = net.validate(testloader)

    logging.info('Run_fit.py complete.')
    return

if __name__ == "__main__":
    run()

