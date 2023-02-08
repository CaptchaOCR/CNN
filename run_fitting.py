
### Fit params

DATA_DIR = "./Captchas"

BATCH_SIZE = 256
TEST_SIZE = .2
NUM_WORKERS = 100

EPOCHS = 20
LEARNING_RATE = 1e-3

SAVE_PATH = "./CNN_weights.pth"


from glob import glob
import logging
logging.basicConfig(level=logging.DEBUG)
from sys import getsizeof

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from Infrastructure import CaptchaDataset, unique_characters, captcha_length, CNN



CUDA = False
if torch.cuda.is_available():
    CUDA = True
logging.warning(f'CUDA available: {torch.cuda.is_available()}')

# Locate files
file_locations = glob(DATA_DIR+'/*')
captcha_names = [file.split('/')[-1].split('.')[0] for file in file_locations]
logging.info( f'Identified {len(file_locations)} images.' )

# Split training/test data
train_files, test_files = train_test_split(file_locations, test_size = .2)
print(f'Split dataset into 80:20 train/test of sizes {len(train_files)},{len(test_files)}.')

# Instantiate dataset
trainset = CaptchaDataset.from_dir(train_files)
logging.info(f'Loaded imageset into memory: {getsizeof(trainset)}')

# Instantiate loader
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

net = CNN()

net.fit(trainloader, 
        criterion = nn.MultiLabelSoftMarginLoss(),
        optimizer = torch.optim.Adam,
        learning_rate = 1e-3,
        epochs = 20)

if SAVE_PATH is not None: torch.save(net.state_dict(), SAVE_PATH)

logging.info(f'Saved weights to {SAVE_PATH}')


# Validating accuracy of model
logging.info(f'Verifying accuracy')

testloader = DataLoader(CaptchaDataset.from_dir(test_files), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

label_accuracy, char_accuracy = validate_model(net, testloader)

logging.info('Run_fit.py complete.')

