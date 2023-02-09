from PIL import Image
import numpy as np
from typing import List, Tuple
import logging

import torch

import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable

from torchvision.models import resnet18


captcha_length = 5
unique_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
unique_characters += [char.upper() for char in unique_characters]
unique_characters += ['%i'%i for i in range(0,10)]


class CaptchaDataset(Dataset):
    def __init__(self, data, labels):
        self.X = data
        self.y = labels
        return 

    @classmethod
    def import_image(cls, location:str) -> np.ndarray:
        """
        Import a single image.

        Parameters: location (str) Location of image
        Returns: (np.ndarray) Image dimensions = Captchas 40 x 150 x 3 RGB channels
        """
        image = Image.open(location)
        image.load()
        #image.show()
        data = np.asarray(image, dtype='float32')
        return data
    @classmethod
    def stack_images(cls, file_locations:List[str]) -> np.ndarray:
        """
        Stack imageset from directory.

        Parameters: file_locations (List[str]) List of image locations
        Returns: (np.ndarray) len(file_locations) x image dimensions
        """
        return np.array([ cls.import_image(location) for location in file_locations ])
    
    @classmethod
    def read_label_names(cls, file_locations:List) -> List[str]:
        """
        Simply extracts labels from filenames.

        Parameters: file_locations (List) List of image locations, Posix
        Returns: (List[str]) List of label names
        """
        labels = [file.stem for file in file_locations]
        
        return labels

    @classmethod
    def from_dir(cls, file_locations:List[str]):
        """
        Instantiate from only a list of files.

        Parameters: file_locations (List[str]) List of image locations
        Returns: (Captcha_Dataset) object
        """
        return cls(
            cls.stack_images(file_locations),
            cls.read_label_names(file_locations)
        )

    def transform(self, image:np.ndarray) -> torch.Tensor:
        """Apply dataset transform."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(image)

    def encode_label(self, label:str) -> np.ndarray:
        """
        We need to encode the 5-char label into 5, 60-node arrays.
        
        Parameters: label (str)
        Returns: np.ndarray of shape 5,60
        """
        label_array = []
        for char in label:
            node_array = [0]*len(unique_characters)
            node_array[unique_characters.index(char)] += 1
            label_array.append(node_array)
        return np.array(label_array)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        """Select one sample. DataLoader accesses samples through this function."""
        return self.transform(self.X[index]), self.encode_label(self.y[index]), self.y[index]
    
    def __len__(self) -> int:
        """Also needed for DataLoader."""
        return len(self.X)
    


def decode_single_prediction(label):
    """
    Helper function to decode a single prediction.

    Parameters: label
    Returns: predicted label
    """
    captcha = ''
    for char in label:
        outchar = unique_characters[np.argmax(char.detach())]
        captcha += outchar
    return captcha


class nnModuleWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = None
        self.CUDA = torch.cuda.is_available()

    def fit(self, trainloader:torch.utils.data.dataloader.DataLoader,
            criterion = nn.MultiLabelSoftMarginLoss(),
            optimizer = torch.optim.Adam,
            learning_rate:float = 1e-3,
            epochs:int = 20
            ) -> None:
        """
        Run fitting process on our from a set of training images.

        Parameters:
        trainloader (DataLoader object) Training set
        criterion (nn Loss object) Loss criterion
        optimizer (Optimizer function) Optimizer function
        learning_rate (float) 
        epochs (int)
        """

        # Check that we wrapped correctly
        model = self.network
        assert self.network is not None, 'self.network not defined!'
        if self.CUDA: model.cuda()

        # Instantiate optimizer
        optimizer = optimizer(model.parameters(), lr=learning_rate)

        logging.debug(f'{len(trainloader)} batches per epoch.')

        for epoch in range(epochs):
            
            logging.info(f'Starting epoch {epoch + 1}/{epochs}')
 
            running_loss = 0.
            for i, (images, label_array, labels) in enumerate(trainloader):

                # Zero grads
                optimizer.zero_grad()

                # Send to cuda
                if self.CUDA: images = Variable(images).cuda()
                if self.CUDA: label_array = Variable(label_array).cuda()

                # Forward iter
                prediction = model(images)

                # Calculate loss
                loss = criterion(
                        prediction.reshape( prediction.shape[0], captcha_length, len(unique_characters) ),
                        label_array
                        )

                # Backpropagate
                loss.backward()

                # Step optimizer
                optimizer.step()

                # Log stats
                running_loss += loss.item()
                logging.info(f'[{epoch + 1}, {i + 1}] loss: {loss:.3e}')
                
            logging.info(f'Finished epoch {epoch + 1}/{epochs} with total loss {running_loss}')
        logging.info('Finished fitting.')
        return


    def validate(self, testloader:torch.utils.data.dataloader.DataLoader) -> Tuple[float]:
        """
        Validate the accuracy of our model from a set of validation images.

        Parameters: model
        testloader (DataLoader object)
        """

        # Check that we wrapped correctly
        model = self.network
        assert self.network is not None, 'self.network not defined!'


        # Strings counter
        s_total = s_correct = 0

        # Characters counter
        c_total = c_correct = 0

        # Not training -- don't need to calc. gradients
        with torch.no_grad():
            # For each test batch
            for (images, label_array, labels) in testloader:

                # Predict label
                if self.CUDA: images = Variable(images).cuda()
                prediction = model(images)

                # For each label in batch
                for i,pred in enumerate(
                    prediction.reshape( prediction.shape[0], captcha_length, len(unique_characters) )
                    ):

                    # Retrieve correct label
                    correct_label = labels[i]

                    # Retrieve predicted label
                    predicted_label = decode_single_prediction(pred)

                    # Do the captchas match?
                    if correct_label == predicted_label: s_correct += 1
                    s_total += 1
                    
                    # Do any of the characters match (at correct pos.)?
                    for x,y in zip(correct_label, predicted_label):
                        if x == y:
                            c_correct += 1
                        c_total += 1
                    
                    logging.debug('Predicted: %s Ground Truth: %s'%(predicted_label, correct_label))
        
        # Calculate results
        label_accuracy = s_correct / s_total
        char_accuracy = c_correct / c_total
        logging.info(f'Label-level accuracy (whole captcha) : {(100 * label_accuracy):.3f}%')
        logging.info(f'Char-level accuracy (indiv. chars)   : {(100 * char_accuracy):.3f}%')
        return label_accuracy, char_accuracy

class ResNetWrapper(nnModuleWrapper):
    # https://arxiv.org/pdf/1512.03385v1.pdf
    def __init__(self):
        super().__init__()

        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        model.fc = nn.Linear(512, len(unique_characters) * captcha_length, bias=True)

        self.network = model

class BasicCNN(nnModuleWrapper):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, # 3 input channels
            6, # 6 output channels
            5, bias = False) # kernel of size 5x5

        self.pool = nn.MaxPool2d(2, # kernel size of 2
            2) # stride of 2

        self.conv2 = nn.Conv2d(6, 16, 5) # 6 in / 16 out / 5x5 kernel

        self.fc1 = nn.Linear(3808, # features in
            1404) # features out

        self.fc2 = nn.Linear(1404, 702)
        self.fc3 = nn.Linear(702, 
                            len(unique_characters) * captcha_length # 300 output nodes: 5-chars 60 nodes
                            , bias = True)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)#nn.Softmax(self.fc3(x))
        return x




class CNN(nnModuleWrapper):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

                nn.Conv2d(3, 32, kernel_size = 3, padding = 2, bias = False),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2),


                nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2),

                nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2),

                nn.Flatten(1),
                nn.Linear(18_432, len(unique_characters)*captcha_length)
                
                )
        
    def forward(self, x):
        return self.network(x)



def debug_steps(input, net):
    output = input
    for step in net.children():
        output = step(output)
        print(step, output.shape)
    return output
