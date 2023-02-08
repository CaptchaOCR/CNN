from PIL import Image
import numpy as np
from typing import List, Tuple

import torchvision.transforms as T
from torch.utils.data import Dataset

import torch.nn as nn
import torch

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
        return np.array([cls.import_image(location) for location in file_locations ])
    
    @classmethod
    def read_label_names(cls, file_locations:List[str]) -> List[str]:
        """
        Simply extracts labels from filenames.

        Parameters: file_locations (List[str]) List of image locations
        Returns: (List[str]) List of label names
        """
        labels = [file.split('/')[-1].split('.')[0] for file in file_locations]
        
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
        return T.ToTensor()(image) # This is a hack for now.
        # Not sure why, but this transforming doesn't work. It's weird. Idk.
        # I originally tried using only PIL images and then resizing from there, but it didn't work.
        # Tried now going from PIL --> ndarray --> PIL --> Tensor; also doesn't work. 
        # Bit lost.
        # return  T.Compose([
        #     T.ToPILImage(),
        #     T.Resize([40, 150]),
        #     T.ToTensor()
        #     ])(image)

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
    

def decode_prediction(prediction:torch.Tensor, print_comparison=False, return_accuracy=True, labels=None):
    """
    Helper function to decode the prediction matrix.

    Parameters: prediction (torch.Tensor)
    print_comparison (bool) Print each decoded / ground truth label for comparison.
    return_accuracy (bool) Return count for number correct & total

    Returns:
    if return_accuracy == True: Tuple[prediction labels (List[str]), correct counts (int), total counts (int)]
    else: prediction labels (List[str])
    """

    total=correct=0

    pred_labels = []
    for i,single_pred in enumerate(prediction.reshape(prediction.shape[0], captcha_length, len(unique_characters))):
        captcha = ''
        for char in single_pred:
            outchar = unique_characters[np.argmax(char.detach())]
            captcha += outchar
        pred_labels.append(captcha)

    if print_comparison: print('Predicted: %s Ground Truth: %s'%(captcha, labels[i]))

    if return_accuracy:
        if captcha == labels[i]: correct+=1
        total+=1

    if return_accuracy: return (pred_labels, correct, total)
    else: pred_labels


class BasicCNN(nn.Module):
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


class CNN(nn.Module):
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

"""
class ModularCNN(nn.Module):
    def __init__(self):
        super().__init__()
"""
"""Let's organize this one ~ like resnet -- https://arxiv.org/pdf/1512.03385v1.pdf
Self-connected blocks of conv -> batchnorm -> relu -> conv -> batchnorm """