# CaptchaOCR Neural Network

The backend OCR CNN to recognize CaptchaV1 characters.
Built with PyTorch. Uses [Captcha Dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset).

## Roadmap

Backend:
- Build/Tune/Train NN
- (?) Setup Captcha generation for frontend samples: [PHP Captcha](https://github.com/Gregwar/Captcha)
- Build backend communication with Django Framework
- Deploy through a Docker container

Frontend:
- Upload/generate Captcha images

## Installation
To install for development:

### MacOS and Linux

1. Make sure python is installed.  
`python3 --version`

2. Create a python virtual environment.  
`python3 -m venv NNEnvironment`

3. Activate the CNN virtual environment.  
    `source ./NNEnvironment/bin/activate`

`.\NNEnvironment\Scripts\activate.bat`

4. Install the dependencies for NNEnvironment.  

    a. For machines with discrete GPUs (Graphics Cards):  
`pip install -r ./requirements_CUDA.txt`

    b. For machines without:  
`pip install -r ./requirements.txt`

### Windows

1. Make sure python is installed.  
`py --version`

2. Create a python virtual environment.  
`py -m venv NNEnvironment`

3. Activate the CNN virtual environment.  

    a. For cmd:  
`.\NNEnvironment\Scripts\activate.bat`

    b. For PowerShell:  
`.\NNEnvironment\Scripts\activate.ps1`

4. Install the dependencies for NNEnvironment.  

    a. For machines with discrete GPUs (Graphics Cards):  
`pip install -r ./requirements_CUDA.txt`

    b. For machines without:  
`pip install -r ./requirements.txt`


## Run
Instructions to run the server will come soon.

1. Run the virtual environment.

    a. For macOS and Linux:  
`source ./NNEnvironment/bin/activate`

    b. For Windows:  
`.\NNEnvironment\Scripts\activate.bat`

2. Train the nerual network.  
`python run_fitting.py`