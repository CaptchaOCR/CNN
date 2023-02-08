# CaptchaOCR-API

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

1. Make sure python is installed.
`python3 --version`

2. Create a python virtual environment.
`python3 -m venv CNN`

3. Activate the CNN virtual environment. 
    
    a. For macOS and Linux:
`source ./CNN/bin/activate`

    b. For Windows:
`.\CNN\Scripts\activate.bat`

4. Install the dependencies for CNN.
`pip install -r ./requirements.txt`


## Run
Instructions to run the server will come soon.

## Deploy
Instructions for Docker deployment coming soon.