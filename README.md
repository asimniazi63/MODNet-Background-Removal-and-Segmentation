# Original Repo
https://github.com/ZHKKKe/MODNet

# Pre-trained model

create a directory named "pretrained"

Pre-trained model can be downloaded using following command:

$ gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O pretrained/modnet_photographic_portrait_matting.ckpt


or use the following link to download directly from google drive:

https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing

# dependencies

$ pip install -r requirements.txt

Note: Demo Tested with torch==1.1.0 and torchvision==0.3.0

# Demo
python inference.py --input example.jpg
