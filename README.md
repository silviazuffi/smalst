# Three-D Safari: Learning to Estimate Zebra Pose, Shape, and Texture from Images "In the Wild"

Silvia Zuffi<sup>1</sup>, Angjoo Kanazawa<sup>2</sup>, Tanja Berger-Wolf<sup>3</sup>, Michael J. Black<sup>4</sup>
<sup>1</sup>IMATI-CNR, Milan, Italy, <sup>2</sup>University of California, Berkely
<sup>3</sup>University of Illinois at Chicago, <sup>4</sup>Max Planck Institute for Intelligent Systems, Tuebingen, Germany
In ICCV 2019


### Requirements
- Python 2.7
- [PyTorch](https://pytorch.org/) tested on version `0.5.0`

### Installation

#### Setup virtualenv
```
virtualenv venv_smalst
source venv_smalst/bin/activate
pip install -U pip
deactivate
source venv_smalst/bin/activate
pip install -r requirements.txt
```

#### Install Neural Mesh Renderer and Perceptual loss
```
cd external;
bash install_external.sh
```
#### Install SMPL model
download the [SMPL model](https://ps.is.tuebingen.mpg.de/code/smpl/) and create a directory smpl_webuser under the smalst/smal_model directory

#### Usage

See the script in smalst/script directory for training and testing

#### Notes
The code in this repository is widely based on the project https://github.com/akanazawa/cmr
