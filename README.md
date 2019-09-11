# Three-D Safari: Learning to Estimate Zebra Pose, Shape, and Texture from Images "In the Wild"

Silvia Zuffi<sup>1</sup>, Angjoo Kanazawa<sup>2</sup>, Tanya Berger-Wolf<sup>3</sup>, Michael J. Black<sup>4</sup>

<sup>1</sup>IMATI-CNR, Milan, Italy, <sup>2</sup>University of California, Berkely
<sup>3</sup>University of Illinois at Chicago, <sup>4</sup>Max Planck Institute for Intelligent Systems, Tuebingen, Germany

In ICCV 2019

![alt text](https://github.com/silviazuffi/smalst/blob/master/docs/teaser4.jpg)

<p align="center">
  <img src="https://github.com/silviazuffi/smalst/blob/master/docs/zebra_video.gif">
</p>


[paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/533/6034.pdf)

[suppmat](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/535/6034_supp.pdf)


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

#### Download data
- [Trained network](https://drive.google.com/a/berkeley.edu/file/d/1ZkKmqlbs3LlcGTrMK1j0ZVBpddg9b6Jf/view?usp=drivesdk)

- [Training data](https://drive.google.com/a/berkeley.edu/file/d/1yVy4--M4CNfE5x9wUr1QBmAXEcWb6PWF/view?usp=drivesdk) 

- [Test data](https://drive.google.com/a/berkeley.edu/file/d/1g5jZeA2ptAgdKVOAbZoVqsU-dNE-HD-e/view?usp=drivesdk)

- [Validation data](https://drive.google.com/a/berkeley.edu/file/d/1Ae0J83Y7Un1zBYFVd2za94d1KNnks8IL/view?usp=drivesdk)

The test and validation data are images collected in [The Great Grevy's Rally 2018](https://www.marwell.org.uk/media/other/cs_report_ggr_2018v.4.pdf)

#### Usage

See the script in smalst/script directory for training and testing

#### Notes
The code in this repository is widely based on the project https://github.com/akanazawa/cmr

#### Citation

If you use this code please cite
```
@inproceedings{Zuffi:ICCV:2019,
  title = {Three-D Safari: Learning to Estimate Zebra Pose, Shape, and Texture from Images "In the Wild"},
  author = {Zuffi, Silvia and Kanazawa, Angjoo and Berger-Wolf, Tanya and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  month = oct,
  year = {2019},
  month_numeric = {10}
}
```

#<p align="center">
#  <img src="https://github.com/silviazuffi/smalst/blob/master/docs/zebra_video.gif">
#</p>


