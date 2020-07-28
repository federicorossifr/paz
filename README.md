# (PAZ) Perception for Autonomous Systems
![Python package](https://github.com/oarriaga/paz/workflows/Python%20package/badge.svg)

Multi-level perception library in Python.

* PAZ contains the following tutorials/examples with training and inference pipelines as well as pre-trained models in tf2:

<center>

| Task (link to tutorial)    |Model (link to paper)  |
|---------------------------:|-----------------------| 
|Object detection            |SSD-512                |
|Probabilistic keypoint est. |Gaussian Mixture CNN   |
|Keypoint estimation         |HRNet                  |
|6D Pose estimation          |KeypointNet2D          |
|Implicit orientation        |AutoEncoder            |
|Emotion classification      |MiniXception           |
|Discovery of Keypoints      |KeypointNet            |
|Keypoint estimation         |KeypointNet2D          |
|Attention                   |Spatial Transformers   |
|Object detection            |HaarCascades           |

</center>

* PAZ has only three dependencies: [Tensorflow2.0](https://www.tensorflow.org/), [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).

## High-level API
PAZ has easy out-of-the-box inference:

``` python
from paz.pipelines import SSD512COCO

detect = SSD512COCO()

# you can detect directly from an RGB image
inferences = detect(image)
```

## Mid-level API (sequential)
PAZ allows you to construct easy data-augmentation pipelines:

``` python
from paz.abstract import SequentialProcessor
from paz import processors as pr

augment = SequentialProcessor()
augment.add(pr.RandomContrast())
augment.add(pr.RandomBrightness())
augment.add(pr.RandomSaturation())
augment.add(pr.RandomHue())

# you can now use this now as a normal function
image = augment(image)
```

Pipelines with **Mid-level API** doesn't stop here. PAZ has out-of-the-box data-augmentation pipelines for object detection, keypoint-estimation, image-classification and domain-randomization and multiple inferences.

## Mid-level API ()

``` python
class EmotionDetector(Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = XceptionClassifierFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name = self.classify(cropped_image)['class_name']
        return self.draw(image, boxes2D)
 ```
 
* For example, a simple API for detecting common-objects (COCO) from an image (check the demo): 


* PAZ has a low-level API for using functions as helpers for your project!

``` python
from paz.backend import boxes, camera, image, keypoints, quaternion
```

* PAZ has built-in messages e.g. ''Pose6D'' for easier data exchange with other libraries or frameworks.

* PAZ has custom callbacks for evaluating MAP in object detectors while training and drawing inferences of any model

* PAZ has modular losses for calculating metrics while training
    
* PAZ comes with data loaders for the following datasets:
    OpenImages, VOC, YCB-Video, FAT, FERPlus, FER2013

* PAZ has automatic batch creating and dispatching

## Installation

3. Run: `pip install . --user`

### Motivation
Even though there are multiple high-level computer vision libraries in different DL frameworks, I felt there was not a consolidated deep learning library for robot-perception in my framework of choice (Keras).

#### Why Keras over other frameworks/libraries?
In simple terms, I have always felt the API to be more mature.
It allowed me to express my ideas at the level of complexity that was required. 
Keras was often misinterpreted as a "beginners" framework; however, once you learn to abstract: Layer, Callbacks, Loss, Metrics or Model, the API remained intact and helpful for more complicated ideas. 
It allowed me to automate and write down experiments with no extra boilerplate code.
Furthermore, if someone wanted to abandon such comfort one could still create a custom training loop.

As a final remark, I would like to mention, that I feel that we might tend to forget the great effort and emotional carriage behind every open-source project.
I feel it's easy to blurry a company name with the individuals behind their project, and we forget that there is someone feeling our criticism and our praise.
Therefore, whatever good code you can find here, is all dedicated to the software-engineers and contributors of open-source projects like Pytorch, Tensorflow and Keras.
You put your craft out there for all of us to use and appreciate, and we ought first to give you our thankful consideration before we lay upon you our hardened criticism.

### Why PAZ?


