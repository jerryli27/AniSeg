# AniSeg

Tensorflow models for Anime character object detection.

## Samples

### Face detection

![Face detection sample 1](samples/face_detection_outputs/PL0bHKk6wuUGL_Qd34mf0XsQnyiDk2OeGR_020_XNf_K-86qfc-0053.jpg)

![Face detection sample 2](samples/face_detection_outputs/s%20-%20799900.jpg)

### Figure segmentation

![Figure detection sample 1](samples/figure_detection_outputs/s%20-%20799910.jpg)

![Figure detection sample 2](samples/figure_detection_outputs/s%20-%20799903.jpg)

![Figure detection sample 3](samples/figure_detection_outputs/PL0bHKk6wuUGL_Qd34mf0XsQnyiDk2OeGR_040_up3W6vwnqWE-0017.jpg)

## Usage

We provide two pretrained models: One for face detection and one for figure segmentation. Download them at [Google Drive](https://drive.google.com/drive/folders/19PlNcku9V9pcJifSgWoBkZZZQhRDzV0W?usp=sharing)!

Here is a full example script to classify faces in the sample images:

```shell
# Assume you have saved the pretrained models under "model/" directory.

python infer_from_image.py \
--input_images=samples/inputs/*
--output_path=samples/face_detection_outputs
--inference_graph=model/face_detection/frozen_inference_graph.pb
--visualize_inference=True
```

In general, use the script with the following format

```shell
python infer_from_image.py \
--input_images=/PATH/TO/IMAGES/* \
--output_path=/PATH/TO/OUTPUT/FOLDER/  \
--inference_graph=/PATH/TO/frozen_inference_graph.pb  \
--visualize_inference=True
```

For figure segmentation model, please point `--inference_graph` to the segmentation model and add the additional flag

```shell
--detect_masks=True
```

## About the models

The Face Detection is trained on the outputs of [animeface-2009](https://github.com/nagadomi/animeface-2009) detector, which is under the MIT license. The main motivation for a tensorflow version is for **a much faster GPU-based detection** (5-10x speedup). 

The Figure Segmentation model is trained on our [TODO figure segmentation dataset](), which you can download [TODO here]().

Both model is trained using the [Tensorflow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) repo. Our repo is a simplified fork of that.

## Error cases

- Duplicate objects

![Detection error 1](samples/face_detection_outputs/PL0bHKk6wuUGL_Qd34mf0XsQnyiDk2OeGR_033_6ic7vtLu27o-0077.jpg)

- False negatives (missing detection)

![Detection error 2](samples/face_detection_outputs/PL0bHKk6wuUGL_Qd34mf0XsQnyiDk2OeGR_040_up3W6vwnqWE-0058.jpg)


- False positives

![Detection error 2](samples/figure_detection_outputs/PL0bHKk6wuUGL_Qd34mf0XsQnyiDk2OeGR_024_BkknXGYLZjg-0027.jpg)

## Anime related repos and datasets

Shameless self promotion of my [TwinGAN](https://github.com/jerryli27/TwinGAN) model to turn people into anime characteres and cats!

Sketch coloring using [PaintsTransfer](https://github.com/lllyasviel/style2paints) and [PaintsChainer](http://paintschainer.preferred.tech/).

Create anime portraits at [Crypko](https://crypko.ai/) and [MakeGirlsMoe](https://make.girls.moe/#/)

The all-encompassing anime dataset [Danbooru2017](https://www.gwern.net/Danbooru2017) by gwern.

## Common questions

#### What's the use case?

Getting clean data is hard. The object detectors provided here can make that process easier. For example it can be used to crop faces from the [Danbooru2017](https://www.gwern.net/Danbooru2017) dataset. Here we show some sample results of combining the two mdoels:

![use case 1](samples/use_cases/face_sample_0_0_1526222796.png)
![use case 1](samples/use_cases/face_sample_0_0_1526222838.png)
![use case 1](samples/use_cases/face_sample_0_0_1526222842.png)
![use case 1](samples/use_cases/face_sample_0_0_1526222843.png)
![use case 1](samples/use_cases/face_sample_0_0_1526222795.png)

#### Results are too noisy

There is always a tradeoff between precision and recall. Please adjust the `--min_score_thresh` flag accordingly. 

Higher threshold means more accurate results but also more false negatives (higher chance to miss objects). Lower value gives more faces but also contains non-face objects.

#### Inference is too slow

Inference using CPU is slow. Please make sure you have a GPU/TPU. 

If you want to create a full dataset using this model, we recommend following [TODO this guide](). 

## Todo items

- clean up code for creating tfrecords using the object detection model.
- provide figure segmentation dataset