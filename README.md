# cnn_depth_tensorflow
cnn_depth_tensorflow is an implementation of depth estimation using tensorflow.

Original paper is "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network".
https://arxiv.org/abs/1406.2283

Presentation of this project:
https://docs.google.com/presentation/d/1ytuDdWJUG9VTK7wCVm3BzbW1uuGI_sRgH0TCZc1PNe0/edit?usp=sharing

## requierments
- TensorFlow 1.3 with python 2.7
- Numpy
- Pillow

## File descriptions
- Depth.py defines the model and training operations. We built our computational graph and other training details here.
- Test.py is for testing using any .jpg input.

## Testing
1. Put your .jpg image at 'data/'
2. Make sure your filename appears in test_test.csv
3. Run

```
python2 Test.py
```
4. To output coarse results, edit Test.py:
```
REFINE_TRAIN = True
```

5. Since the original dataset is not included here, before training, you may have to first download http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat and drop it at 'data/'. Then run 
```
python convert_mat_to_img.py
```

Then run
```
python2 Depth.py
```

Some configuration:

Coarse train:
REFINE_TRAIN = False
FINE_TUNE = False

Fine train:
REFINE_TRAIN = True
FINE_TUNE = True

Coarse train should come first.
