# Keras Image Classification

A simplified dog-vs-cat classifier using Keras and VGG16 transfer learning.

## Requirements

- keras
- numpy
- h5py
- pillow

## Usage

1. Arrange your dataset like this:

   * project
     * data
       * train
         * dogs
         * cats
       * validation
         * dogs
         * cats
       * test
         * test

2. Run the simplified training/prediction script:

```bash
python img_clf.py
```

3. Predictions are written to `prediction.csv`.

## Included files

- `img_clf.py` — simplified training and prediction pipeline
- `cats_n_dogs.ipynb` — notebook version
- `cats_n_dogs_BN.ipynb` — notebook with batch normalization enhancements
- `vgg_bn.py` — VGG16 with batch normalization helper class
