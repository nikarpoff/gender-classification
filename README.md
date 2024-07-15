# gender-classification
Gender classification task by using CLIP ViT model for creating embeddings.

### Usage
In this repository you can find scripts for training models and for testing models. To use any of this provided instruments (exclude gui.py) you should load your testing/training datasets to package data, change paths to images and update datasets (use dataPreparing.loaders) in the code if it required.

To work with PyTorch&CLIP:
```bash
  $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  $ pip install ftfy regex tqdm
  $ pip install git+https://github.com/openai/CLIP.git
```

To work with data:
```bash
  $ pip install pillow pandas numpy scikit-learn
```

To run gui.py:
```bash
  $ pip install tk
```

To run tester.py with report generation:
```bash
  $ pip install reportlab
```

#### Training
If you want to train new model, just run `python -m main`. You also can change (from code of main.py):
- train and test dataset
- number of k-fold splits
- number of epochs
- batch size
- optimizer
- loss function

#### Testing
##### Demo
You can run demo script by `python -m testing.demo`. By default, this code makes 9 predictions for 9 randomly chosen images from the CelebA dataset (./data/celeba/). Default model is gc-simple-dnn-1.

##### Tester
You can run testing script by `python -m testing.tester`. This script takes two command-line arguments: model_name and dataset_name. By default: `python -m testing.tester gc-simple-dnn-1 celeba`. This script creates a pdf report with testing information.

##### GUI
To select your own image without dataset you can use gui.py: `python -m testing.gui`. This script allows you to select any image from your local storage and predicts the gender of a human from this picture.
