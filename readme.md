# Introductions
This project is the code of the paper "Multi-scale adversarial learning with difficult region supervision learning models for primary tumor segmentation"

# Network Architecture
![alt 属性文本](MSALDS_UNet.png "The overall framework of our proposed model")
# Usage

1. Required environment

```pip install -r requirements.txt```

2. Prepare the dataset

KiTS21:<https://kits-challenge.org/kits21/>

MSD Brain:<http://medicaldecathlon.com/>

Pancreas:<http://medicaldecathlon.com/>

3. Data augmentation

```run python Data_preprocess.py```

4. Randomly assign training, test and validation sets

```run python voc_annotation_medical.py```

5. train

```run python train_4_scale_gan_kidney.py```

6. test

```run python predict.py```