# kaggle-renthop

### Introduction
Repo for competing in Two Sigma / RentHop-sponsored Kaggle competition.

Check out the competition and get the data at https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries


### How to Use This Repo
There are three main scripts:

- **munging.py** does the work of feature engineering and manipulation. It is executed with the following commands in the terminal:

    ```$ python munging.py <train filepath> <test_filepath>```


- **modeling.py** builds and trains an XGBoost model with previously tuned hyperparameters, then makes predictions on the test data and saves these to a .csv in the format required by the competition. It is executed with the following commands in the terminal:

    ```$ python modeling.py <train filepath> <test filepath> <save filepath>```


- **run.py** is a wrapper script for executing munging.py and modeling.py simultaneously. It is executed with the following commands in the terminal:

    ```$ python run.py <train filepath> <test filepath> <save filepath>```


### Engineered Features:

- **num_photos**, number of photos in listing
- **num_features**, count of the length of the "features" column
- **num_description_words**, count of number of words in the description column
- **created**, the creation date as a python datetime object
- **price_per_bed**, the price / bedroom ratio
- **price_per_bath**, the price / bathroom ratio
- **bed_to_bath**, the bedroom / bathroom ratio
- **image_sizes**, list of photo dimensions (tuple of number of pixels, else None)
- **pixel_counts**, list of photo pixel counts, else None
- **mean_pixel_ct**, the mean of pixel counts, else 0
