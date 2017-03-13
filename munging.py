import pandas as pd
import numpy as np
import itertools
from collections import Counter
import os
import sys
import glob
from PIL import Image


def munge(train_filepath, test_filepath):

    # read in the data
    train_df = pd.read_json(train_filepath)
    test_df = pd.read_json(test_filepath)

    # count of photos #
    train_df["num_photos"] = train_df["photos"].apply(len)
    test_df["num_photos"] = test_df["photos"].apply(len)

    # count of "features" #
    train_df["num_features"] = train_df["features"].apply(len)
    test_df["num_features"] = test_df["features"].apply(len)

    # count of words present in description column
    train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
    test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

    # convert the created column to datetime object so as to extract more features
    train_df["created"] = pd.to_datetime(train_df["created"])
    test_df["created"] = pd.to_datetime(test_df["created"])

    # # Extract features like year, month, day, hour from date columns
    # train_df["created_year"] = train_df["created"].dt.year
    # test_df["created_year"] = test_df["created"].dt.year
    # train_df["created_month"] = train_df["created"].dt.month
    # test_df["created_month"] = test_df["created"].dt.month
    # train_df["created_day"] = train_df["created"].dt.day
    # test_df["created_day"] = test_df["created"].dt.day
    # train_df["created_hour"] = train_df["created"].dt.hour
    # test_df["created_hour"] = test_df["created"].dt.hour
    # train_df["day_of_week"] = train_df["created"].dt.dayofweek
    # test_df["day_of_week"] = test_df["created"].dt.dayofweek

    # Create price-per-bed, price-per-bath, bed-to-bath ratio features
    train_df['price_per_bed'] = train_df.price/train_df.bedrooms
    test_df['price_per_bed'] = test_df.price/test_df.bedrooms
    train_df['price_per_bath'] = train_df.price/train_df.bathrooms
    test_df['price_per_bath'] = test_df.price/test_df.bathrooms
    train_df['bed_to_bath'] = train_df.bedrooms/train_df.bathrooms
    test_df['bed_to_bath'] = test_df.bedrooms/test_df.bathrooms


    # Extract image sizes and total number of pixels as lists, else None
    def extract_image_sizes(listing_id):
        image_dimensions = None
        folderpath = 'data/images_sample/'+str(listing_id)
        for image_file in glob.glob(os.path.join(folderpath, '*.jpg')):
            im = Image.open(image_file)
            width, height = im.size
            if image_dimensions:
                image_dimensions.append((width, height))
            else:
                image_dimensions = [(width, height)]
        return image_dimensions

    def extract_pixel_counts(listing_id):
        pixel_counts = None
        folderpath = 'data/images_sample/'+str(listing_id)
        for image_file in glob.glob(os.path.join(folderpath, '*.jpg')):
            im = Image.open(image_file)
            width, height = im.size
            n_pixels = width*height
            if pixel_counts:
                pixel_counts.append(n_pixels)
            else:
                pixel_counts = [n_pixels]
        return pixel_counts

    train_df['image_sizes'] = train_df.listing_id.map(extract_image_sizes)
    test_df['image_sizes'] = test_df.listing_id.map(extract_image_sizes)
    train_df['image_pixels'] = train_df.listing_id.map(extract_pixel_counts)
    test_df['image_pixels'] = test_df.listing_id.map(extract_pixel_counts)


    # Extract mean pixel count for images associated with each listing, else 0
    def mean_pixel_ct(image_pixels_list):
        if image_pixels_list:
            return sum(image_pixels_list)*1./len(image_pixels_list)
        else:
            return 0

    train_df['mean_pixel_count'] = train_df.image_pixels.map(mean_pixel_ct)
    test_df['mean_pixel_count'] = test_df.image_pixels.map(mean_pixel_ct)


    # Save munged data to .json
    train_df.to_json(train_filepath[:-5]+"_munged.json")
    test_df.to_json(test_filepath[:-5]+"_munged.json")



if __name__ == '__main__':
    # Import data
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]

    # Munge data
    munge(train_filepath, test_filepath)
