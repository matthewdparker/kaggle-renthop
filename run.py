import sys
import os
from munging import munge
from modeling import model

if __name__ == '__main__':
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]

    munged_train_filepath = train_filepath[:-5]+'_munged.json'
    munged_test_filepath = test_filepath[:-5]+'_munged.json'
    save_predictions_path = sys.argv[3]

    munge(train_filepath, test_filepath)
    model(munged_train_filepath, munged_test_filepath, save_predictions_path)
