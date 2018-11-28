MUSI 8903 Group 6: Beach Clark and Jason Smith

Title: Classifying Spotify Music Using Deep Learning

This project contains three models to train with data generated from the Spotify API.

## Requirements
    (All requirements can be installed with pip)
    pytorch
    numpy
    tqdm
    sklearn
    matplotlib
    
## train.py
    train
        training loop for an epoch, prints training loss and accuracy/r2
    test
        evaluates test data on trained model, prints training loss and accuracy/r2


## utils.py
    ArtPopDataset
        custom dataset with one input set and one label set
    
    KeyDataset
        custom dataset with two input sets and one label set
    
    prepare_art_pop_datasets
        create datasets with one input and split into training, validation, and testing
    
    prepare_key_datasets
        create datasets with one input and split into training, validation, and testing
    
    evaluate
        returns loss and accuracy/r2 for a model
    
    adjust_learning_rate
        multiplies learning rate parameter
    
    precision_recall_f1score
        prints following metrics (average over an epoch) for classification models:
            precision: ability to not label false positives
            recall: ability to find positives
            f1score: average of precision and recall
    
    eval_regression
        returns r2 score for regression model
    
    save
        saves best model, called when validation loss exceeds the previous best
    
    load
        loads best model for testing

## models.py
    Key
        2-layer CNN for pitch vectors
        2-layer CNN for timbre vectors
        2-layer RNN for concatenated CNN outputs
        Linear output layers, Tanh activations
        
    Artist
        3 fully-connected layers, Tanh activations
        
    Popularity
        3 fully-connected layers, Tanh activations

## How to Use
    1) run train.py with desired model type and parameters
    2) system prints loss and metrics for training, validation, and testing
    3) view loss and accuracy(key, artist) or r2 score(popularity)
    
## Link to Data
    https://drive.google.com/open?id=1BxjgrdCs2t7Z70Z2ldaLxPkUT5JTxZEc
    download and place in the data folder
