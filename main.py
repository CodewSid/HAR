# -*- coding: utf-8 -*-
"""Human Activity Recognition (HAR) using FNN and RNN Deep Learning Models."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from models import FullyConnectedNetwork, RecursiveNeuralNetwork

# Ensure results directory exists
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Preprocess Data

def downSampleData(dataset, downsamplesize=100000):
    dataset = np.asarray(dataset)
    downSizeFeatureDatas = []
    yValue = ['bike', 'sit', 'stairsdown', 'stairsup', 'stand', 'walk']
    for output in yValue:
        typeIndex = np.where(dataset[:, 4] == output)[0]
        if len(typeIndex) == 0:
            continue
        actual_size = min(downsamplesize, len(typeIndex))
        downSizeFeatureIndex = np.random.choice(typeIndex, size=actual_size, replace=False)
        downSizeFeatureData = dataset[downSizeFeatureIndex]
        downSizeFeatureDatas.append(downSizeFeatureData)

    dataDown = np.concatenate(downSizeFeatureDatas, axis=0)
    target = dataDown[:, 4]
    feature = dataDown[:, 0:3]

    for i in range(len(target)):
        target[i] = str(target[i])
    enc = LabelEncoder().fit(target)
    target = enc.transform(target)

    # to add userdetails to dataset
    userDetails = dataDown[:, 3]
    for i in range(len(userDetails)):
        userDetails[i] = str(userDetails[i])
    userEnc = LabelEncoder().fit(userDetails)
    userDetails = userEnc.transform(userDetails)

    dataDown = np.concatenate((feature, userDetails[:, None]), axis=1)
    dataDown = np.concatenate((dataDown, target[:, None]), axis=1)
    return dataDown, target, enc.classes_


def preProcessData(processData, down_size):
    cols_to_drop = [c for c in ['Arrival_Time', 'Creation_Time', 'Index', 'Model', 'Device'] if c in processData.columns]
    processData = processData.drop(labels=cols_to_drop, axis=1)
    to_drop = ['null']
    processData = processData[~processData['gt'].isin(to_drop)]
    processData = processData.dropna()
    processData = processData.reset_index(drop=True)
    return downSampleData(processData, down_size)


def create_loaders(downSampledData, batch_size=100, test_split=0.2, random_seed=42):
    dataset_size = len(downSampledData)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    downSampledData = downSampledData.astype(float)
    downSampledData = torch.from_numpy(downSampledData).float()

    train_loader = torch.utils.data.DataLoader(downSampledData, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(downSampledData, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Run HAR Deep Learning Models")
    parser.add_argument("--data-path", type=str, default="data/Phones_accelerometer.csv", help="Path to Phones_accelerometer.csv")
    parser.add_argument("--sample-size", type=int, default=100000, help="Downsample size per activity (default: 100000)")
    parser.add_argument("--fnn-epochs", type=int, default=50, help="Epochs for FNN (default: 50)")
    parser.add_argument("--rnn-epochs", type=int, default=50, help="Epochs for RNN (default: 50)")
    parser.add_argument("--fnn-batch-size", type=int, default=100, help="Batch size for FNN (default: 100)")
    parser.add_argument("--rnn-batch-size", type=int, default=1000, help="Batch size for RNN (default: 1000)")
    parser.add_argument("--skip-fnn", action="store_true", help="Skip FNN training")
    parser.add_argument("--skip-rnn", action="store_true", help="Skip RNN training")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively with plt.show()")
    args = parser.parse_args()

    # Resolve data path
    data_path = args.data_path
    if not os.path.exists(data_path):
        alt_path = os.path.join(os.path.dirname(__file__), data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            print(f"Error: Dataset not found at '{data_path}'.")
            print("Please run 'python prepare_data.py --sample' to generate a sample dataset,")
            print("or 'python prepare_data.py --download' to download the official UCI dataset.")
            sys.exit(1)

    print(f"Loading data from: {data_path}")
    data1 = pd.read_csv(data_path)
    print(f"Dataset loaded: {data1.shape[0]} rows, {data1.shape[1]} columns")

    # ==========================================
    # 1. Fully Connected Network (FNN)
    # ==========================================
    if not args.skip_fnn:
        print("\n" + "=" * 50)
        print("Training Fully Connected Network (FNN)")
        print("=" * 50)

        downSampledData, yData, yDataLabels = preProcessData(data1, args.sample_size)
        print(f"Preprocessed data shape: {downSampledData.shape}")
        print(f"Activity classes: {list(yDataLabels)}")

        train_loader, test_loader = create_loaders(downSampledData, batch_size=args.fnn_batch_size)

        noEpoch = args.fnn_epochs
        noInput = downSampledData.shape[1] - 1
        fnn = FullyConnectedNetwork(noEpoch, noInput)

        print(f"Training FNN for {noEpoch} epochs (inputs: {noInput}, batch size: {args.fnn_batch_size})...")
        trainAccuracy, testAccuracy = fnn.train(train_loader, test_loader)

        # Plot Accuracy
        rcParams['figure.figsize'] = 10, 4
        plt.figure()
        plt.plot(trainAccuracy)
        plt.plot(testAccuracy)
        plt.title('FNN Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fnn_acc_path = os.path.join(RESULTS_DIR, 'accuracy_fnn.png')
        plt.savefig(fnn_acc_path)
        print(f"Saved: {fnn_acc_path}")
        if args.show_plots:
            plt.show()
        plt.close()

        # Plot Loss
        plt.figure()
        trainLoss = 1 - np.asarray(trainAccuracy)
        testLoss = 1 - np.asarray(testAccuracy)
        plt.plot(trainLoss)
        plt.plot(testLoss)
        plt.title('FNN Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fnn_loss_path = os.path.join(RESULTS_DIR, 'loss_fnn.png')
        plt.savefig(fnn_loss_path)
        print(f"Saved: {fnn_loss_path}")
        if args.show_plots:
            plt.show()
        plt.close()

        # FNN Confusion Matrix
        yTrue, yPred, pred_fnn = fnn.predict(test_loader)
        CM = confusion_matrix(yTrue, yPred)
        fig, ax = plot_confusion_matrix(conf_mat=CM, figsize=(10, 6))
        plt.xticks(range(len(yDataLabels)), yDataLabels, rotation=45)
        plt.yticks(range(len(yDataLabels)), yDataLabels)
        fnn_cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_fnn.png')
        plt.savefig(fnn_cm_path)
        print(f"Saved: {fnn_cm_path}")
        if args.show_plots:
            plt.show()
        plt.close()

    # ==========================================
    # 2. Recurrent Neural Network (RNN)
    # ==========================================
    if not args.skip_rnn:
        print("\n" + "=" * 50)
        print("Training Recurrent Neural Network (RNN)")
        print("=" * 50)

        downSampledData, yData, yDataLabels = preProcessData(data1, args.sample_size)
        train_loader, test_loader = create_loaders(downSampledData, batch_size=args.rnn_batch_size)

        noOfNeurons = 10
        epochs = args.rnn_epochs
        noOfInputs = 4

        rnn = RecursiveNeuralNetwork(noOfInputs, noOfNeurons, epochs)
        rnn.batch_size = args.rnn_batch_size

        print(f"Training RNN for {epochs} epochs (neurons: {noOfNeurons}, batch size: {args.rnn_batch_size})...")
        rtrain_acc, rtest_acc, rloss = rnn.train(train_loader, test_loader)
        RNN_acc, RNN_targ, RNN_ypred = rnn.predict(test_loader)
        print(f"\nFinal RNN Test Accuracy: {RNN_acc:.2f}%")

        # RNN Confusion Matrix
        CM = confusion_matrix(RNN_targ, RNN_ypred)
        fig, ax = plot_confusion_matrix(conf_mat=CM, figsize=(10, 6))
        plt.xticks(range(len(yDataLabels)), yDataLabels, rotation=45)
        plt.yticks(range(len(yDataLabels)), yDataLabels)
        rnn_cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_rnn.png')
        plt.savefig(rnn_cm_path)
        print(f"Saved: {rnn_cm_path}")
        if args.show_plots:
            plt.show()
        plt.close()

    print("\nAll experiments completed successfully!")


if __name__ == '__main__':
    main()