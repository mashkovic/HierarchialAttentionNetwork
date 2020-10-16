import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths, get_evaluation
from dataset import HANData
from HAN import HierAttNet
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from typing import Any

print(torch.__version__)

# following standards
# feeding arguements in the py file instead of stdin
opt = {
    'vocab_path': './vocab.pkl',
    'saved_path': './out',
    'batch_size': 64,
    'train_set': './myCorpus.csv',
    'word_hidden_size': 200,
    'sent_hidden_size': 100,
    'lr': 0.0001,
    'num_epochs': 20,
    'test_interval': 1,
    'es_min_delta': 0.0001,
    'es_patience': 10,
    'momentum': 0.9,
    'output': './output/mashrur_new_test'
}


def get_args():
    """Parses arguements for the model to run
    """
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=200)
    parser.add_argument("--sent_hidden_size", type=int, default=100)
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=0.001,
        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument(
        "--es_patience",
        type=int,
        default=10,
        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="./myCorpus.csv")
    parser.add_argument("--test_set", type=str, default="./myCorpus.csv")
    parser.add_argument(
        "--test_interval",
        type=int,
        default=1,
        help="Number of epoches between testing phases")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="../vocab/newsela_vocab.pk")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="mashrur_new_test")
    # parser.add_argument('configs/Newsela/han.yaml', default='test')

    args = parser.parse_args()
    return args


def train(
        opt,
        task,
        data_path=None,
        fed_data=None,
        k_fold=0,
        cols_to_train=['tweets', 'judgments'],
        subsample=1,
        overwrite=True,
        config=''):
    """
    Runs the HAN with the given parameters

    Args:
        opt (dict): Configuration for the model
        task (str): Name of the task
        data_path (str, optional): Path to the data. Defaults to None.
        fed_data (pd.DataFrame, optional): [description]. Defaults to None.
        k_fold (int, optional): [description]. Defaults to 0.
        cols_to_train (list, optional): [description]. Defaults to ['tweets', 'judgments'].
        subsample (int, optional): [description]. Defaults to 1.
        overwrite (bool, optional): [description]. Defaults to True.
        config (str, optional): [description]. Defaults to ''.

    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2019)
    else:
        torch.manual_seed(2019)

    output_file = open(opt['saved_path'] + "logs.txt", "w")

    output_file.write("Model's parameters: {}".format(opt.items()))

    training_params = {"batch_size": opt['batch_size'],
                       "shuffle": False,
                       "drop_last": True}

    test_params = {"batch_size": opt['batch_size'],
                   "shuffle": False,
                   "drop_last": False}

    max_word_length, max_sent_length = get_max_lengths(
        opt['train_set'], fed_data=fed_data)

    if fed_data is None:
        df_data = pd.read_csv(data_path, encoding='utf8', sep=',')
    else:
        df_train = pd.DataFrame(fed_data[0], columns=cols_to_train),
        df_test = pd.DataFrame(fed_data[1], columns=cols_to_train)

    kf = model_selection.KFold(n_splits=5)

    predicted_all_folds = []
    true_all_folds = []
    counter = 0
    accuracies_all_folds = []
    precision_all_folds = []
    recall_all_folds = []
    f1_all_folds = []

    if fed_data is None:
        kf = model_selection.KFold(n_splits=5)
        iterator = kf.split(df_data)
    else:
        iterator = [(0, 0)]

    for train_index, test_index in iterator:

        counter += 1

        print()
        print("CV fold: ", counter)
        print()

        if os.path.exists(opt['vocab_path']):
            os.remove(opt['vocab_path'])

        if fed_data is None:
            df_train, df_test = df_data.iloc[train_index], df_data.iloc[test_index]
            sep_idx = int(df_test.shape[0] / 2)
            df_valid = df_test[:sep_idx]
            df_test = df_test[sep_idx:]
            print(
                "Train size: ",
                df_train.shape,
                "Valid size: ",
                df_valid.shape,
                "Test size: ",
                df_test.shape)

        training_set = HANData(
            df_train,
            opt['vocab_path'],
            task,
            max_sent_length,
            max_word_length)
        training_generator = DataLoader(training_set, **training_params)

        test_set = HANData(
            df_test,
            opt['vocab_path'],
            task,
            max_sent_length,
            max_word_length)
        test_generator = DataLoader(test_set, **test_params)

        valid_set = HANData(
            df_valid,
            opt['vocab_path'],
            task,
            max_sent_length,
            max_word_length)
        valid_generator = DataLoader(valid_set, **test_params)

        model = HierAttNet(
            opt['word_hidden_size'],
            opt['sent_hidden_size'],
            opt['batch_size'],
            training_set.num_classes,
            opt['vocab_path'],
            max_sent_length,
            max_word_length)

        if torch.cuda.is_available():
            model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
        best_loss = 1e5
        best_epoch = 0
        num_iter_per_epoch = len(training_generator)

        for epoch in range(opt['num_epochs']):
            model.train()
            for iter, (feature, label) in enumerate(training_generator):
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                model._init_hidden_state()
                predictions = model(feature)
                print(predictions.shape)
                loss = criterion(predictions, label)
                loss.backward()
                optimizer.step()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                training_metrics = get_evaluation(
                    label.cpu().numpy(),
                    predictions.cpu().detach().numpy(),
                    list_metrics=["accuracy"])
                print(
                    "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                        epoch + 1,
                        opt['num_epochs'],
                        iter + 1,
                        num_iter_per_epoch,
                        optimizer.param_groups[0]['lr'],
                        loss,
                        training_metrics["accuracy"]))

            if epoch % opt['test_interval'] == 0:
                model.eval()
                loss_ls = []
                te_label_ls = []
                te_pred_ls = []
                for te_feature, te_label in valid_generator:
                    num_sample = len(te_label)
                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()
                    with torch.no_grad():
                        model._init_hidden_state(num_sample)
                        te_predictions = model(te_feature)
                    te_loss = criterion(te_predictions, te_label)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.append(te_predictions.clone().cpu())
                te_loss = sum(loss_ls) / test_set.__len__()
                te_pred = torch.cat(te_pred_ls, 0)
                te_label = np.array(te_label_ls)
                test_metrics = get_evaluation(
                    te_label, te_pred.numpy(), list_metrics=[
                        "accuracy", "confusion_matrix"])

                print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt['num_epochs'],
                    optimizer.param_groups[0]['lr'],
                    te_loss, test_metrics["accuracy"]))
                if te_loss + opt['es_min_delta'] < best_loss:
                    best_loss = te_loss
                    best_epoch = epoch
                    print('Saving model')
                    torch.save(
                        model,
                        opt['saved_path'] +
                        os.sep +
                        f"{epoch}_{iter+1}_subsample_{subsample}_han.bin")

                # Early stopping
                if epoch - best_epoch > opt['es_patience'] > 0:
                    print(
                        "Stop training at epoch {}. The lowest loss achieved is {}".format(
                            epoch, best_loss))
                    break

        print()
        print('Evaluation: ')
        print()
        torch.save(
            model,
            opt['saved_path'] +
            os.sep +
            "whole_model_han.pt")
        model.eval()
        model = torch.load(opt['saved_path'] + os.sep + "whole_model_han.pt")
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for te_feature, te_label in test_generator:
            num_sample = len(te_label)
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            with torch.no_grad():
                model._init_hidden_state(num_sample)
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(
            te_label,
            te_pred.numpy(),
            list_metrics=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "confusion_matrix"])

        true = te_label
        preds = np.argmax(te_pred.numpy(), -1)
        predicted_all_folds.extend(preds)
        true_all_folds.extend(true)

        f1 = f1_score(true, preds, average='weighted')
        macro_f1 = f1_score(true, preds, average='macro')
        micro_f1 = f1_score(true, preds, average='micro')
        rmse = sqrt(mean_squared_error(true, preds))

        print("Test set accuracy: {}".format(test_metrics["accuracy"]))
        print("Test set precision: {}".format(test_metrics["precision"]))
        print("Test set recall: {}".format(test_metrics["recall"]))
        print("Test set f1: {}".format(test_metrics["f1"]))
        print("Test set cm: {}".format(test_metrics["confusion_matrix"]))

        accuracies_all_folds.append(test_metrics["accuracy"])
        precision_all_folds.append(test_metrics["precision"])
        recall_all_folds.append(test_metrics["recall"])
        f1_all_folds.append(test_metrics["f1"])
        print()

    print()
    print("Task: ", task)
    print("Accuracy: ", accuracy_score(true_all_folds, predicted_all_folds))
    print(
        'Confusion matrix: ',
        confusion_matrix(
            true_all_folds,
            predicted_all_folds))
    print('All folds accuracy: ', accuracies_all_folds)
    print('All folds precision: ', precision_all_folds)
    print('All folds recall: ', recall_all_folds)
    print('All folds f1: ', f1_all_folds)

    if fed_data is not None:
        class FakeMagpie:
            def predict_from_text(self, text: str, return_float=False):
                df_single = pd.DataFrame(
                    [[text, -1]], columns=[
                        cols_to_train[0], cols_to_train[1]])
                el_set = HANData(
                    df_single,
                    opt['vocab_path'],
                    task,
                    max_sent_length,
                    max_word_length)
                el_generator = DataLoader(el_set, **test_params)
                for te_feature, te_label in el_generator:
                    num_sample = len(te_label)
                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()
                    with torch.no_grad():
                        model._init_hidden_state(num_sample)
                        te_predictions = model(te_feature)
                if return_float:
                    return float(te_predictions.clone().cpu())
                else:
                    return int(te_predictions.clone().cpu())
        return FakeMagpie(), f1, macro_f1, micro_f1, rmse


if __name__ == "__main__":
    # Comment out below line to parse args from stdin
    # opt = get_args()
    train(opt, 'Twitter', opt['train_set'])


def main(
        task,
        fed_data=None,
        k_fold=0,
        subsample=1,
        overwrite=True,
        config='') -> Any:
    # Comment out below line to parse args from stdin

    # opt = get_args()
    return train(opt,
                 task,
                 data_path=None,
                 fed_data=fed_data,
                 k_fold=k_fold,
                 subsample=subsample,
                 overwrite=overwrite,
                 config=config)
