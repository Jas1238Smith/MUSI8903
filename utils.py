import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_recall_fscore_support

torch.manual_seed(0)
np.random.seed(0)


class ArtPopDataset(Dataset):
    def __init__(self, v_train, l_train):
        super(ArtPopDataset, self).__init__()
        self.v_train = v_train
        self.l_train = l_train

    def __getitem__(self, index):
        return self.v_train[index], self.l_train[index]

    def __len__(self):
        return len(self.l_train)


class KeyDataset(Dataset):
    def __init__(self, t_train, c_train, l_train):
        super(KeyDataset, self).__init__()
        self.t_train = t_train
        self.c_train = c_train
        self.l_train = l_train

    def __getitem__(self, index):
        return self.t_train[index], self.c_train[index], self.l_train[index]

    def __len__(self):
        return len(self.l_train)


def prepare_art_pop_datasets(value_path='./data/value_train_popularity.npy',
                     label_path='./data/label_train_popularity.npy', splits=[0.7, 0.15, 0.15]):
    # Preparing data
    assert np.sum(splits) == 1
    assert splits[0] != 0
    assert splits[1] != 0
    assert splits[2] != 0

    # Load data into torch tensors
    value_train = torch.Tensor(np.load(value_path))
    label_train = torch.Tensor(np.load(label_path))

    X = ArtPopDataset(value_train, label_train)

    # Split dataset into training, validation, and test
    n_points = label_train.shape[0]

    train_split = (0, int(splits[0] * n_points))
    val_split = (train_split[1], train_split[1] + int(splits[1] * n_points))
    test_split = (val_split[1], val_split[1] + int(splits[2] * n_points))

    shuffle_indices = np.random.permutation(np.arange(n_points))

    train_indices = shuffle_indices[train_split[0]:train_split[1]]
    val_indices = shuffle_indices[val_split[0]:val_split[1]]
    test_indices = shuffle_indices[test_split[0]:test_split[1]]

    # Create torch datasets
    train_set = torch.utils.data.TensorDataset(torch.Tensor(X[train_indices][0]), torch.Tensor(X[train_indices][1]))
    val_set = torch.utils.data.TensorDataset(torch.Tensor(X[val_indices][0]), torch.Tensor(X[val_indices][1]))
    test_set = torch.utils.data.TensorDataset(torch.Tensor(X[test_indices][0]), torch.Tensor(X[test_indices][1]))

    return train_set, val_set, test_set


def prepare_key_datasets(timbre_path='./data/timbre_train_key.npy', chroma_path='./data/chroma_train_key.npy',
                     label_path='./data/label_train_key.npy', splits=[0.7, 0.15, 0.15]):
    # Preparing data
    assert np.sum(splits) == 1
    assert splits[0] != 0
    assert splits[1] != 0
    assert splits[2] != 0

    # load data into torch tensors
    timbre_train = torch.Tensor(np.load(timbre_path))
    chroma_train = torch.Tensor(np.load(chroma_path))
    label_train = torch.Tensor(np.load(label_path))

    X = KeyDataset(timbre_train, chroma_train, label_train)

    # Split dataset into training, validation, and test
    n_points = label_train.shape[0]

    train_split = (0, int(splits[0] * n_points))
    val_split = (train_split[1], train_split[1] + int(splits[1] * n_points))
    test_split = (val_split[1], val_split[1] + int(splits[2] * n_points))

    shuffle_indices = np.random.permutation(np.arange(n_points))

    train_indices = shuffle_indices[train_split[0]:train_split[1]]
    val_indices = shuffle_indices[val_split[0]:val_split[1]]
    test_indices = shuffle_indices[test_split[0]:test_split[1]]

    # Create torch datasets
    train_set = torch.utils.data.TensorDataset(torch.Tensor(X[train_indices][0]), torch.Tensor(X[train_indices][1]),
                                               torch.Tensor(X[train_indices][2]))
    val_set = torch.utils.data.TensorDataset(torch.Tensor(X[val_indices][0]), torch.Tensor(X[val_indices][1]),
                                             torch.Tensor(X[val_indices][2]))
    test_set = torch.utils.data.TensorDataset(torch.Tensor(X[test_indices][0]), torch.Tensor(X[test_indices][1]),
                                              torch.Tensor(X[test_indices][2]))

    return train_set, val_set, test_set


def evaluate(data_loader, model, name, criterion):
    mean_training_loss = 0.0
    running_loss = 0.0
    model.eval()
    correct = 0
    running_examples = 0
    total_predictions = np.array([])
    total_labels = np.array([])

    for i, batch in tqdm(enumerate(data_loader)):
        # inputs
        if name == 'key':
            inputs_a, inputs_b, labels = batch
        else:
            inputs, labels = batch

        # normalize regression labels
        if name == 'popularity':
            labels = labels / 100
        else:
            labels = labels.type(torch.LongTensor)
            labels = labels.view(-1)

        # enable gpu
        if torch.cuda.is_available():
            if name == 'key':
                inputs_a = inputs_a.cuda()
                inputs_b = inputs_b.cuda()
            else:
                inputs = inputs.cuda()
            labels = labels.cuda()

        # forward pass
        if name == 'key':
            model.init_hidden(inputs_a.shape[0])
            outputs = model(inputs_a, inputs_b)
        else:
            outputs = model(inputs)

        # print statistics
        loss_size = criterion(outputs, labels)

        if name == 'popularity':
            total_predictions = torch.cat((total_predictions, outputs.data), 0) if total_predictions.size else outputs.data
            total_labels = torch.cat((total_labels, labels), 0) if total_labels.size else labels
        else:
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total_labels = np.concatenate((total_labels, labels.detach().cpu().numpy()), axis=0)
            total_predictions = np.concatenate((total_predictions, predicted.detach().cpu().numpy()), axis=0)
            correct += (predicted == labels).sum().item()

        running_loss += loss_size.item()
        running_examples += len(labels)

    # regression metrics, used only for popularity
    r_sq, accuracy = eval_regression(total_labels, total_predictions)
    if name != 'popularity':
        # classification metrics
        precision_recall_f1score(total_labels, total_predictions)
        accuracy = correct / running_examples
    running_loss = running_loss / len(data_loader)
    return running_loss, accuracy, r_sq


def adjust_learning_rate(epoch, optimizer, adjust_every=10, rate=0.9):
    if epoch % adjust_every == (adjust_every - 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate
            print('lr: ', param_group['lr'])


def precision_recall_f1score(tar, pred):
    p, r, f, _ = precision_recall_fscore_support(tar, pred)
    print('precision: {:.3f}'.format(p.mean()))
    print('recall: {:.3f}'.format(r.mean()))
    print('f1score: {:.3f}'.format(f.mean()))


def eval_regression(target, pred):
    r_sq = r2_score(target, pred)
    accu = 0
    return r_sq, accu


def save(model, path):
    print('saving model')
    torch.save(model.state_dict(), path)


def load(path):
    state_dict = torch.load(path)
    return state_dict
