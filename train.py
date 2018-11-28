import argparse
import os
import numpy as np

import torch
import torch.utils.data
from utils import prepare_art_pop_datasets, prepare_key_datasets, evaluate, eval_regression, save, load, \
    ArtPopDataset, KeyDataset, adjust_learning_rate
from models import Artist, Popularity, Key
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='MUSI 8903 Final Project')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate')
# parser.add_argument('--momentum', type=float, metavar='M',
#                     help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=100,
                    help='number of epochs to train')
parser.add_argument('--model', default='popularity',
                    choices=['artist', 'popularity', 'key'],
                    help='which model to train/evaluate')
parser.add_argument('--save-dir', default='models/')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(0)
np.random.seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)

# Fetch torch Datasets
train_set, val_set, test_set = '', '', ''
if args.model == 'artist':
    train_set, val_set, test_set = prepare_art_pop_datasets(value_path='./data/value_train_artist.npy',
                                                            label_path='./data/label_train_artist.npy',
                                                            splits=[0.7, 0.15, 0.15])
elif args.model == 'popularity':
    train_set, val_set, test_set = prepare_art_pop_datasets(value_path='./data/value_train_popularity.npy',
                                                            label_path='./data/label_train_popularity.npy',
                                                            splits=[0.7, 0.15, 0.15])
elif args.model == 'key':
    train_set, val_set, test_set = prepare_key_datasets(timbre_path='./data/timbre_train_key.npy',
                                                        chroma_path='./data/chroma_train_key.npy',
                                                        label_path='./data/label_train_key.npy',
                                                        splits=[0.7, 0.15, 0.15])
else:
    raise Exception('Incorrect model name')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
print('loader : ', train_loader)

# Initialize the model
if args.model == 'artist':
    model = Artist()
elif args.model == 'popularity':
    model = Popularity()
elif args.model == 'key':
    model = Key()
else:
    raise Exception('Incorrect model name')

if args.cuda:
    model.cuda()

# Loss function
if args.model == 'popularity':
    loss = torch.nn.MSELoss()
else:
    loss = torch.nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Saved losses for plotting
losses = []
val_losses = []
accs = []
val_accs = []
r_sqs = []
val_r_sqs = []


def train(epoch):
    # adaptive learning rate
    adjust_learning_rate(epoch, optimizer, adjust_every=10, rate=0.9)

    mean_training_loss = 0.0
    running_loss = 0.0
    model.train()
    correct = 0.0
    running_examples = 0.0
    total_predictions = np.array([])
    total_labels = np.array([])

    for i, batch in tqdm(enumerate(train_loader)):
        # inputs
        if args.model == 'key':
            inputs_a, inputs_b, labels = batch
        else:
            inputs, labels = batch

        # normalize regression labels, or reformat classification labels
        if args.model == 'popularity':
            labels = labels / 100
        else:
            labels = labels.type(torch.LongTensor)
            labels = labels.view(-1)

        # enable gpu
        if torch.cuda.is_available():
            if args.model == 'key':
                inputs_a = inputs_a.cuda()
                inputs_b = inputs_b.cuda()
            else:
                inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        # forward pass, backward pass, optimizer
        if args.model == 'key':
            model.init_hidden(inputs_a.shape[0])
            outputs = model(inputs_a, inputs_b)
        else:
            outputs = model(inputs)

        loss_size = loss(outputs, labels)
        loss_size.backward()
        optimizer.step()

        # print statistics
        if args.model == 'popularity':
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

    mean_training_loss = running_loss / len(train_loader)

    print('Training Epoch:', epoch)

    # regression metrics, used only for popularity
    r_sq, accuracy = eval_regression(total_labels, total_predictions)

    if args.model != 'popularity':
        accuracy = correct / running_examples
        print('Training Epoch:', epoch)
        print('Training Loss: {:.6f} \t'
              'Training Acc.: {:.6f}'.format(mean_training_loss, accuracy))
    else:
        print('Training Loss: {:.6f} \t'
              'Training R_Sq.: {:.6f}'.format(mean_training_loss, r_sq))

    losses.append(mean_training_loss)
    accs.append(accuracy)
    r_sqs.append(r_sq)

    return


# Training and evaluation loop
# Save model with best val loss

best_val_loss = 9.0
for i in range(args.epochs):
    train(i)
    criterion = loss

    val_loss, val_acc, val_r_sq = evaluate(val_loader, model, args.model, criterion)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_r_sqs.append(val_r_sq)

    if args.model == 'popularity':
        print('Validation Loss: {:.6f} \t'
               'Validation RSQ.: {:.6f}'.format(val_loss, val_r_sq))
    else:
        print('Validation Loss: {:.6f} \t'
                'Validation Acc.: {:.6f}'.format(val_loss, val_acc))

    if val_loss < best_val_loss:
        if args.model == 'artist':
            save(model, 'artist.pth')
        elif args.model == 'popularity':
            save(model, 'popularity.pth')
        elif args.model == 'key':
            save(model, 'key.pth')
        best_val_loss = val_loss


def test(model):
    criterion = loss
    test_loss, test_acc, test_r_sq = evaluate(test_loader, model, args.model, criterion)

    if args.model == 'popularity':
        print('Test Loss: {:.6f} \t'
              'Test RSQ.: {:.6f}'.format(test_loss, test_r_sq))
    else:
        print('Test Loss: {:.6f} \t'
              'Test Acc.: {:.6f}'.format(test_loss, test_acc))


# Load best model and test
if args.model == 'artist':
    msd = load('artist.pth')
    model.load_state_dict(msd)
elif args.model == 'key':
    msd = load('key.pth')
    model.load_state_dict(msd)

test(model)

# Plot results
plt.plot(range(args.epochs), losses, 'k', range(args.epochs), val_losses, 'b')
plt.legend(['loss', 'val loss'])
plt.show()
if args.model == 'popularity':
    plt.plot(range(args.epochs), r_sqs, 'k', range(args.epochs), val_r_sqs, 'b')
    plt.legend(['r2', 'val r2'])
else:
    plt.plot(range(args.epochs), accs, 'k', range(args.epochs), val_accs, 'b')
    plt.legend(['accs', 'val accs'])
plt.show()
