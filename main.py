import os
import sys
import errno
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import prosody_dataset
from prosody_dataset import Dataset
from prosody_dataset import load_embeddings
from model import Bert, BertLSTM, LSTM, BertRegression, LSTMRegression, WordMajority, ClassEncodings, BertAllLayers #, TransformerModel
from transformer import TransformerModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import ast

parser = ArgumentParser(description='Prosody prediction')

parser.add_argument('--datadir',
                    type=str,
                    default='./data')
parser.add_argument('--gpt',
                    type=int,
                    default=0)
parser.add_argument('--train_set',
                    type=str,
                    choices=['train_100',
                             'train_360', 'emma'],
                    default='train_360')
parser.add_argument('--batch_size',
                    type=int,
                    default=32)
parser.add_argument('--epochs',
                    type=int,
                    default=2)
parser.add_argument('--model',
                    type=str,
                    choices=['BertUncased',
                             'BertCased',
                             'BertLSTM',
                             'LSTM',
                             'BiLSTM',
                             'BertRegression',
                             'LSTMRegression',
                             'WordMajority',
                             'ClassEncodings',
                             'BertAllLayers',
                             'Transformer'],
                    default='BertUncased')
parser.add_argument('--nclasses',
                    type=int,
                    default=3)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=600)
parser.add_argument('--embedding_file',
                    type=str,
                    default='embeddings/glove.840B.300d.txt')
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--save_path',
                    type=str,
                    default='results.txt')
parser.add_argument('--log_every',
                    type=int,
                    default=10)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.00005)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=None)
parser.add_argument('--fraction_of_train_data',
                    type=float,
                    default=1
                    )
parser.add_argument("--optimizer",
                    type=str,
                    choices=['rprop',
                             'adadelta',
                             'adagrad',
                             'rmsprop',
                             'adamax',
                             'asgd',
                             'adam',
                             'sgd'],
                    default='adam')
parser.add_argument('--include_punctuation',
                    action='store_false',
                    dest='ignore_punctuation')
parser.add_argument('--sorted_batches',
                    action='store_true',
                    dest='sorted_batches')
parser.add_argument('--mask_invalid_grads',
                    action='store_true',
                    dest='mask_invalid_grads')
parser.add_argument('--invalid_set_to',
                    type=float,
                    default=-2) # -2 = log(0.01)
parser.add_argument('--log_values',
                    action='store_true',
                    dest='log_values')
parser.add_argument('--weighted_mse',
                    action='store_true',
                    dest='weighted_mse')
parser.add_argument('--shuffle_sentences',
                    action='store_true',
                    dest='shuffle_sentences')
parser.add_argument('--seed',
                    type=int,
                    default=1234)


def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def weighted_mse_loss(input,target):
    tgt_device = target.device
    BUFFER = torch.Tensor([3.0]).to(tgt_device)
    SOFT_MAX_BOUND = torch.Tensor([6.0]).to(tgt_device) + BUFFER
    weights = (torch.min(target + BUFFER, SOFT_MAX_BOUND) / SOFT_MAX_BOUND)
    weights = weights / torch.sum(weights)
    weights = weights.cuda()
    sq_err = (input-target)**2
    weighted_err = sq_err * weights.expand_as(target)
    loss = weighted_err.mean()
    return loss


def main():

    with open('/content/waveletProsody/contrastDict.txt', 'r') as file:
        contents = file.read()
        dictionary = ast.literal_eval(contents)

    np.set_printoptions(suppress=True)
    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU[{}] (torch.device({})).".format(config.gpu, device))
    else:
        device = torch.device('cpu')
        print("GPU not available so training on CPU (torch.device({})).".format(device))
        device = 'cpu'

    # Optimizer
    if config.optimizer == 'adadelta':
        optim_algorithm = optim.Adadelta
    elif config.optimizer == 'adagrad':
        optim_algorithm = optim.Adagrad
    elif config.optimizer == 'adam':
        optim_algorithm = optim.Adam
    elif config.optimizer == 'adamax':
        optim_algorithm = optim.Adamax
    elif config.optimizer == 'asgd':
        optim_algorithm = optim.ASGD
    elif config.optimizer == 'rmsprop':
        optim_algorithm = optim.RMSprop
    elif config.optimizer == 'rprop':
        optim_algorithm = optim.Rprop
    elif config.optimizer == 'sgd':
        optim_algorithm = optim.SGD
    else:
        raise Exception('Unknown optimization optimizer: "%s"' % config.optimizer)

    splits, tag_to_index, index_to_tag, vocab, fileDict = prosody_dataset.load_dataset(config)

    if config.model == "BertUncased" or config.model == "BertCased":
        model = Bert(device, config, labels=len(tag_to_index))
    elif config.model == "BertLSTM":
        model = BertLSTM(device, config, labels=len(tag_to_index))
    elif config.model == "LSTM" or config.model == "BiLSTM":
        model = LSTM(device, config, vocab_size=len(vocab), labels=len(tag_to_index))
    elif config.model == "BertRegression":
        model = BertRegression(device, config)
        config.ignore_punctuation = True
    elif config.model == "LSTMRegression":
        model = LSTMRegression(device, config, vocab_size=len(vocab))
        config.ignore_punctuation = True
    elif config.model == "WordMajority":
        model = WordMajority(device, config, index_to_tag)
    elif config.model == "ClassEncodings":
        model = ClassEncodings(device, config, index_to_tag, tag_to_index)
    elif config.model == "BertAllLayers":
        model = BertAllLayers(device, config, labels=len(tag_to_index))
    elif config.model == "Transformer":
        model = TransformerModel(device, 5, 768, 2, 150, 2, config)
        #model = TransformerModel(device, 5, 4096, 2, 150, 2, config)
        #model = TransformerModel(device, config, labels=len(tag_to_index))
    else:
        raise NotImplementedError("Model option not supported.")

    model.to(device)

    # word embeddings
    if config.model == 'LSTM' or config.model == 'BiLSTM':
        weights, word_to_embid = load_embeddings(config, vocab)
        model.word_embedding.weight.data = torch.Tensor(weights).to(device)
    else:
        word_to_embid = None

    train_dataset = Dataset(splits["train"], fileDict["train"], tag_to_index, config, word_to_embid)
    eval_dataset = Dataset(splits["dev"], fileDict["dev"], tag_to_index, config, word_to_embid)
    test_dataset = Dataset(splits["test"], fileDict["test"], tag_to_index, config, word_to_embid)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=not(config.sorted_batches), # will manually shuffle if sorted_batches desired
                                 num_workers=1,
                                 collate_fn=prosody_dataset.pad)
    dev_iter = data.DataLoader(dataset=eval_dataset,
                               batch_size=config.batch_size,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=prosody_dataset.pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=prosody_dataset.pad)

    if config.model in ["WordMajority"]:
        optimizer = None
    else:
        optimizer = optim_algorithm(model.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

    if config.model in ['BertRegression', 'LSTMRegression']:
        if config.weighted_mse:
            criterion = weighted_mse_loss
        else:
            criterion = nn.MSELoss()
    elif config.model == 'ClassEncodings':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    config.cells = config.layers

    if config.model == 'BiLSTM':
        config.cells *= 2

    if config.model == 'WordMajority': # 1 pass over the dataset is enough to collect stats
        config.epochs = 1

    print('\nTraining started...\n')
    best_dev_acc = 0
    best_dev_epoch = 0

    if config.model in ['BertRegression', 'LSTMRegression']:
        for epoch in range(config.epochs):
            print("Epoch: {}".format(epoch + 1))
            train_cont(model, train_iter, optimizer, criterion, device, config)
            valid_cont(model, dev_iter, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch + 1)
        test_cont(model, test_iter, criterion, index_to_tag, device, config)

    else:
        for epoch in range(config.epochs):
            print("Epoch: {}".format(epoch+1))
            train(model, train_iter, optimizer, criterion, device, config)
            valid(model, dev_iter, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch+1)
        test(model, test_iter, criterion, index_to_tag, device, config, dictionary)

# --------------- FUNCTIONS FOR DISCRETE MODELS --------------------

def train(model, iterator, optimizer, criterion, device, config):
    model.train()
    for i, batch in enumerate(iterator):
        #words, x, is_main_piece, tags, y, seqlens, _, _,  = batch
        words, x, is_main_piece, tags, y, seqlens, _, _ , prevSeq, lps, prevWords, file_id  = batch

        if config.model == 'WordMajority':
            model.collect_stats(x, y)
            continue

        optimizer.zero_grad()
        if x.shape[1] >511:
            print(x.shape)
            x = torch.cat((x[:, 0:1], x[:, -511:]), 1)
            y = torch.cat((y[:, 0:1], y[:, -511:]), 1)
            x = x.to(device)
            y = y.to(device)
        else:
            x = x.to(device)
            y = y.to(device)

        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        mask_slice = torch.zeros(logits.shape, dtype=bool)
        mask_sliceY = torch.zeros(y.shape, dtype=bool)
        for i, row in enumerate(mask_slice):
            row[lps[i]+1:, :] = 1
        mask_slice[:,0, :] = 1

        for i, row in enumerate(mask_sliceY):
            row[lps[i]+1:] = 1
        mask_sliceY[:,0] = 1

        logits = torch.masked_select(logits.to(device), mask_slice.to(device)).view(-1, logits.shape[-1])
        y = torch.masked_select(y.to(device), mask_sliceY.to(device))

        if config.model == 'ClassEncodings':
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1, y.shape[-1])  # also (N*T, VOCAB)
            loss = criterion(logits.to(device), y.to(device))
        else:
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            loss = criterion(logits.to(device), y.to(device))

        loss.backward()
        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))

    if config.model == 'WordMajority':
        model.save_stats()


def valid(model, iterator, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch):
    if config.model == 'WordMajority':
        return

    model.eval()
    dev_losses = []
    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #words, x, is_main_piece, tags, y, seqlens, _, _ = batch
            words, x, is_main_piece, tags, y, seqlens, _, _ , prevSeq, lps, prevWords, file_id  = batch
            x = x.to(device)
            y = y.to(device)

            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits.to(device), labels.to(device))

            dev_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
        preds = [index_to_tag[hat] for hat in y_hat]

        if config.model != 'LSTM' and config.model != 'BiLSTM':
            tagslice = tags.split()[1:-1]
            predsslice = preds[1:-1]
            assert len(preds) == len(words.split()) == len(tags.split())
        else:
            tagslice = tags.split()
            predsslice = preds
        for t, p in zip(tagslice, predsslice):
            if config.ignore_punctuation:
                if t != 'NA':
                    true.append(t)
                    predictions.append(p)
            else:
                true.append(t)
                predictions.append(p)

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    print(confusion_matrix(y_true, y_pred))

    if acc > best_dev_acc:
        best_dev_acc = acc
        best_dev_epoch = epoch
        dev_snapshot_path = 'best_model_{}_devacc_{}_epoch_{}.pt'.format(config.model, round(best_dev_acc, 2), best_dev_epoch)

        # save model, delete previous snapshot
        torch.save(model, dev_snapshot_path)
        for f in glob.glob('best_model_*'):
            if f != dev_snapshot_path:
                os.remove(f)

    print('Validation accuracy: {:<5.2f}%, Validation loss: {:<.4f}\n'.format(round(acc, 2), np.mean(dev_losses)))


def test(model, iterator, criterion, index_to_tag, device, config, dictionary):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Y_hat, prevSeqs, file_ids, allLogits = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #words, x, is_main_piece, tags, y, seqlens, _, _ = batch
            words, x, is_main_piece, tags, y, seqlens, _, _ , prevSeq, lps, prevWords, file_id = batch
            if x.shape[1] >511:
                print(x.shape)
                x = torch.cat((x[:, 0:1], x[:, -511:]), 1)
                y = torch.cat((y[:, 0:1], y[:, -511:]), 1)
                x = x.to(device)
                y = y.to(device)
            else:
                x = x.to(device)
                y = y.to(device)

            logitsO, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'ClassEncodings':
                logits = logitsO.view(-1, logitsO.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logitsO.view(-1, logitsO.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits, labels)

            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            prevSeqs.extend(prevWords)
            file_ids.extend(file_id)
            allLogits.extend(logitsO.detach().cpu().numpy().tolist())


    true = []
    predictions = []

    contrastTrue = []
    contrastPred = []
    contrastProb = []
    contrastFiles = ['023-65', '015-13', '012-14', '046-70', '046-48', '041-28', '041-16', '013-71', '022-54', '025-57', '040-102', '047-31', '047-68', '047-100','038-10', '036-49', '031-87', '052-15', '052-121', '052-104', '052-7', '001-30', '055-32', '037-39', '001-40', '039-96', '006-59', '006-7','006-71', '045-75', '020-38', '027-45', '027-68', '027-48', '018-253','018-45', '018-233', '011-71', '011-21', '016-53', '016-52', '044-64', '043-228', '043-87', '043-192', '017-52', '017-50', '028-16', '026-126','021-81', '021-50', '021-1', '035-35', '059-90', '057-41', '057-66','057-5', '057-80', '034-76', '033-78', '002-27', '056-125', '056-168','056-54', '058-106', '058-42', '058-92', '003-99', '004-20', '005-33','005-39', '005-44', '006-2', '008-72', '009-67', '010-39', '011-99', '013-2', '061-8', '060-50', '059-112', '057-39', '056-87', '056-51', '055-63', '052-75', '052-73', '052-54', '048-108', '048-107', '047-204', '046-97', '045-36', '043-214','041-12', '040-99', '039-37', '035-136', '035-41', '033-91', '026-102', '027-46','024-37', '024-36', '023-75', '020-18', '019-56', '018-52', '018-29', '018-30','016-107', '016-45', '003-60', '014-8', '047-76', '054-75', '009-58', '009-55', '052-71', '001-20', '020-58', '016-44', '016-11', '044-43', '035-49', '061-51', '057-56', '057-11', '051-71', '004-8', '005-24','005-41', '006-21', '008-20', '008-29', '009-106', '009-109', '061-40', '060-14', '060-13', '059-175', '059-119', '058-27', '057-57', '056-156', '056-113', '056-81', '056-77', '055-137','054-101', '053-71', '052-83', '052-84', '052-41', '052-25', '051-3', '047-167', '046-114', '046-96', '039-88', '034-60', '023-67', '021-48', '020-66', '019-30', '018-82', '018-44', '017-11', '016-105', '016-101', '016-47', '016-38']

    contrastFile = open('contrast.txt', "w")
    contrastFile.write(str(index_to_tag)+'\n')
    contrastFile.close()

    # gets results and save
    with open(config.save_path, 'w') as results:
        results.write(str(index_to_tag)+'\n')
        for words, is_main_piece, tags, y_hat, prevSeq, file_id, logits in zip(Words, Is_main_piece, Tags, Y_hat, prevSeqs, file_ids, allLogits):
            y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
            logits = [hat for head, hat in zip(is_main_piece, logits) if head == 1]
            preds = [index_to_tag[hat] for hat in y_hat]
            if config.model != 'LSTM' and config.model != 'BiLSTM':
                tagslice = tags.split()[1:-1]
                tagslice = tagslice[len(prevSeq.split()):]
                predsslice = preds[1:-1]
                predsslice = predsslice[len(prevSeq.split()):]
                logitslice = logits[1:-1]
                logitslice = logitslice[len(prevSeq.split()):]
                wordslice = words.split()[1:-1]
                wordslice = wordslice[len(prevSeq.split()):]

                #assert len(preds) == len(words.split()) == len(tags.split())
                #tagslice = np.array(tags.split())[np.where(np.array(is_main_piece)==1)[0]]
                #predsslice = np.array(preds)[np.where(np.array(is_main_piece)==1)[0]]
                #wordslice = np.array(words.split())[np.where(np.array(is_main_piece)==1)[0]]
            else:
                tagslice = tags.split()
                predsslice = preds
                wordslice = words.split()

            results.write(file_id[1] +'\n')
            if file_id[1] in dictionary:
                contrastFile = open('contrast.txt', "a")
                contrastFile.write(file_id[1] +'\n')
                contrastFile.close()

            counter = 0
            for w, t, p, l in zip(wordslice, tagslice, predsslice, logitslice):
                results.write("{}\t{}\t{}\t{}\n".format(w, t, p, softmax(np.array(l))))
                if config.ignore_punctuation:
                    if t != 'NA':
                        true.append(t)
                        predictions.append(p)
                        if file_id[1] in dictionary:
                            contrastFile = open('contrast.txt', "a")
                            contrastFile.write("{}\t{}\t{}\t{}\n".format(w, t, p, softmax(np.array(l))))
                            contrastFile.close()
                            if counter in dictionary[file_id[1]]:
                                contrastTrue.append(t)
                                contrastPred.append(p)
                                contrastProb.append(softmax(np.array(l)))
                        counter +=1
                else:
                    true.append(t)
                    predictions.append(p)
                    if file_id[1] in dictionary:
                        contrastFile = open('contrast.txt', "a")
                        contrastFile.write("{}\t{}\t{}\t{}\n".format(w, t, p, softmax(np.array(l))))
                        contrastFile.close()
                        if counter in dictionary[file_id[1]]:
                            contrastTrue.append(t)
                            contrastPred.append(p)
                            contrastProb.append(softmax(np.array(l)))
                    counter +=1
            results.write("\n")

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)

    yc_true = np.array(contrastTrue)
    yc_pred = np.array(contrastPred)

    f1 = f1_score(y_true, y_pred, average='weighted')
    print('f1 weighted', f1)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('f1 macro', f1)
    print(confusion_matrix(y_true, y_pred))

    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    accC = 100. * (yc_true == yc_pred).astype(np.int32).sum() / len(yc_true)

    print('Test accuracy: {:<5.2f}%, Test loss: {:<.4f} after {} epochs.\n'.format(round(acc, 2), np.mean(test_losses),
                                                                                   config.epochs))
    print('Contrast Accuracy: ', accC)

    final_snapshot_path = 'final_model_{}_testacc_{}_epoch_{}.pt'.format(config.model,
                                                                 round(acc, 2), config.epochs)
    torch.save(model, final_snapshot_path)



# ---------------- FUNCTIONS FOR CONTINUOUS MODELS ------------------
''' These are used only the BertRegression and LSTMRegression models for now '''

def train_cont(model, iterator, optimizer, criterion, device, config):

    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch

        optimizer.zero_grad()
        x = x.to(device)
        values = values.to(device)

        predictions, true = model(x, values)
        loss = criterion(predictions.to(device), true.float().to(device))
        loss.backward()
        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))


def valid_cont(model, iterator, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch):
    model.eval()
    dev_losses = []
    Words, Is_main_piece, Tags, Y, Predictions, Values = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch
            x = x.to(device)
            values = values.to(device)

            predictions, true = model(x, values)
            loss = criterion(predictions.to(device), true.float().to(device))
            dev_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Predictions.extend(predictions.cpu().numpy().tolist())
            Values.extend(values.cpu().numpy().tolist())

    print('Validation loss: {:<.4f}\n'.format(np.mean(dev_losses)))
    final_snapshot_path = 'final_model_{}_testacc_{}_epoch_{}.pt'.format(config.model,
                                                                 np.mean(dev_losses), config.epochs)
    torch.save(model, final_snapshot_path)



def test_cont(model, iterator, criterion, index_to_tag, device, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Predictions, Values = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch
            x = x.to(device)
            values = values.to(device)

            predictions, true = model(x, values)
            loss = criterion(predictions.to(device), true.float().to(device))
            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Predictions.extend(predictions.cpu().numpy().tolist())
            Values.extend(values.cpu().numpy().tolist())

    true = []
    preds_to_eval = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, preds, values in zip(Words, Is_main_piece, Tags, Predictions, Values):
            valid_preds = [p for head, p in zip(is_main_piece, preds) if head == 1]

            predsslice = valid_preds[1:-1]
            valuesslice = values[1:-1]
            wordslice = words.split()[1:-1]

            for w, v, p in zip(wordslice, valuesslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, v, p))
                if v != invalid_set_to:
                    true.append(v)
                    preds_to_eval.append(p)
            results.write("\n")
    # calc metric
    y_true = np.array(true)
    y_pred = np.array(preds_to_eval)

    print('Test loss: {:<.4f}\n'.format(np.mean(test_losses)))
    # Correlation is calculated afterwards with a separate script.
    final_snapshot_path = 'final_model_{}_testacc_{}_epoch_{}.pt'.format(config.model,
                                                                 np.mean(test_losses), config.epochs)
    torch.save(model, final_snapshot_path)


if __name__ == "__main__":
    main()
