import sys
sys.path.insert(0, './utils/')
sys.path.insert(0, './losses/')
sys.path.insert(0, './layers/')
sys.path.insert(0, './data/')
sys.path.insert(0, './models/')


import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from losses import signer_transfer_loss
from data import KinectLeap, split_data, getSplitter, Triesch
from cnn import CNN
from softCrossEntropy import softCrossEntropyUniform
from cnn_adversial import ADVERSIAL_CNN
from signer_transfer import DEEP_TRANSFER
from utils import (inverse_transform, get_evaluation_protocol)
from torchsummary import summary
from torchvision import transforms
import pickle
from torch.nn import functional as F


# Hyperparameters
MODEL_LIST = {'cnn': CNN, 'transf_cnn': DEEP_TRANSFER}
DATASETS_LIST = {'staticSL': KinectLeap, 'triesch': Triesch}
BATCH_SIZE = 32
LEARNING_RATE = 1e-04
EPOCHS = 500
REG = 1e-04

# layers to regularize
CONV_LAYERS = -1
DENSE_LAYERS = 0

# Cross-entropy
loss_fn = F.cross_entropy


def slit_batch_per_signer(x, y, g_norm, h_conv, h_dense, y_task, n_signers):
    """split data per signer identity

    Parameters:
    x (type): batch of data
    y (type): batch of gesture labels
    g_norm (type): batch of signer iodentities labels
    h_conv (type): activations of conv layers
    h_dense (type): activations of dense layers
    y_task (type): class labels predictions
    n_signers (type): number of training signer identities

    Returns:
    x_split (type): x splitted by signer identity
    y_split (type): y splitted by signer identity
    g_split (type): g splitted by signer identity
    h_conv_split (type): h_conv splitted by signer identity
    h_dense_split (type): h_dense splitted by signer identity
    y_task_split (type): y_task splitted by signer identity

    """
    x_split = [False]*n_signers
    y_split = [False]*n_signers
    g_split = [False]*n_signers
    y_task_split = [False]*n_signers
    h_conv_split = [False]*n_signers
    h_dense_split = [False]*n_signers

    for s in range(n_signers):
        x_split[s] = x[g_norm == s]
        y_split[s] = y[g_norm == s]
        g_split[s] = g_norm[g_norm == s]

        h_conv_split[s] = [torch.mean(h[g_norm == s], dim=0)
                           for h in h_conv[CONV_LAYERS:]]
        h_dense_split[s] = [torch.mean(h[g_norm == s], dim=0)
                            for h in h_dense[DENSE_LAYERS:]]
        y_task_split[s] = y_task[g_norm == s]

    return x_split, y_split, g_split, h_conv_split, h_dense_split, y_task_split


def fit(model, data, device, output, n_signers):
    """Training routine

    Parameters:
    model (type): pytorch model
    data (type): dataloaders(a tuple with both training and validation loaders)
    device (type): cpu or gpu id if available
    n_signers (type): number of training signer identities

    Returns:
    model (type): trained model
    train_history (type): training history
    valid_loader (type): validation dataloader

    """

    global TRANSFER_WEIGHT, ADV_WEIGHT

    #  train and validation loaders
    train_loader, valid_loader = data
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))

    # Set the optimizers
    # task-specific optimizer
    task_opt = torch.optim.Adam(list(model.feat_extractor.parameters()) +
                                list(model.dense_block.parameters()) +
                                list(model.task_classifier.parameters()),
                                lr=LEARNING_RATE,
                                weight_decay=REG)

    # adversarial-specific optimizer
    adv_opt = torch.optim.Adam(list(model.adversial_classifier.parameters()),
                               lr=LEARNING_RATE,
                               weight_decay=REG)

    # Start training
    train_history = {'tr_loss': [], 'tr_task_loss': [], 'tr_transf_loss': [],
                     'tr_adv_loss': [], 'tr_ce_uniform_loss': [], 'tr_acc': [],
                     'val_task_loss': [], 'val_acc': []}

    # Best validation params
    best_val = float('inf')
    best_epoch = 0

    print('>> ADV_WEIGHT: ', ADV_WEIGHT)
    print('>> TRANSFER_WEIGHT: ', TRANSFER_WEIGHT)

    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        for i, (x, y, g, r, g_norm) in enumerate(train_loader):
            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)
            r = r.to(device)
            g_norm = g_norm.to(device)

            # forward pass
            h_conv, h_dense, y_task, y_adversial = model(x)

            # split batch per signer
            (x_split,
             y_split,
             g_split,
             h_conv_split,
             h_dense_split,
             y_task_split) = slit_batch_per_signer(x, y, g_norm,
                                                   h_conv, h_dense, y_task,
                                                   n_signers)

            signers_on_batch = [i for i in range(len(g_split))
                                if len(g_split[i])]

            # compute signer-transfer loss
            transfer_loss = signer_transfer_loss(h_conv_split, h_dense_split,
                                                 signers_on_batch)

            # Compute task-specific loss
            task_loss = loss_fn(y_task, y)

            # Compute adversial loss
            adv_loss = loss_fn(y_adversial, g_norm)

            # total loss
            # loss = task_loss - \
            #     ADV_WEIGHT*adv_loss + \
            #     transfer_loss*TRANSFER_WEIGHT

            loss = task_loss + \
                ADV_WEIGHT*softCrossEntropyUniform(y_adversial) + \
                transfer_loss*TRANSFER_WEIGHT

            # Backprop and optimize
            # task-specific step
            task_opt.zero_grad()
            loss.backward(retain_graph=True)
            task_opt.step()

            # adversial step
            adv_opt.zero_grad()
            adv_loss.backward()
            adv_opt.step()

            # display the mini-batch loss
            print('........{}/{} mini-batch loss: {:.3f} |'
                  .format(i + 1, len(train_loader), loss.item()) +
                  ' task_loss: {:.3f} |'
                  .format(task_loss.item()) +
                  ' transfer_loss: {:.3f} |'
                  .format(transfer_loss.item()) +
                  ' adv_loss: {:.3f} |'
                  .format(adv_loss.item()), flush=True, end='\r')

        # Validation
        (tr_loss, tr_task_loss, tr_transf_loss, tr_adv_loss,
         tr_ce_uniform_loss, tr_acc, tr_acc_3, tr_acc_5) = eval_model(model, train_loader, n_signers, device)
        train_history['tr_loss'].append(tr_loss.item())
        train_history['tr_task_loss'].append(tr_task_loss.item())
        train_history['tr_transf_loss'].append(tr_transf_loss.item())
        train_history['tr_adv_loss'].append(tr_adv_loss.item())
        train_history['tr_ce_uniform_loss'].append(tr_ce_uniform_loss.item())
        train_history['tr_acc'].append(tr_acc)

        (_, val_task_loss, _, _, _,
         val_acc, val_acc_3, val_acc_5) = eval_model(model, valid_loader, n_signers, device, debug=True)
        train_history['val_task_loss'].append(val_task_loss.item())
        train_history['val_acc'].append(val_acc)

        # save best validation model
        if best_val > val_task_loss:
            torch.save(model.state_dict(), os.path.join(*(output, 'cnn.pth')))
            best_val = val_task_loss
            best_epoch = epoch

        # display the training loss
        print()
        print('>> Train loss: {:.5f} |'.format(tr_loss.item()) +
              ' tr_task_loss: {:.5f} |'.format(tr_task_loss.item()) +
              ' tr_transf_loss: {:.5f} |'.format(tr_transf_loss.item()) +
              ' tr_adv_loss: {:.5f} |'.format(tr_adv_loss.item()) +
              ' tr_ce_uniform_loss: {:.5f}'.format(tr_ce_uniform_loss.item()) +
              ' Train Acc: {:.5f}'.format(tr_acc))

        print('>> Valid loss: {:.5f} |'.format(val_task_loss.item()) +
              ' Valid Acc: {:.5f} |'.format(val_acc))
        print('>> Best model: {}/{:.5f}'.format(best_epoch+1, best_val))
        print()

    # save train/valid history
    plot_fn = os.path.join(*(output, 'cnn_history.png'))
    plot_train_history(train_history, plot_fn=plot_fn)

    # return best validation model
    model.load_state_dict(torch.load(os.path.join(*(output, 'cnn.pth'))))

    return model, train_history, valid_loader


def plot_train_history(train_history, plot_fn=None):
    """plot or save training history

    Parameters:
    train_history (type): dictionary with training losses
    plot_fn (type): path for saving training plot

    Returns:
    None

    """

    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    best_val_epoch = np.argmin(train_history['val_task_loss'])
    best_val_acc = train_history['val_acc'][best_val_epoch]
    best_val_loss = train_history['val_task_loss'][best_val_epoch]
    plt.figure(figsize=(7, 5))
    epochs = len(train_history['tr_loss'])
    x = range(epochs)
    plt.subplot(511)
    plt.plot(x, train_history['tr_loss'], 'r-')
    plt.xlabel('Epoch')
    plt.legend(['tr_loss'])
    plt.axis([0, epochs, min(train_history['tr_loss']), max(train_history['tr_loss'])])
    plt.subplot(512)
    plt.plot(x, train_history['tr_transf_loss'], 'r-')
    plt.xlabel('Epoch')
    plt.legend(['tr_transf_loss'])
    plt.axis([0, epochs, min(train_history['tr_transf_loss']), max(train_history['tr_transf_loss'])])
    plt.subplot(513)
    plt.plot(x, train_history['tr_adv_loss'], 'r-')
    plt.plot(x, train_history['tr_ce_uniform_loss'], 'y--')
    plt.xlabel('Epoch')
    plt.legend(['tr_adv_loss', 'tr_ce_uniform_loss'])
    plt.axis([0, epochs, 0, 2.0])
    plt.subplot(514)
    plt.plot(x, train_history['tr_task_loss'], 'r--')
    plt.plot(x, train_history['val_task_loss'], 'g--')
    plt.plot(best_val_epoch, best_val_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val loss')
    plt.legend(['tr_task_loss', 'val_task_loss'])
    plt.axis([0, epochs, 0, max(train_history['tr_task_loss'])])
    plt.subplot(515)
    plt.plot(x, train_history['tr_acc'], 'r-')
    plt.plot(x, train_history['val_acc'], 'g-')
    plt.plot(best_val_epoch, best_val_acc, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val acc')
    plt.legend(['train_acc', 'val_acc'])
    plt.axis([0, epochs, 0, 1])
    if plot_fn:
        plt.show()
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


def eval_model(model, data_loader, n_signers, device, debug=False):
    """Validation and testing routine

    Parameters:
    model (type): pytorch model
    data_loader:dataloaders (a tuple with both training and validation loaders)
    n_signers (type): number of training signer identities
    device (type): cpu or gpu id if available
    debug (type): debug flag

    Returns:
    loss_eval (type): total loss
    task_loss_eval (type): task-specific loss
    transf_loss_eval (type): signer-transfer loss
    adv_loss_eval (type): signer loss
    CE_unif_loss_eval (type): adversarial loss
    acc (type): classification accuracy
    acc_3 (type): top-3 accuracy
    acc_5 (type): top-5 accuracy

    """

    global TRANSFER_WEIGHT, ADV_WEIGHT

    with torch.no_grad():
        # set model to train
        model.eval()
        loss_eval = 0
        task_loss_eval = 0
        transf_loss_eval = 0
        adv_loss_eval = 0
        CE_unif_loss_eval = 0
        N = 0
        n_correct = 0
        n_correct_3 = 0
        n_correct_5 = 0
        for i, (x, y, g, r, g_norm) in enumerate(data_loader):
            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)
            r = r.to(device)
            g_norm = g_norm.to(device)

            # mask = r != 2
            # x = x[mask]
            # y = y[mask]
            # g = g[mask]
            # g_norm = g_norm[mask]

            # forward pass
            h_conv, h_dense, y_task, y_adversial = model(x)

            # split batch per signer
            (x_split,
             y_split,
             g_split,
             h_conv_split,
             h_dense_split,
             y_task_split) = slit_batch_per_signer(x, y, g_norm,
                                                   h_conv, h_dense, y_task,
                                                   n_signers)

            signers_on_batch = [i for i in range(len(g_split))
                                if len(g_split[i])]

            # compute signer-transfer loss
            transfer_loss = 0
            if (not debug) or (len(signers_on_batch) > 1):
                transfer_loss = signer_transfer_loss(h_conv_split,
                                                     h_dense_split,
                                                     signers_on_batch)
            # else:
            #     plt.switch_backend('TkAgg')
            #     for iii in x:
            #         plt.figure()
            #         plt.subplot(111)
            #         plt.imshow(inverse_transform((iii)))
            #         plt.axis('off')
            #         plt.show()

            # Compute task-specific loss
            task_loss = loss_fn(y_task, y)

            # Compute adversial loss
            adv_loss = 0
            if not debug:
                adv_loss = loss_fn(y_adversial, g_norm)

            # classification loss
            task_loss = loss_fn(y_task, y)

            # total loss
            loss = task_loss + \
                ADV_WEIGHT*softCrossEntropyUniform(y_adversial) + \
                transfer_loss*TRANSFER_WEIGHT

            # Compute task-specific loss
            loss_eval += loss * x.shape[0]
            task_loss_eval += task_loss * x.shape[0]
            transf_loss_eval += transfer_loss * x.shape[0]
            adv_loss_eval += adv_loss * x.shape[0]
            CE_unif_loss_eval += softCrossEntropyUniform(y_adversial) * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_task, dim=1)
            # print(ypred_)
            n_correct += torch.sum(1.*(ypred_ == y)).item()

            # top-N accruracy
            ypred_5 = torch.argsort(y_task, dim=1)[:, -5:]
            n_correct_5 += torch.sum(1.*(ypred_5 == y.unsqueeze(1))).item()
            ypred_3 = torch.argsort(y_task, dim=1)[:, -3:]
            n_correct_3 += torch.sum(1.*(ypred_3 == y.unsqueeze(1))).item()

        loss_eval = loss_eval / N
        task_loss_eval = task_loss_eval / N
        transf_loss_eval = transf_loss_eval / N
        adv_loss_eval = adv_loss_eval / N
        CE_unif_loss_eval = CE_unif_loss_eval / N
        acc = n_correct / N
        acc_3 = n_correct_3 / N
        acc_5 = n_correct_5 / N
        return (loss_eval, task_loss_eval, transf_loss_eval, adv_loss_eval,
                CE_unif_loss_eval, acc, acc_3, acc_5)


def main():
    global ADV_WEIGHT, TRANSFER_WEIGHT

    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    # Parsing arguments
    parser = argparse.ArgumentParser(description='signer-independent project')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--adv_weight', type=float, required=True)
    parser.add_argument('--transf_weight', type=float, required=True)
    parser.add_argument('--output', default='./output_cnn/')

    args = parser.parse_args()

    # set adversarial and transfer weights
    TRANSFER_WEIGHT = args.transf_weight
    ADV_WEIGHT = args.adv_weight

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = DATASETS_LIST[args.dataset](model=args.model)
    X_to_split = np.zeros((len(dataset), 1))
    print(len(dataset[0]))

    # evaluation protocol
    (IM_SIZE, MODE, SPLITS,
     n_signers, n_classes) = get_evaluation_protocol(args.dataset)

    # get data splitter
    dataSplitter = getSplitter(dataset, n_splits=SPLITS, mode=MODE,
                               test_size=.10)

    results = []
    split = 0

    for split, (tr_indexes, test_indexes) in enumerate(dataSplitter):
        output_fn = os.path.join(args.output, 'split_' + str(split))

        if not os.path.isdir(output_fn):
            os.mkdir(output_fn)

        # split data
        (train_loader,
         valid_loader,
         test_loader) = split_data(dataset,
                                   (tr_indexes, test_indexes),
                                   BATCH_SIZE,
                                   dataAug=True,
                                   mode=MODE)

        # Initialize the model
        model = MODEL_LIST[args.model](input_shape=IM_SIZE,
                                       output_signers=n_signers,
                                       output_classes=n_classes,
                                       hasAdversial=True).to(device)
        print(model)

        # Train or test
        if args.mode == 'train':
            # Fit model
            model, train_history, valid_loader = fit(model=model,
                                                     data=(train_loader,
                                                           valid_loader),
                                                     device=device,
                                                     output=output_fn,
                                                     n_signers=n_signers)

            # save train history
            res_fn = os.path.join(*(output_fn, '_history.pckl'))
            pickle.dump(train_history, open(res_fn, "wb"))

        elif args.mode == 'test':
            model.load_state_dict(torch.load(
                                  os.path.join(*(output_fn, 'cnn.pth'))))

            # load train history
            res_fn = os.path.join(*(output_fn, '_history.pckl'))
            train_history = pickle.load(open(res_fn, "rb"))
            plot_fn = os.path.join(*(output_fn, 'cnn_history.png'))
            plot_train_history(train_history, plot_fn=plot_fn)

        # Test results
        (_, test_loss, _, _, _,
         test_acc, test_acc_3, test_acc_5) = eval_model(model, test_loader,
                                                        n_signers, device,
                                                        debug=True)
        print('##!!!! Test loss: {:.5f} |'.format(test_loss.item()) +
              ' Test Acc: {:.5f}'.format(test_acc))

        results.append((test_loss.item(), test_acc, test_acc_3, test_acc_5))

        # TSNE maps
        # tsne(model, test_loader, device,
        #      plot_fn=os.path.join(*(output_fn, 'tsne.png')))

    # save results
    print(results)
    # asdas
    res_fn = os.path.join(args.output, 'res.pckl')
    pickle.dump(results, open(res_fn, "wb"))
    results = pickle.load(open(res_fn, "rb"))

    # Compute average and std
    print(results)
    acc_array = np.array([i[1] for i in results])
    acc3_array = np.array([i[2] for i in results])
    acc5_array = np.array([i[3] for i in results])
    print('Average acc: ', np.mean(acc_array))
    print('Average acc3: ', np.mean(acc3_array))
    print('Average acc5: ', np.mean(acc5_array))
    print('Std acc: ', np.std(acc_array))


if __name__ == '__main__':
    main()
