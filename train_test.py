import time
import random
import torch
import torch.nn.functional as F
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(args, model, optimizer, train_loader, val_loader, test_loader, i_fold):
    """

    :param train_loader:
    :param model: model
    :type optimizer: Optimizer

    """
    min_loss = 1e10
    patience = 0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            data = data.to(args.device)
            out, y_b, lam = model(data, mixup=True, alpha=args.alpha)
            loss = lam * F.nll_loss(out, data.y) + (1 - lam) * F.nll_loss(out, y_b)
            # out = model(data)
            # loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_acc, val_loss = test_model(args, model, val_loader)
        test_acc, test_loss = test_model(args, model, test_loader)

        print('Epoch: {:04d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss),
              'val_loss: {:.6f}'.format(val_loss), 'val_acc: {:.6f}'.format(val_acc),
              'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc),
              'time: {:.6f}s'.format(time.time() - t))

        if val_loss < min_loss:
            torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i_fold))
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if patience == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t0))

    return best_epoch


def test_model(args, model, loader):
    model.eval()
    correct = 0.
    test_loss = 0.
    for data in loader:
        with torch.no_grad():
            data = data.to(args.device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            test_loss += F.nll_loss(out, data.y).item()
    test_acc = correct / len(loader.dataset)
    return test_acc, test_loss
