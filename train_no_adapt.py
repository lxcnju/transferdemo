import argparse

import torch
import torch.nn as nn
from torch.utils import data

from data import load_data
from data import DigitsDataset

from networks import DigitsNet

from utils import count_acc, Averager


def construct_loaders(args):
    train_imgs, train_labels = load_data(args.source)
    train_set = DigitsDataset(
        train_imgs, train_labels, is_train=True
    )

    train_loader = data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )
    
    target_imgs, target_labels = load_data(args.target)
    target_test_set = DigitsDataset(
        target_imgs, target_labels, is_train=False
    )

    target_loader = data.DataLoader(
        target_test_set, batch_size=args.batch_size,
        shuffle=False, drop_last=False
    )
    return train_loader, target_loader


def train(model, optimizer, source_loader, args):
    model.train()

    acc_avg = Averager()
    loss_avg = Averager()
    
    iter_source = iter(source_loader)
    num_iter = len(iter_source)
    for i in range(num_iter):
        sx, sy = iter_source.next()

        if args.cuda:
            sx, sy = sx.cuda(), sy.cuda()

        _, logits = model(sx)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, sy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = count_acc(logits, sy)
        acc_avg.add(acc)
        loss_avg.add(loss.item())

    acc = acc_avg.item()
    loss = loss_avg.item()
    return loss, acc

def test(model, target_loader, args):
    model.eval()

    acc_avg = Averager()
    with torch.no_grad():
        iter_target = iter(target_loader)
        num_iter = len(iter_target)
        for i in range(num_iter):
            tx, ty = iter_target.next()

            if torch.cuda.is_available():
                tx, ty = tx.cuda(), ty.cuda()

            _, logits = model(tx)
            acc = count_acc(logits, ty)
            acc_avg.add(acc)

    acc = acc_avg.item()
    return acc


def main(args):
    source_loader, target_loader = construct_loaders(args)

    model = DigitsNet(args)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    
    for epoch in range(args.max_epoch):
        train_loss, train_acc = train(
            model, optimizer, source_loader, args
        )
        test_acc = test(model, target_loader, args)
        print("[Epoch:{}] [TrLoss:{:.5f}] [TrAcc:{:.3f}] [TeAcc:{:.3f}]".format(
            epoch, train_loss, train_acc, test_acc
        ))
        lr_scheduler.step()

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="mnist")
    parser.add_argument("--target", type=str, default="svhn")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--cuda", type=int, default=1)
    
    parser.add_argument("--train_url", type=str)
    parser.add_argument("--entry", type=str)
    parser.add_argument("--jp", type=str)
    parser.add_argument("--custom", type=str)
    parser.add_argument("--init_method", type=str)
    parser.add_argument("--data_urll", type=str)
    
    args = parser.parse_args()
    main(args)
