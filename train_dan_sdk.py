import argparse

import torch
import torch.nn as nn
from torch.utils import data

from data import load_data
from data import DigitsDataset

from networks import DigitsEncoder

from utils import count_acc, Averager

from naie.transfer_learning import DeepTransferLearning

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


def train(model, optimizer, source_loader, target_loader, args):
    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)

    avg_loss = Averager()
    for i in range(1, num_iter):
        sx, sy = iter_source.next()

        try:
            tx, _ = iter_target.next()
        except Exception:
            iter_target = iter(target_loader)
            tx, _ = iter_target.next()

        if args.cuda:
            sx, sy = sx.cuda(), sy.cuda()
            tx = tx.cuda()

        loss = model.forward(sx, tx, sy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.add(loss.item())

    loss = avg_loss.item()
    return loss


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

            logits = model.predict(tx)
            acc = count_acc(logits, ty)
            acc_avg.add(acc)

    acc = acc_avg.item()
    return acc


def main(args):
    source_loader, target_loader = construct_loaders(args)

    encoder = DigitsEncoder(args)
    
    embed_size = encoder.embed_size
    hidden_sizes = [embed_size, 32, 10]
    
    # build DAN model
    model = DeepTransferLearning(
        "DAN", encoder=encoder, hidden_sizes=hidden_sizes,
        loss_fn="ce", lamb=0.01
    )
    
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
        train_loss = train(
            model, optimizer, source_loader, target_loader, args
        )
        train_acc = test(model, source_loader, args)
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
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--cuda", type=int, default=0)
    
    parser.add_argument("--train_url", type=str)
    parser.add_argument("--entry", type=str)
    parser.add_argument("--jp", type=str)
    parser.add_argument("--custom", type=str)
    parser.add_argument("--init_method", type=str)
    parser.add_argument("--data_urll", type=str)
    
    args = parser.parse_args()
    main(args)

