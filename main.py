import torch
import time

from data_provider.dataset import Dataset
from util.process import Trainer
from model.FreConvNet import Model
from args import args
import torch.utils.data as Data



def main():
    torch.set_num_threads(6)
    train_dataset = Dataset(device=args.device, mode='train')
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    print('train',train_loader)
    args.data_shape = train_dataset.shape()                # data_shape
    test_dataset = Dataset(device=args.device, mode='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('data_shape_origin', args.data_shape)
    print('dataset initial ends')

    model = Model(args)  # modernTCN

    print('model initial ends')
    trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
    trainer.train()


if __name__ == '__main__':
    main()
