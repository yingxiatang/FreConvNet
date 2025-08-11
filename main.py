import torch
import time

from dataset import Dataset
from process import Trainer
from FreConvNet import Model
from args import args
import torch.utils.data as Data



def main():
    torch.set_num_threads(2)
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
    start_time = time.time()

    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("the execution time of program", execution_time, "seconds")
