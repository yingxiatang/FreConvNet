import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from util.loss import CE, BCE
from torch.optim.lr_scheduler import LambdaLR
from util.optimizers import get_optimizer


class Trainer():
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.train_optim = get_optimizer(args.train_optim)

        self.cr = CE(self.model) if args.loss == 'ce' else BCE(self.model)
        self.folder = args.dataset_name

        self.test_cr = torch.nn.CrossEntropyLoss() if args.loss == 'ce' else torch.nn.BCELoss()
        self.num_epoch = args.num_epoch
        self.eval_per_steps = args.eval_per_steps
        self.all_save = args.save_path
        self.save_path = args.save_path+'/'+args.dataset_name
        # if self.num_epoch:
        #     self.result_file = open(self.save_path + '/result.txt', 'w')
        #     self.result_file = open(self.save_path + '/result.txt', 'w')
        #     self.result_file.close()

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'

    def train(self):
        self.optimizer = self.train_optim(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        loss_train_epoch = []
        loss_test_epoch = []
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost,test_loss = self._train_one_epoch()
            loss_train_epoch.append(loss_epoch)
            loss_test_epoch.append(test_loss)
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            # self.all_result_file = open(self.save_path+'/acc.txt', 'a+')
            self.print_process(
                 'Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        # self.result_file = open(self.save_path + '/result.txt', 'a+')
        self.result_file = open(self.all_save + '/result.txt', 'a+')
        print(f'{self.folder}  best_acc', self.best_metric, file = self.result_file)
        # print('train_loss with epoch', loss_train_epoch, file = self.result_file)
        # print('test_loss with epoch', loss_test_epoch, file = self.result_file)
        self.result_file.close()
        self.print_process(self.best_metric)
        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader   # self.verbose是否使用tqdm包装来显示进度条

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]   # 将批数据移动到指定设备

            self.optimizer.zero_grad()     # 梯度清零
            loss = self.cr.compute(batch)
            loss_sum += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)    # 限制梯度范数防止爆炸
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                metric = self.eval_model()            # metric = {'acc', 'f1', 'micro_f1', 'test_loss'}
                self.print_process(metric)
                ############
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                if metric[self.metric] >= self.best_metric:
                    if self.args.save_model:
                        torch.save(self.model, self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric[self.metric]
                self.model.train()

        return loss_sum / (idx + 1), time.perf_counter() - t0, metric['test_loss']

    def eval_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        confusion_mat = self._confusion_mat(label, pred)
        self.print_process(confusion_mat)
        self.result_file = open(self.save_path + '/result.txt', 'a+') #######
        print(confusion_mat, file=self.result_file)         #
        self.result_file.close()
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred)
            metrics['precision'] = precision_score(y_true=label, y_pred=pred)
            metrics['recall'] = recall_score(y_true=label, y_pred=pred)
        else:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        if len(batch) == 2:
            seqs, label = batch
            scores = self.model(seqs)
        else:
            seqs1, seqs2, label = batch
            scores = self.model((seqs1, seqs2))
        if self.args.loss == 'ce':
            _, pred = torch.topk(scores, 1)
            test_loss = self.test_cr(scores, label.view(-1).long())
            pred = pred.view(-1).tolist()
            return pred, label.tolist(), test_loss
        else:
            pred = (scores > self.threshold).int().view(-1).tolist()
            test_loss = self.test_cr(scores.view(-1), label.view(-1).float())
            return pred, label.tolist(), test_loss

    def _confusion_mat(self, label, pred):
        if self.args.loss == 'ce':
            mat = np.zeros((self.args.num_class, self.args.num_class))
        else:
            mat = np.zeros((2, 2))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)
