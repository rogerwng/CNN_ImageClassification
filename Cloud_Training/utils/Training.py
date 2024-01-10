# training loops and early stopper

import torch
from tqdm import tqdm


# early stopping class
# criterion: validation accuracy
class EarlyStop():
    def __init__(self, patience, delta=0.003):
        self.patience = patience
        self.delta = delta
        self.best = 0
        self.count = 0

    def checkForStop(self, acc):
        if acc > self.best:
            self.best = acc
            self.count = 0
            return False
        elif acc <= self.best-self.delta:
            if self.count > self.patience:
                return True
            self.count = self.count + 1
            return False
        
    def reset(self):
        self.best = 0
        self.count = 0
        
# basic training loop, constant learning rate, 'cuda'
def train(model, trainloader, valloader, epochs, delta=0.003, top_3 = False):
    # prepare model
    model.initOptimizer()
    cuda = torch.device('cuda')
    model.to(cuda)

    # initialize early stopping
    stopper = EarlyStop(3,delta)

    # training loop w/ pbar
    pbar = tqdm(range(epochs))
    for i in range(epochs):
        # track train, val scores
        train_loss = 0
        val_loss = 0
        val_acc = 0
        val_acc_3 = 0

        model.train()
        # training over batches of train set
        for n, (traindata,trainlabel) in enumerate(trainloader):
            traindata, trainlabel = traindata.to(cuda), trainlabel.to(cuda)
            model.optim.zero_grad()
            logits = model.forward(traindata)
            loss = model.loss((logits,trainlabel))
            loss.backward()
            model.optim.step()

            train_loss += loss.item()
        train_loss /= len(trainloader)

        # validation over batches of val set
        model.eval()
        for n, (val,vallabel) in enumerate(valloader):
            val, vallabel = val.to(cuda), vallabel.to(cuda)
            with torch.no_grad():
                logits = model.forward(val)
                loss = model.loss((logits,vallabel))
                accuracy = model.accuracy((logits,vallabel))
                accuracy_3 = model.accuracy_3((logits,vallabel)) if top_3 else 0

                val_loss += loss.item()
                val_acc += accuracy
                val_acc_3 += accuracy_3
        val_loss /= len(valloader)
        val_acc /= len(valloader)
        val_acc_3 /= len(valloader)

        # update pbar
        pbar.update(1)
        if top_3:
            pbar.set_description(f'val_acc: {val_acc:.3f} val_acc_3: {val_acc_3:.3f} train_loss: {train_loss:.3f}')
        else:
            pbar.set_description(f'val_acc: {val_acc:.3f} train_loss: {train_loss:.3f}')

        # check for early stopping, stop when train_loss = zero since no further training will happen
        if stopper.checkForStop(val_acc) or train_loss <= 0.0003:
            break

    pbar.close()

# training loop with learning rate scheduler, reduce learning rate by 10% when model stops improving until final lr < initial*0.7, max 200 epochs
# instead of using this function, just call train several times in main script, early stopper will move onto next train loop automatically
def train_schedule(model,trainloader,valloader,delta=0):
    # prepare model
    model.initOptimizer()
    cuda = torch.device('cuda')
    model.to(cuda)

    # initialize early stopping and initialLR
    stopper = EarlyStop(4,delta)
    initialLR = model.lr
    epoch = 0

    # train loop
    while model.lr > initialLR * 0.7 and epoch < 200:
        # track train/val scores
        train_loss = 0
        val_loss = 0
        val_acc = 0

        model.train()
        # training over batches of train set
        for n, (traindata,trainlabel) in enumerate(trainloader):
            traindata, trainlabel = traindata.to(cuda), trainlabel.to(cuda)
            model.optim.zero_grad()
            logits = model.forward(traindata)
            loss = model.loss((logits,trainlabel))
            loss.backward()
            model.optim.step()

            train_loss += loss.item()
        train_loss /= len(trainloader)

        # validation over batches of val set
        model.eval()
        for n, (val,vallabel) in enumerate(valloader):
            val, vallabel = val.to(cuda), vallabel.to(cuda)
            with torch.no_grad():
                logits = model.forward(val)
                loss = model.loss((logits,vallabel))
                accuracy = model.accuracy((logits,vallabel))

                val_loss += loss.item()
                val_acc += accuracy
        val_loss /= len(valloader)
        val_acc /= len(valloader)

        # update progress
        print(f'val_acc: {val_acc:.3f} train_loss: {train_loss:.3f} epoch: {epoch+1} lr: {model.lr}')

        if stopper.checkForStop(val_acc):
            model.lr *= 0.9
            stopper.rest()

        epoch += 1