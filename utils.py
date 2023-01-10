import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, train_loss, model):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
            self.counter = 0

    def save_checkpoint(self, train_loss, model):
        '''Saves model when train loss decrease.'''
        if self.verbose:
            self.trace_func(f'Train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.train_loss_min = train_loss

""""""

def train(model, flow_train_loader, link_train_loader, criteon, optimizer, early_stopping,\
          n_epochs, check_point_name, check_point_last_name):

    for epoch in range(n_epochs):
        train_losses = [] 
        y_iter_train = iter(flow_train_loader)

        for batch_id, x in enumerate(link_train_loader):
            y = next(y_iter_train)
            x_hat = model(x)
            loss = criteon(x_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
           
        train_loss = np.average(train_losses)       
        print(epoch, 'train_loss:', train_loss)
              
        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), check_point_last_name)
    model.load_state_dict(torch.load(check_point_name))
    
    return model


def test(model, flow_test_loader, link_test_loader, criteon):

    test_losses = []
    predict_flows = np.empty([0, flow_test_loader.dataset.shape[1]])
    y_iter_test = iter(flow_test_loader)

    for batch_id, x in enumerate(link_test_loader):
        with torch.no_grad():
            x_hat = model(x)
            predict_flows = np.row_stack([predict_flows, x_hat.cpu().numpy()])
            y = next(y_iter_test)
            test_loss = criteon(x_hat, y)

        test_losses.append(test_loss.item())
    
    test_loss = np.average(test_losses)
    print('test_loss:', test_loss.item())

    return predict_flows


""""""

def autotomo_loss(true_id, flow_n, h_pre, h_true, x_true, criteon, rm):

    all_flows = np.arange(flow_n)
    false_id = np.setdiff1d(all_flows, true_id)
    
    h_pre_cat = torch.zeros_like(h_true)
    h_pre_cat[:,false_id] = h_pre[:,false_id]
    
    h_pre_cat[:,true_id] = h_true[:,true_id]
       
    loss1 = criteon(h_pre[:, true_id], h_true[:, true_id])    
    loss2 = criteon(h_pre_cat @ rm, x_true)

    final_loss = loss1 + loss2

    return final_loss


def auto_tomo_train(model, flow_train_loader, link_train_loader, 
                    criteon, optimizer, early_stopping, n_epochs, known_train_id,
                    check_point_name, last_point_name, rm, device):

    flow_n = flow_train_loader.dataset.shape[1]

    for epoch in range(n_epochs):
        train_losses = []
        y_iter_train = iter(flow_train_loader)

        for batch_id, x in enumerate(link_train_loader):
            x = x.to(device)
            y = next(y_iter_train)
            h_hat = model(x)
            loss = autotomo_loss(known_train_id, flow_n, h_hat, y, x, criteon, rm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.average(train_losses)
        print(epoch, 'train_loss:', train_loss)

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), last_point_name)
    model.load_state_dict(torch.load(check_point_name))

    return model


def auto_tomo_test(model, flow_test_loader, link_test_loader, criteon):

    test_losses_y = []
    predict_flow = np.empty([0, flow_test_loader.dataset.shape[1]])
    y_iter_test = iter(flow_test_loader)

    for batch_id, x in enumerate(link_test_loader):
        with torch.no_grad():
            y = next(y_iter_test)
            h_hat = model(x)
            predict_flow = np.row_stack([predict_flow, h_hat.cpu().numpy()])
            test_loss_y = criteon(h_hat, y)

        test_losses_y.append(test_loss_y.item())

    test_loss = np.average(test_losses_y)
    print('test_loss:', test_loss.item())

    return predict_flow