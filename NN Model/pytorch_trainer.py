import sys, os, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from IPython.display import clear_output


def clear_any_output(notebook_start: bool):
    if not notebook_start:
        os.system("cls")
    else:
        clear_output(True)
    


class TrainProcessPlotter:
    
    @staticmethod
    def show_results_ipython(train_loss,
                             train_metric,
                             dev_loss=None,
                             dev_metric=None):
        
        train_val_dict = {'Loss': [train_loss, dev_loss], 'Metric': [train_metric, dev_metric]}
        plt.figure(figsize=(16, 4))
        
        for i, key in enumerate(train_val_dict):
            plt.subplot(1, 2, i + 1)
            
            item = train_val_dict[key]
            train_item = item[0]
            plt.title(key)
            plt.plot(train_item, label=f'train ({train_item[-1]:.3})')

            if item[1] is not None:
                val_item = item[1]
                plt.plot(val_item, label=f'dev ({val_item[-1]:.3})')
            plt.xlabel('Epoch #')
            plt.ylabel(key)
            plt.legend()
            plt.grid(ls='--')
        
        plt.show()


class Trainer:
    
    def __init__(self, model, train_loader, val_loader=None, same_shape_tagret=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = None
        self.same_shape_tagret = same_shape_tagret
        
        ppid = os.getppid()
        # self.notebook_start = True if "python" in psutil.Process(ppid).name() else False
               
    def train(self, opt, loss_fn, metric_fn, device, scheduler=None, epochs=15, log=True, plot_every_epoch=True, end_plot=False):
        self.model.to(device)
        self.opt = opt
        train_loss, train_metric = [], []
        val_loss, val_metric = None, None
        
        if self.val_loader is not None:
            val_loss, val_metric = [], []
            
        
        for epoch in range(1, epochs + 1):
            lr = self.opt.param_groups[0]['lr']
            print(f'[ Current lr: {lr}]')
            
            self.model.train()
            if log:
                
                
                if not plot_every_epoch:
                    clear_output(True)
                    # clear_any_output(self.notebook_start)
                print(f'[ Training.. {epoch}/{epochs} ]')
                
                
            train_epoch_loss, train_epoch_metric = self.__epoch_step(opt=opt, 
                                                                     loss_fn=loss_fn, 
                                                                     metric_fn=metric_fn, 
                                                                     device=device,
                                                                     loader=self.train_loader)
            
            train_loss.append(train_epoch_loss)
            train_metric.append(train_epoch_metric)
            
            if self.val_loader is not None:
                
                
                if log:
                    print(f'[ Validating.. {epoch}/{epochs} ]', flush=True)
                val_epoch_loss, val_epoch_metric = self.__validation_step(loss_fn=loss_fn,
                                                                          metric_fn=metric_fn, 
                                                                          device=device)
                val_loss.append(val_epoch_loss)
                val_metric.append(val_epoch_metric)
            
            if scheduler is not None:
                if isinstance(scheduler,  torch.optim.lr_scheduler.ReduceLROnPlateau) and self.val_loader is not None:
                    scheduler.step(val_epoch_metric)
                else:
                    scheduler.step()

            if plot_every_epoch:  
                with torch.no_grad():
                    clear_output(True)
                    # clear_any_output(self.notebook_start)
                    TrainProcessPlotter.show_results_ipython(train_loss, train_metric, val_loss, val_metric)
                    
        if end_plot:  
            with torch.no_grad():
                clear_output(True)
                # clear_any_output(self.notebook_start)
                TrainProcessPlotter.show_results_ipython(train_loss, train_metric, val_loss, val_metric)
            
            
    def __epoch_step(self, loss_fn, metric_fn, device, opt, loader):
        
        epoch_loss = 0.0
        predictions = []
        labels = []
        for i_step, (x, y) in enumerate(loader):
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            prediction = self.model(x)
            
            if self.same_shape_tagret:
                y = y.reshape(prediction.shape)
                
            loss_value = loss_fn(prediction, y)
            
            if opt is not None:
                opt.zero_grad()
                loss_value.backward()
                opt.step()
            
            predictions.append(prediction)
            labels.append(y)
            
            epoch_loss += loss_value.item()
        
        with torch.no_grad():
            num_batches = i_step + 1
            
            predictions, labels = torch.cat(predictions), torch.cat(labels)
            epoch_loss /= num_batches
            epoch_metric = metric_fn(predictions, labels)
            return epoch_loss, epoch_metric


    def __validation_step(self, loss_fn, metric_fn, device):
        
        with torch.no_grad():
            self.model.eval()
            val_loss, val_metric = self.__epoch_step(loss_fn=loss_fn,
                                                     metric_fn=metric_fn,
                                                     device=device,
                                                     opt=None,
                                                     loader=self.val_loader)
            return val_loss, val_metric

        
    def test(self, test_loader, device, metric_fn):
        with torch.no_grad():
            self.model.eval()
            predictions = []
            labels = []

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                prediction = self.model(x)
                predictions.append(prediction)
                labels.append(y)


            predictions, labels = torch.cat(predictions), torch.cat(labels)
            metric = metric_fn(predictions, labels)
            return metric


                    
                    
                    
                    
                    
                    
                    
                    
                
            
        