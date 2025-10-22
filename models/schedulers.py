import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def scheduler(epoch, lr):
    if epoch%500 == 0:
        return lr*tf.math.exp(-0.1)
    else:
        return lr

class SchedulerandTrackerCallback(Callback):
    def __init__(self,scheduler):
        self.scheduler = scheduler
        self.epoch_lr = []
        self.epoch_loss = []
        
    def on_epoch_begin(self,epoch,logs = None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        new_lr = self.scheduler(epoch,current_lr)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
    def on_epoch_end(self,epoch,logs = None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        loss = logs.get('loss')
        self.epoch_lr.append(current_lr)
        self.epoch_loss.append(loss)