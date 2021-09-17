import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


learning_curves = pd.read_csv('learning_curves.csv')

# Loss figure:
plt.figure(figsize=(20, 10))
plt.plot(range(learning_curves['epoch']), learning_curves['loss_train'], label='train')
plt.plot(range(learning_curves['epoch']), learning_curves['loss_val'], label='val')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
plt.grid()
plt.savefig('loss_plot.png', bbox_inches='tight')

# metric figure:
train_avg_metric = [np.mean(i) for i in learning_curves['metric_train']]
val_avg_metric = [np.mean(i) for i in learning_curves['metric_val']]

plt.figure(figsize=(20, 10))
plt.plot(range(learning_curves['epoch']), train_avg_metric, label='train')
plt.plot(range(learning_curves['epoch']), val_avg_metric, label='val')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Avg metric', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
plt.grid()
plt.savefig('metric_plot.png')#, bbox_inches='tight')