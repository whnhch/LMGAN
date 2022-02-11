import time, datetime
import sklearn.metrics as metrics
import pandas as pd

def metric(predictions, true_labels):
    accuracy = metrics.accuracy_score(true_labels, predictions)
    precision = metrics.precision_score(true_labels, predictions, average='micro')
    recall = metrics.recall_score(true_labels, predictions, average='micro')
    f1_micro = metrics.f1_score(true_labels, predictions, average='micro')
    f1_macro = metrics.f1_score(true_labels, predictions, average='macro')

    return accuracy, precision, recall, f1_micro, f1_macro

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def print_loss(step, len_train_dataloader, elapsed, losses):
    print('  [{:>5,}/{:>5,}]    Elapsed: {:}'.format(step,len_train_dataloader,elapsed))

    for loss in losses:
        loss_str=''
        for key in loss.keys():
            val = sum(loss[key][: step]) / step
            loss_str += key
            loss_str += ': '
            loss_str += str(val)
            loss_str += ' '

        print(loss_str)

 # sum(g_total_loss[: (step)]) / (step),
 # sum(d_total_loss[:step]) / step))

def save_loss(model_name, epoch_i, losses):
    for loss in losses:
        for loss_name in loss.keys():
            val = loss[loss_name]
            df_result = pd.DataFrame(val)
            df_result.to_csv('./'+model_name+loss_name+str(epoch_i)+'.csv', header=None, index=False)
