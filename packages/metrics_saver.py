import pandas


def update_metrics(model_name, validation_acc, validation_loss, train_accuracy, train_loss, precision, specificity, f1):
    try:
        metrics_df = pandas.read_csv('./models/' + model_name + '/metrics.csv')
    except:
        print('no metrics file found! creating a new one')
        metrics_df = pandas.DataFrame(
            0, columns=['validation_acc', 'validation_loss', 'train_accuracy', 'train_loss', 'precision', 'specificity', 'f1-score'], index=range(0))
    index = len(metrics_df)

    metrics_df.loc[index, ['validation_acc']] = validation_acc
    metrics_df.loc[index, ['validation_loss']] = validation_loss

    metrics_df.loc[index, ['train_accuracy']] = train_accuracy
    metrics_df.loc[index, ['train_loss']] = train_loss

    metrics_df.loc[index, ['precision']] = precision
    metrics_df.loc[index, ['specificity']] = specificity
    metrics_df.loc[index, ['f1-score']] = f1

    metrics_df.fillna(0, inplace=True)
    metrics_df.to_csv('./models/' + model_name + '/metrics.csv', index=False)


def get_metrics(model_name):
    metrics_df = pandas.read_csv('./models/' + model_name + '/metrics.csv')
    return metrics_df
