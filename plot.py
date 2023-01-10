import matplotlib.pyplot as plt
import numpy as np


def sp_mse(true_test_flow, pred_test_flow):
    """
    Compute mean square error of every OD pair.

    :param true_test_flow: real OD flows
    :param pred_test_flow: predicted OD flows
    :return: MSE of every OD pair
    """
    tn, fn = true_test_flow.shape 
    se = np.square(true_test_flow-pred_test_flow)
    mse = np.mean(se, axis=0)
    return np.arange(fn), mse


def tm_mse(true_test_flow, pred_test_flow, pat=12):
    """
    Compute mean square error of every time batch.

    :param true_test_flow: real OD flows
    :param pred_test_flow: predicted OD flows
    :param pat: time batch size
    :return: MSE of every time batch
    """
    tn, fn = true_test_flow.shape
    true_flows_sum = np.zeros((tn//pat, fn))
    predict_flows_sum = np.zeros((tn//pat, fn))

    for i in range(tn//pat):
        true_flows_sum[i, :] = np.sum(true_test_flow[i*pat:(i+1)*pat, :])
        predict_flows_sum[i, :] = np.sum(pred_test_flow[i*pat:(i+1)*pat, :])
    
    se = np.square(true_flows_sum-predict_flows_sum)
    mse = np.mean(se, axis=-1)
    return np.arange(tn//pat), mse


def sp_l1(true_test_flow, pred_test_flow):
    """
    Compute mean absolute error of every OD pair..

    :param true_test_flow: real OD flows
    :param pred_test_flow: predicted OD flows
    :return: MAE of every OD pair.
    """
    tn, fn = true_test_flow.shape 
    ae = np.abs(true_test_flow-pred_test_flow)
    nmae = np.mean(ae, axis=0)
    return np.arange(fn), nmae


def tm_l1(true_test_flow, pred_test_flow, pat=12):
    """
    Compute mean absolute error of every time batch.

    :param true_test_flow: real OD flows
    :param pred_test_flow: predicted OD flows
    :param pat: time batch size
    :return: MAE of every time batch
    """
    tn, fn = true_test_flow.shape
    true_flows_sum = np.zeros((tn//pat, fn))
    predict_flows_sum = np.zeros((tn//pat, fn))

    for i in range(tn//pat):
        true_flows_sum[i, :] = np.sum(true_test_flow[i*pat:(i+1)*pat, :])
        predict_flows_sum[i, :] = np.sum(pred_test_flow[i*pat:(i+1)*pat, :])
    
    ae = np.abs(true_flows_sum-predict_flows_sum)
    nmae = np.mean(ae, axis=-1)
    return np.arange(tn//pat), nmae


def plot_sp(real_flow, pred_flow, label, loss, pict_name):
    """
    :param real_flow: real OD flows
    :param pred_flow: predicted OD flows
    :param label: curve name
    :param loss: y value (mse or mae)
    :param pict_name: save path
    """
    fig, ax = plt.subplots()

    if loss == "mse":
        x, f = sp_mse(real_flow, pred_flow)
        y_label = "Mean Square Error"
    elif loss == "l1norm":
        x, f = sp_l1(real_flow, pred_flow)
        y_label = "L1 Norm Error"
    
    ax.plot(x, f, label=label)
    ax.set_xlabel('FlowId', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label,  fontsize=14, fontweight='bold')
    ax.set_title('Figure 1')
    ax.legend()
    plt.savefig(pict_name)
    plt.show()


def plot_tm(real_flow, pred_flow, label, loss, pict_name):
    """
    :param real_flow: real OD flows
    :param pred_flow: predicted OD flows
    :param label: curve name
    :param loss: y value (mse or mae)
    :param pict_name: save path
    """
    fig, ax = plt.subplots()

    if loss == "mse":
        x, f = tm_mse(real_flow, pred_flow)
        y_label = "Mean Square Error"
    elif loss == "l1norm":
        x, f = tm_l1(real_flow, pred_flow)
        y_label = "L1 Norm Error"

    ax.plot(x, f, label=label)
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label,  fontsize=14, fontweight='bold')
    ax.set_title('Figure 2')
    ax.legend()
    plt.savefig(pict_name)
    plt.show()