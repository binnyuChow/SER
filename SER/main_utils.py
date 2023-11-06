# Copyright (c) 2020 Anita Hu and Kevin Su
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


# calculate the accuracy that the true label is in y_true's top k rank
# k: list of int. each <= num_cls
# y_pred: np array of probablities. (batch_size * cls_num) (output of softmax)
# y_true: batch_size * 1.
# return: list of acc given list of k

# 计算top_k https://zhuanlan.zhihu.com/p/340760336?ivk_sa=1024320u
def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # print(batch_size,"batch_size")
    _, pred = output.topk(maxk, 1, True, True)
    # 转置
    pred = pred.t()
    # print(pred, "t")
    # 两两比较两个tensor是否相等 .view相当于把tensor转换为一维 .expand_as相当于 tensor_1.expand_as(tensor_2) ：把tensor_1扩展成和tensor_2一样的形状
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct, "correct")
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    # print(res, "res")

def calculate_weighted_accuracy(predictions, labels):
    num_samples = len(predictions)

    # Step 1: Move predictions tensor to CPU and convert it to a numpy array

    # Step 2: Apply softmax function to convert logits into probabilities
    _, predicted_labels = predictions.topk(1, 1, True, True)
    predicted_labels = predicted_labels.t()
    predicted_labels.cpu()

    # Step 4: Compare predicted labels with true labels
    matched_samples = torch.sum(predicted_labels == labels).item()

        # Step 5: Calculate weighted accuracy

    weighted_accuracy = matched_samples / num_samples

    return weighted_accuracy * 100


def class_accuracy(output, target):
    # 计算每个类别的样本数量和预测正确的样本数量
    num_classes =[0,2,3,4]
    class_acc = 0
    n_classes = 0
    _, pred = output.topk(1, 1, True, True)
    # 转置
    pred = pred.t().squeeze().cpu()
    target = target.cpu()
    for c in num_classes:
        class_pred = np.multiply((target == pred),
                                 (target == c)).sum()
        class_pred = int(class_pred)
        if (target == c).sum() > 0:
            class_pred /= (target == c).sum()
            n_classes += 1

            class_acc += class_pred
    res = class_acc / n_classes * 100
    return float(res)


def train(get_X, log_interval, model, device, train_loader, optimizer, loss_func, metric_topk, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = [[]] * len(metric_topk)
    N_count = 0  # counting total trained sample in one epoch

    for batch_idx, sample in enumerate(train_loader):
        # distribute data to device
        X, n = get_X(device, sample)
        y = sample["emotion"].to(device).squeeze()
        output = model(X).to(device) # 加载数据

        N_count += n
        optimizer.zero_grad()
        loss = loss_func(output, y)
        losses.append(loss.item())

        step_score = accuracy_topk(output, y, topk=metric_topk)
        for i, ss in enumerate(step_score):
            scores[i].append(int(ss))

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
            for i, each_k in enumerate(metric_topk):
                print("Top {} accuracy: {:.2f}%".format(each_k, float(step_score[i])))

    return losses, scores


def validation(get_X, model, device, loss_func, val_loader, metric_topk, show_cm=True):
    # set model as testing mode
    model.eval()

    test_loss = []
    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for sample in val_loader:
            # distribute data to device
            X, _ = get_X(device, sample)
            # 真实标签
            y = sample["emotion"].to(device).squeeze()
            # 预测标签
            output = model(X)

            loss = loss_func(output, y).to(device)
            test_loss.append(loss.item())  # sum up batch loss

            # collect all y and y_pred in all batches
        #真实标签数组
            all_y.extend(y)
        #预测标签数组
            all_y_pred.extend(output)

    test_loss = np.mean(test_loss)

    # compute accuracy
    # 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)

    _, pred = all_y_pred.topk(1, 1, True, True)
    test_score = [float(t_acc) for t_acc in accuracy_topk(all_y_pred, all_y, topk=metric_topk)]
    Wa = calculate_weighted_accuracy(all_y_pred,all_y)
    Ua =class_accuracy(all_y_pred,all_y)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}'.format(len(all_y), test_loss))
    print('WA accuracy:{:.2f}%'.format(Wa) )
    print('UA accuracy:{:.2f}%'.format(Ua))
    for i, each_k in enumerate(metric_topk):
        print("Top {} accuracy: {:.2f}%".format(each_k, test_score[i]))
    print("\n")

    if show_cm:
        cm = confusion_matrix(all_y.cpu().data.squeeze().numpy(), pred.cpu().data.squeeze().numpy())
        print("Confusion matrix")
        print(cm)

    return test_loss, test_score ,Ua
