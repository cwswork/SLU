import random
import numpy as np
import torch


############
def task_divide(idx, n):
    ''' 划分成N个任务 '''
    ls_len = len(idx)
    if n <= 0 or 0 == ls_len or n>=ls_len:
        return [idx]
    elif n == ls_len:
        return [[i] for i in idx]
    else:
        j = ls_len // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def divide_list(idx_list, batch_size):
    ''' 划分成N个任务 '''
    total = len(idx_list)
    if batch_size <= 0 or total <= batch_size:
        return [idx_list]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            batchs_list.append(idx_list[beg:beg + batch_size])

        if beg + batch_size < total:
            batchs_list.append(idx_list[beg + batch_size:])
        return batchs_list

def divide_range(total, batch_size):
    ''' 划分成N个任务 '''
    if batch_size <= 0 or total <= batch_size:
        return [0, total]
    else:
        n = total // batch_size +1
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            batchs_list.append(beg)
        end = beg + batch_size
        if end > total:
            end = total
        batchs_list.append(end)
        return batchs_list

def divide_dict(idx_dict_list, batch_size):
    ''' 划分成N个任务 '''
    total = len(idx_dict_list)
    if batch_size <= 0 or total <= batch_size:
        newid_batch = np.array(idx_dict_list)
        return [newid_batch]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            newid_batch = np.array(idx_dict_list[beg:beg + batch_size])
            batchs_list.append(newid_batch)

        beg = beg + batch_size
        diff = total - beg
        if diff > 0:
            if diff < batch_size/4:#
                newid_batch = batchs_list[-1]
                newid_batch = np.vstack((newid_batch, np.array(idx_dict_list[beg:])))
                batchs_list[-1] = newid_batch
            else:
                newid_batch = np.array(idx_dict_list[beg:])
                batchs_list.append(newid_batch)
        return batchs_list


def divide_array(idx_dict_array, batch_size):
    ''' 划分成N个任务 '''
    total = idx_dict_array.shape[0]
    if batch_size <= 0 or total <= batch_size:
        newid_batch = idx_dict_array
        return [newid_batch]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            newid_batch = idx_dict_array[beg:beg + batch_size, :]
            batchs_list.append(newid_batch)

        beg = beg + batch_size
        diff = total - beg
        if diff > 0:
            if diff < batch_size/4:#
                newid_batch = batchs_list[-1]
                newid_batch = np.vstack((newid_batch, idx_dict_array[beg:, :]))
                batchs_list[-1] = newid_batch
            else:
                newid_batch = idx_dict_array[beg:, :]
                batchs_list.append(newid_batch)
        return batchs_list

def divide_array2(idx_dict_array, batch_num):
    ''' 划分成N个任务 '''
    total = idx_dict_array.shape[0]
    if batch_num==1:
        newid_batch = idx_dict_array
        return [newid_batch]
    else:
        batch_size = total // batch_num
        batch_size = batch_size if batch_size*batch_num==0 else (batch_size+1)

        batchs_list = []
        for i in range(batch_num-1):
            beg = i * batch_size
            newid_batch = idx_dict_array[beg:beg + batch_size, :]
            batchs_list.append(newid_batch)

        newid_batch = idx_dict_array[beg + batch_size:, :]
        batchs_list.append(newid_batch)

        return batchs_list
