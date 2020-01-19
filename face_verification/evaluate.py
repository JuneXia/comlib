#! /usr/bin/python
import os
import sys
import argparse
import math
import numpy as np
import cv2


DEBUG = False
DEBUG_SHOW_HARD_SAMPLE = True

RELEASE = True
if RELEASE:
    sys.path.append('/disk1/home/xiaj/dev/FlaskFace')
else:
    sys.path.append('/home/xiajun/dev/FlaskFace')

from face_identification import faceid_pipeline  # TODO: faceid_pipeline 不应该在这python文件里调用的。
from face_verification import dist_verification
from sklearn.model_selection import KFold
from scipy import interpolate
from utils import dataset as datset
from utils import tools
import shutil
from multiprocessing import Process
from multiprocessing import Manager


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

g_faceid_model = None
g_is_similarity = True
g_is_best_threshold_deadline = 0.0  # 等于0时表示不使用，曾用值：0.28
g_hard_image_save_path = '/disk1/home/xiaj/res/tmptest_hard_image'

g_test_list = ['郑华晨', '张振宇', '罗鹏', '吴张勇',
               '徐黎明', '龚翔',  '云轶舟', '陈贤波',
               '冯艳晓', '纪书保',  '徐骋远', '夏俊',
               '靳博',
#'方小强',
#'胡成楠', '赵成伟',  '朱见平', # 没有大厅数据
               # '鲁帅', '戚佳宇', '周洋', '夏俊',
               ]


def distance(embeddings1, embeddings2, distance_metric=0, cos_similarity=False):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        if cos_similarity:
            dist = similarity
        else:
            dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

"""
def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    tmp_best_threshold = []

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds = thresholds[np.where(acc_train == acc_train[best_threshold_index])]
        best_threshold = best_thresholds[len(best_thresholds)//2]

        tmp_best_threshold.append(best_threshold)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(best_threshold, dist[test_set],
                                                      actual_issame[test_set])
        print('[calculate_roc]: best_threshold: ', best_threshold)

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        tools.view_bar('calculate_roc: ', fold_idx + 1, nrof_folds)
    print('')

    if DEBUG_SHOW_HARD_SAMPLE:
        global g_faceid_model
        tmp_best_threshold = np.array(tmp_best_threshold)
        tmp_best_threshold = tmp_best_threshold.mean()
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1, embeddings2]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)
        predict_issame = np.less(dist, tmp_best_threshold)
        hard_index = np.where(predict_issame != actual_issame)[0]
        hard_dist = dist[hard_index]
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

        save_path = os.path.join('/disk1/home/xiaj/res/face/tmp', g_faceid_model.split('/')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)

        with open('tmp_hard_list.csv', 'w') as f:
            for i, (index, dist) in enumerate(zip(hard_index, hard_dist)):
                img1, img2 = image_pairs[index]
                image1 = cv2.imread(img1)
                image2 = cv2.imread(img2)
                image = cv2.hconcat([image1, image2])

                text = 'pred, act, thr, dist'
                image = cv2.putText(image, text, (20, 20), font, 0.5, (255, 0, 0), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                text = '{}, {}, {:.3f}, {:.3f}'.format(int(predict_issame[index]), int(actual_issame[index]), tmp_best_threshold, dist)
                image = cv2.putText(image, text, (20, 50), font, 0.5, (255, 0, 0), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度

                img_path = os.path.join(save_path, str(i)+'.jpg')
                cv2.imwrite(img_path, image)

                # cv2.imshow('image1', image1)
                # cv2.imshow('image2', image2)
                # cv2.waitKey(0)

    return tpr, fpr, accuracy
"""


def calculate_roc_deadline(best_threshold, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    tmp_best_threshold = []

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings1, embeddings2]), axis=0)
    else:
        mean = 0.0
    dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric, cos_similarity=g_is_similarity)
    _, _, accuracy = calculate_accuracy(best_threshold, dist, actual_issame, is_similarity=g_is_similarity)

    return tpr, fpr, accuracy


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    tmp_best_threshold = []

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric, cos_similarity=g_is_similarity)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set], is_similarity=g_is_similarity)
        best_threshold_index = np.argmax(acc_train)
        best_thresholds = thresholds[np.where(acc_train == acc_train[best_threshold_index])]
        best_threshold = best_thresholds[len(best_thresholds)//2]
        # best_threshold = best_thresholds.mean()

        tmp_best_threshold.append(best_threshold)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[test_set],
                                                                                                 is_similarity=g_is_similarity)
        _, _, accuracy[fold_idx] = calculate_accuracy(best_threshold, dist[test_set],
                                                      actual_issame[test_set], is_similarity=g_is_similarity)
        print('[calculate_roc]: best_threshold: ', best_threshold)

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        tools.view_bar('calculate_roc: ', fold_idx + 1, nrof_folds)
    print('')

    tmp_best_threshold = np.array(tmp_best_threshold)
    tmp_best_threshold = tmp_best_threshold.mean()
    tmp_best_threshold = 0.5

    if DEBUG_SHOW_HARD_SAMPLE:
        global g_faceid_model

        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1, embeddings2]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric, cos_similarity=g_is_similarity)

        """
        predict_issame = np.greater(dist, 0.5)
        print('把不同的人识别成同一个人的数量：', ((predict_issame != actual_issame) * (1-actual_issame)).sum())
        print('把同一个人识别成同一个人的数量/同一个人的总数量：',(predict_issame * actual_issame+0).sum() / (actual_issame+0).sum())
        """

        if g_is_similarity:
            predict_issame = np.greater(dist, tmp_best_threshold)
        else:
            predict_issame = np.less(dist, tmp_best_threshold)
        hard_index = np.where(predict_issame != actual_issame)[0]
        hard_dist = dist[hard_index]
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

        save_path = os.path.join('/disk1/home/xiaj/res/face/tmp', g_faceid_model.split('/')[-1])
        save_path = save_path + '_thr{:.3f}'.format(tmp_best_threshold)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)

        print('will save hard sample in ', save_path)
        for i, (index, dist) in enumerate(zip(hard_index, hard_dist)):
            img1, img2 = image_pairs[index]
            image1 = cv2.imread(img1)
            image2 = cv2.imread(img2)
            image = cv2.hconcat([image1, image2])

            text = 'pred, act, thr, dist'
            image = cv2.putText(image, text, (20, 20), font, 0.5, (255, 0, 0), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            text = '{}, {}, {:.3f}, {:.3f}'.format(int(predict_issame[index]), int(actual_issame[index]),
                                                   tmp_best_threshold, dist)
            image = cv2.putText(image, text, (20, 50), font, 0.5, (255, 0, 0), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度

            field = str(i)+'_hard'+str(int(predict_issame[index]))+str(int(actual_issame[index]))
            img_path = os.path.join(save_path, field + '.jpg')
            cv2.imwrite(img_path, image)

            # cv2.imshow('image1', image1)
            # cv2.imshow('image2', image2)
            # cv2.waitKey(0)

    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame, is_similarity=False, use_second_verify=False, second_verify_threshold=0.9):
    """
    :param threshold:
    :param dist:
    :param actual_issame:
    :param is_similarity:
    :param use_second_verify: 仅在人脸识别1:N模式下有效
    :param second_verify_threshold: 仅当use_second_verify为True时有效
    :return:
    """
    if use_second_verify:
        if is_similarity:
            predict_issame = np.greater(dist[:, 0], threshold)
        else:
            predict_issame = np.less(dist[:, 0], threshold)
            raise Exception('距离比对情况下的二次校验策略目前还没有测过，需调试！')

        shift = dist[:, 1]/dist[:, 0]
        second_verify_issame = np.less(shift, second_verify_threshold)
        predict_issame = np.logical_and(predict_issame, second_verify_issame)
    else:
        if len(dist.shape) == 2:
            dist = dist[:, 0]
        elif len(dist.shape) > 2:
            raise Exception('暂不支持的维度')

        if is_similarity:
            predict_issame = np.greater(dist, threshold)
        else:
            predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric, cos_similarity=g_is_similarity)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set], is_similarity=g_is_similarity)
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        print('best threshold: ', threshold)
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set], is_similarity=g_is_similarity)
        tools.view_bar('calculate_val: ', fold_idx + 1, nrof_folds)
    print('')

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame, is_similarity=False):
    # 提取距离小于阈值的下标
    if is_similarity:
        predict_issame = np.greater(dist, threshold)
    else:
        predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))  # 确实是同一个人的数量
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))  # 假的是用一个人的数量
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    s_same = max(1e-10, float(n_same))
    n_diff = max(1e-10, float(n_diff))
    val = float(true_accept) / s_same
    far = float(false_accept) / n_diff
    return val, far


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    global g_is_similarity

    if g_is_similarity:
        if g_is_best_threshold_deadline > 0:
            thr_min = g_is_best_threshold_deadline
            thr_max = min(g_is_best_threshold_deadline + 0.5, 1)
            thresholds_roc = np.arange(thr_min, thr_max, 0.005)
        else:
            # thresholds_roc = np.arange(-1, 1, 0.005)
            thresholds_roc = np.arange(0, 1, 0.005)
    else:
        if g_is_best_threshold_deadline > 0:
            raise Exception('暂时还没到考虑这一步的时候')
        else:
            thresholds_roc = np.arange(0, 4, 0.01)  # TODO: 这个阈值的取值范围可以写到全局的地方去。
    #embeddings1 = embeddings[0::2]
    #embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds_roc, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds_val = np.arange(0, 4, 0.01)
    val, val_std, far = calculate_val(thresholds_val, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, thresholds_roc, accuracy, val, val_std, far


def extract_embeddings(faceid_model, image_pairs, emb_dict):
    if type(faceid_model) == str:
        faceid_model = (faceid_model, None)
    elif (type(faceid_model) == tuple) and (len(faceid_model) == 1):
        faceid_model += (None, )
    faceid_model_path, specify_ckpt = faceid_model
    use_fixed_image_standardization = True
    random_rotate = False
    random_crop = False
    random_flip = False
    fixed_contract = False

    if '20190909-050722' in faceid_model_path:
        use_fixed_image_standardization = False
    elif '20190911-061413' in faceid_model_path:
        fixed_contract = True
        use_fixed_image_standardization = False

    faceid_model = faceid_pipeline.FaceID(faceid_model_path, specify_ckpt)
    if type(image_pairs) == np.ndarray:
        embeddings1 = faceid_model.embedding(image_pairs[:, 0],
                                             use_fixed_image_standardization=use_fixed_image_standardization,
                                             random_rotate=random_rotate, random_crop=random_crop,
                                             random_flip=random_flip,
                                             fixed_contract=fixed_contract)
        embeddings2 = faceid_model.embedding(image_pairs[:, 1],
                                             use_fixed_image_standardization=use_fixed_image_standardization,
                                             random_rotate=random_rotate, random_crop=random_crop,
                                             random_flip=random_flip,
                                             fixed_contract=fixed_contract)
        emb_dict['embeddings1'] = embeddings1
        emb_dict['embeddings2'] = embeddings2
    elif type(image_pairs) == tuple:
        embeddings1 = faceid_model.embedding(image_pairs[0],
                                             use_fixed_image_standardization=use_fixed_image_standardization,
                                             random_rotate=random_rotate, random_crop=random_crop,
                                             random_flip=random_flip,
                                             fixed_contract=fixed_contract)
        embeddings2 = faceid_model.embedding(image_pairs[1],
                                             use_fixed_image_standardization=use_fixed_image_standardization,
                                             random_rotate=random_rotate, random_crop=random_crop,
                                             random_flip=random_flip,
                                             fixed_contract=fixed_contract)
        emb_dict['embeddings1'] = embeddings1
        emb_dict['embeddings2'] = embeddings2
    else:
        raise Exception('暂不支持别的类型')

    faceid_model.__del__()
    faceid_model = None

    # return embeddings1, embeddings2


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='verify mode: database_verify, lfw_verify, ppl_verify(path_path_label_verify).')
    parser.add_argument('--faceid_model', type=str, help='faceid model path.')
    parser.add_argument('--verify_model', type=str, help='verify model path.')
    parser.add_argument('--base_embeddings_file', type=str, help='Path to the database embeddings file.')
    parser.add_argument('--model_save_path', type=str, help='model save path.')
    parser.add_argument('--images_path', type=str, help='aligned images path.')
    parser.add_argument('--pairs_file', type=str, help='pairs file path.')
    parser.add_argument('--dist_threshold', default=1.0, type=float, help='ver dist threshold')
    parser.add_argument('--prob_threshold', default=0.80, type=float, help='ver prob threshold')
    parser.add_argument('--distance_metric', default=0, type=float, help='0: Euclidian distance, 1: Cosine')
    return parser.parse_args(argv)


def get_models(models_path, rmblank=True):
    model_list = []
    models = os.listdir(models_path)
    models.sort()
    for model in models:
        model_path = os.path.join(models_path, model)
        if os.path.isfile(model_path):
            model_list.append(models_path)
            break
        else:
            model_list_ = get_models(model_path, rmblank=rmblank)
            if len(model_list_) > 0:
                model_list.extend(model_list_)
            elif rmblank and (len(model_list_) == 0):
                shutil.rmtree(model_path)

    return model_list


def evaluate_api(embeddings1, embeddings2, distance_metric, model_name):
    tpr, fpr, thresholds_roc, accuracy, val, val_std, far = evaluate(embeddings1, embeddings2, image_pairs_label,
                                                                     nrof_folds=10,
                                                                     distance_metric=distance_metric,
                                                                     subtract_mean=True)
    auc = tools.compute_auc(fpr, tpr)

    model_name = model_name.split('.')[0]
    model_name = tools.strcat(model_name.split('-')[0:2], '-')

    eval_info = {'tpr': tpr, 'fpr': fpr, 'acc': accuracy, 'auc': auc,
                 'val': val, 'val_std': val_std, 'far': far,
                 'model_name': model_name}
    # eval_infos.append(eval_info)

    return eval_info


def evaluate_1vsN_api(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=True, distance_metric=1, is_similarity=True, model_name='', use_second_verify=False, second_verify_threshold=0.9):
    # font = cv2.FONT_HERSHEY_SIMPLEX  # 定
    font = cv2.FONT_HERSHEY_COMPLEX
    tests_name = tests_name.flatten()
    tests_path = tests_path.flatten()
    bases_name = bases_name.flatten()
    bases_path = bases_path.flatten()
    tests_cls_list = set(tests_name)
    total_true_accept = 0
    total_false_accept = 0

    predicts_index = []
    predicts_simil = []
    predicts_name = []
    actual_issame = []
    actual_names = []
    actual_paths = []
    for cls in tests_cls_list:
        index = np.where(tests_name == cls)
        test_path = tests_path[index]
        test_name = tests_name[index]
        test_emb = tests_emb[index]

        tmp_debug_flag = True
        for i, (tpath, tname, temb) in enumerate(zip(test_path, test_name, test_emb)):
            temb = temb.reshape((1, -1))
            if subtract_mean:
                mean = np.mean(np.concatenate([bases_emb, temb]), axis=0)
            else:
                mean = 0.0
            similarity = distance(temb - mean, bases_emb - mean, distance_metric, cos_similarity=is_similarity)

            # simil_index = np.argsort(-similarity)[0:2]  # 降序排列

            simil_index = np.argsort(-similarity)
            preds_name = bases_name[simil_index]
            second_index = 0
            for i in range(len(preds_name) + 1):
                if preds_name[0] != preds_name[i]:
                    second_index = i
                    break
            simil_index = np.array([simil_index[0], simil_index[second_index]])

            pred_simil = similarity[simil_index]
            pred_name = bases_name[simil_index]

            idx = np.where(pred_name == tname)
            issame = np.zeros((len(pred_name)))
            issame[idx] = 1

            predicts_index.append(simil_index)
            predicts_simil.append(pred_simil)
            predicts_name.append(pred_name)
            actual_issame.append(issame)
            actual_names.append(tname)
            actual_paths.append(tpath)

    actual_issame = np.array(actual_issame)
    predicts_simil = np.array(predicts_simil)
    thresholds = np.arange(0, 1, 0.005)

    acc = np.zeros((len(thresholds)))
    tpr = np.zeros((len(thresholds)))
    fpr = np.zeros((len(thresholds)))
    for threshold_idx, threshold in enumerate(thresholds):
        tpr[threshold_idx], fpr[threshold_idx], acc[threshold_idx] = calculate_accuracy(threshold, predicts_simil, actual_issame[:, 0],
                                                            is_similarity=g_is_similarity, use_second_verify=use_second_verify,
                                                            second_verify_threshold=second_verify_threshold)

    f = interpolate.interp1d(fpr, thresholds, kind='slinear')
    # threshold = f(0.001)
    threshold = f(fpr[np.where(fpr>0)[0][-5]])
    print('top-5 best threshold: ', threshold)
    val, far = calculate_val_far(threshold, predicts_simil[:, 0], actual_issame[:, 0], is_similarity=g_is_similarity)

    auc = tools.compute_auc(fpr, tpr)

    # model_name = model_name.split('.')[0]
    # model_name = tools.strcat(model_name.split('-')[0:2], '-')

    if True:
        global g_hard_image_save_path
        save_path = os.path.join(g_hard_image_save_path, model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)

        actual_names = np.array(actual_names)
        actual_paths = np.array(actual_paths)
        predicts_name = np.array(predicts_name)
        bases_path = np.array(bases_path)
        predicts_index = np.array(predicts_index)

        predict_issame = np.greater(predicts_simil[:, 0], threshold)
        fp = np.logical_and(predict_issame, np.logical_not(actual_issame[:, 0]))
        fp_index = np.where(fp)

        fpred_name = predicts_name[fp_index]  # bases_name[predicts_index[fp_index]]
        actual_name = actual_names[fp_index]
        actual_path = actual_paths[fp_index]
        fpred_path = bases_path[predicts_index[fp_index]]
        fpred_simil = predicts_simil[fp_index]

        for i in range(len(fpred_path)):
            bimg1 = cv2.imread(fpred_path[i][0])
            # timg = cv2.putText(timg, cls, (20, 20), font, 0.5, (0, 0, 255), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            text = '{}, {:.3f}'.format(fpred_name[i][0], fpred_simil[i][0])
            bimg1 = cv2ImgAddText(bimg1, text, 20, 20, (255, 0, 0), 20)

            bimg2 = cv2.imread(fpred_path[i][1])
            text = '{}, {:.3f}'.format(fpred_name[i][1], fpred_simil[i][1])
            bimg2 = cv2ImgAddText(bimg2, text, 20, 20, (0, 0, 255), 20)

            aimg = cv2.imread(actual_path[i])
            aimg = cv2ImgAddText(aimg, actual_name[i], 20, 20, (0, 0, 255), 20)

            img = cv2.hconcat((aimg, bimg1, bimg2))

            imfile = os.path.join(save_path, str(i)+'.jpg')
            cv2.imwrite(imfile, img)

            # cv2.imshow('show', img)
            # cv2.waitKey(0)

    eval_info = {'tpr': tpr, 'fpr': fpr, 'acc': acc, 'auc': auc,
                 'val': val, 'val_std': -1, 'far': far,
                 'model_name': model_name, 'thresholds': thresholds}

    return eval_info


def model_evaluate(faceid_models, image_pairs, distance_metric, assemble=False):
    global g_faceid_model

    embeddings1 = np.array([[]]*len(image_pairs))
    embeddings2 = np.array([[]]*len(image_pairs))
    assemble_model_name = 'assemble:'
    eval_infos = []
    emb_dict = Manager().dict()
    for i, faceid_model in enumerate(faceid_models):
        if DEBUG_SHOW_HARD_SAMPLE:
            if type(faceid_model) == str:
                faceid_model = (faceid_model, None)
            elif (type(faceid_model) == tuple) and (len(faceid_model) == 1):
                faceid_model += (None,)
            faceid_model_path, specify_ckpt = faceid_model

            g_faceid_model = faceid_model_path

        p = Process(target=extract_embeddings, args=(faceid_model, image_pairs, emb_dict))
        p.start()
        p.join()

        embs1, embs2 = emb_dict['embeddings1'], emb_dict['embeddings2']

        if type(faceid_model) == tuple:
            faceid_model = faceid_model[0]

        model_name = faceid_model.split('/')[-1]

        if assemble:
            embeddings1 = np.concatenate((embeddings1, embs1), axis=1)
            embeddings2 = np.concatenate((embeddings2, embs2), axis=1)
            assemble_model_name = tools.strcat([assemble_model_name, model_name], cat_mark='_')
        else:
            eval_info = evaluate_api(embs1, embs2, distance_metric, model_name)
            eval_infos.append(eval_info)

        tools.view_bar('[model_evaluate]:: loading: ', i + 1, len(faceid_models))
    print('')

    if assemble:
        g_faceid_model = assemble_model_name
        eval_info = evaluate_api(embeddings1, embeddings2, distance_metric, assemble_model_name)
        eval_infos.append(eval_info)

    return eval_infos


def model_evaluate_1vsN(faceid_models, tests_name, tests_path, bases_name, bases_path, distance_metric, is_similarity=True, assemble=False,
                        use_second_verify=False, second_verify_threshold=0.9):
    global g_faceid_model
    global g_hard_image_save_path
    if os.path.exists(g_hard_image_save_path):
        shutil.rmtree(g_hard_image_save_path)

    embeddings1 = np.array([[]]*len(bases_path))
    embeddings2 = np.array([[]]*len(tests_path))
    assemble_model_name = 'assemble:'
    eval_infos = []
    emb_dict = Manager().dict()
    for i, faceid_model in enumerate(faceid_models):
        if DEBUG_SHOW_HARD_SAMPLE:
            if type(faceid_model) == str:
                faceid_model = (faceid_model, None)
            elif (type(faceid_model) == tuple) and (len(faceid_model) == 1):
                faceid_model += (None,)
            faceid_model_path, specify_ckpt = faceid_model

            g_faceid_model = faceid_model_path

        p = Process(target=extract_embeddings, args=(faceid_model, (bases_path, tests_path), emb_dict))
        p.start()
        p.join()

        embs1, embs2 = emb_dict['embeddings1'], emb_dict['embeddings2']

        if type(faceid_model) == tuple:
            faceid_model = faceid_model[0]

        model_name = faceid_model.split('/')[-1]
        if specify_ckpt is not None:
            model_name = model_name + specify_ckpt.split('.')[-1]

        if assemble:
            embeddings1 = np.concatenate((embeddings1, embs1), axis=1)
            embeddings2 = np.concatenate((embeddings2, embs2), axis=1)
            assemble_model_name = tools.strcat([assemble_model_name, model_name], cat_mark='_')
        else:
            # eval_info = evaluate_api(embs1, embs2, distance_metric, model_name)
            eval_info = evaluate_1vsN_api(tests_name, tests_path, embs2, bases_name, bases_path, embs1,
                                          subtract_mean=True, distance_metric=distance_metric, is_similarity=is_similarity, model_name=model_name,
                                          use_second_verify=use_second_verify, second_verify_threshold=second_verify_threshold)
            eval_infos.append(eval_info)

        tools.view_bar('[model_evaluate]:: loading: ', i + 1, len(faceid_models))
    print('')

    if assemble:
        g_faceid_model = assemble_model_name
        # eval_info = evaluate_api(embeddings1, embeddings2, distance_metric, assemble_model_name)
        eval_info = evaluate_1vsN_api(tests_name, tests_path, embeddings2, bases_name, bases_path, embeddings1,
                                      subtract_mean=True, distance_metric=distance_metric, is_similarity=is_similarity, model_name=assemble_model_name,
                                      use_second_verify=use_second_verify, second_verify_threshold=second_verify_threshold)
        eval_infos.append(eval_info)

    return eval_infos


def get_evaluate_pairs(data_path, pair_save_path=None, over_write=False):
    """
    当pair_save_path不为空，且pair_save_path存在，且over_write为False时，才会直接读取pair_save_path文件中的样本对；
    否则都将从data_path生成样本对。
    :param data_path:
    :param pair_save_path:
    :param over_write: 是否覆盖pair_save_path文件。
    :return:
    """
    if pair_save_path and os.path.exists(pair_save_path) and (not over_write):
        with open(pair_save_path, 'r') as f:
            lines_info = []
            lines = f.readlines()
            for line in lines:
                lines_info.append(line.strip('\n').split(','))
        lines = np.array(lines_info)
        image_pairs = lines[:, 0:2]
        image_pairs_label = lines[:, 2]
        image_pairs_label = image_pairs_label.astype(np.int32)
    else:
        validation_images_path, validation_images_label = datset.load_dataset(data_path, shuffle=True)
        validation_dataset = datset.SiameseDataset(validation_images_path, validation_images_label, is_train=False)

        image_pairs = []
        image_pairs_label = []
        for batch, (images, labels) in enumerate(validation_dataset):
            image_pairs.append(images)
            image_pairs_label.append(labels)

        if pair_save_path:
            with open(pair_save_path, 'w') as f:
                for i, ((img1, img2), lab) in enumerate(zip(image_pairs, image_pairs_label)):
                    f.write(tools.strcat([img1, img2, lab]) + '\n')

        image_pairs = np.array(image_pairs)
        image_pairs_label = np.array(image_pairs_label)

    return image_pairs, image_pairs_label


def test_filter_cb(imgs):
    if len(g_test_list) > 0:
        name = imgs[0].split('/')[-2]
        if name in g_test_list:
            return imgs
        else:
            return []
    else:
        return imgs


def base_filter_cb(imgs):
    images = []
    if len(g_test_list) > 0:
        name = imgs[0].split('/')[-2]
        if name in g_test_list:
            for img in imgs:
                if '手机' in img:
                    continue
                # if ('展厅' in img) or ('大厅' in img):
                images.append(img)
            if len(images) == 0:
                print('debug')
    else:
        for img in imgs:
            if '手机' in img:
                continue
            # if ('展厅' in img) or ('大厅' in img):
            images.append(img)
    return images


def _get_images_path(data_path, filter_cb=None, max_nrof_cls=100):
    if DEBUG:
        datas_path, datas_label = datset.load_dataset(data_path, shuffle=False, max_nrof_cls=4, filter_cb=filter_cb)
    else:
        datas_path, datas_label = datset.load_dataset(data_path, shuffle=False, max_nrof_cls=max_nrof_cls, filter_cb=filter_cb)
    datas_name = []
    for dpath in datas_path:
        person_name = dpath.split('/')[-2]
        datas_name.append(person_name)
    datas_name = np.array(datas_name)
    datas_name = datas_name.reshape((-1, 1))
    datas_path = datas_path.reshape((-1, 1))

    return datas_path, datas_name


from PIL import Image, ImageDraw, ImageFont
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    fontText = ImageFont.truetype('NotoSansCJK-Light.ttc', textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



"""

def metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=True, distance_metric=1, threshold=0.5, is_similarity=True):
    # font = cv2.FONT_HERSHEY_SIMPLEX  # 定
    font = cv2.FONT_HERSHEY_COMPLEX
    tests_cls_list = set(tests_name)
    total_true_accept = 0
    total_false_accept = 0
    for cls in tests_cls_list:
        index = np.where(tests_name == cls)
        test_path = tests_path[index]
        test_name = tests_name[index]
        test_emb = tests_emb[index]

        predicts_index = []
        predicts_simil = []
        tmp_debug_flag = True
        for i, (tpath, tname, temb) in enumerate(zip(test_path, test_name, test_emb)):
            temb = temb.reshape((1, -1))
            if subtract_mean:
                mean = np.mean(np.concatenate([bases_emb, temb]), axis=0)
            else:
                mean = 0.0
            similarity = distance(temb - mean, bases_emb - mean, distance_metric, cos_similarity=is_similarity)
            simil_index = np.argsort(-similarity)[0:2]  # 降序排列
            simil = similarity[simil_index]
            if simil[0] > threshold:
                predicts_index.append(simil_index)
                predicts_simil.append(simil)

                if DEBUG_SHOW_HARD_SAMPLE:
                    pred_name = bases_name[simil_index]
                    pred_path = bases_path[simil_index]

                    if pred_name[0] != cls and tmp_debug_flag:
                        print(tpath)
                        timg = cv2.imread(tpath)
                        # timg = cv2.putText(timg, cls, (20, 20), font, 0.5, (0, 0, 255), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                        timg = cv2ImgAddText(timg, cls, 20, 20, (255, 0, 0), 20)

                        pimg0 = cv2.imread(pred_path[0])
                        text = '{}, {:.3f}'.format(pred_name[0], simil[0])
                        pimg0 = cv2ImgAddText(pimg0, text, 20, 20, (255, 0, 0), 20)

                        pimg1 = cv2.imread(pred_path[1])
                        text = '{}, {:.3f}'.format(pred_name[1], simil[1])
                        pimg1 = cv2ImgAddText(pimg1, text, 20, 20, (0, 0, 255), 20)

                        img = cv2.hconcat((timg, pimg0, pimg1))
                        cv2.imshow('show', img)
                        cv2.waitKey(0)

        true_accept = false_accept = 0
        if len(predicts_index) > 0:
            predicts_index = np.array(predicts_index)
            predicts_simil = np.array(predicts_simil)
            predicts_name = bases_name[predicts_index[:, 0]]

            true_accept = len(np.where(predicts_name == cls)[0])  # 预测正确的
            false_accept = len(np.where(predicts_name != cls)[0])  # 预测正确的
            total_true_accept += true_accept
            total_false_accept += false_accept

        tar = true_accept / (len(test_emb))
        far = false_accept / (len(test_emb))
        tar_text = 'tar={}/{}={:.4f}'.format(true_accept, len(test_emb), tar)
        far_text = 'far={}/{}={:.4f}'.format(false_accept, len(test_emb), far)
        print('test name: {}\t {}@{}'.format(cls, tar_text, far_text))

    tar = total_true_accept / (len(tests_name))
    far = total_false_accept / (len(tests_name))
    tar_text = 'tar={}/{}={:.4f}'.format(total_true_accept, len(tests_name), tar)
    far_text = 'far={}/{}={:.4f}'.format(total_false_accept, len(tests_name), far)
    print('total {}@{}'.format(tar_text, far_text))


"""

if __name__ == '__main__1':
    tprs = np.array([9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90334907e-01,
       9.90334907e-01, 9.90334907e-01, 9.90334907e-01, 9.90110137e-01,
       9.89885367e-01, 9.89660598e-01, 9.89660598e-01, 9.89660598e-01,
       9.89660598e-01, 9.89660598e-01, 9.89660598e-01, 9.89660598e-01,
       9.89435828e-01, 9.89435828e-01, 9.89435828e-01, 9.89211059e-01,
       9.89211059e-01, 9.89211059e-01, 9.88986289e-01, 9.88087211e-01,
       9.87862441e-01, 9.87862441e-01, 9.87862441e-01, 9.87637671e-01,
       9.87188132e-01, 9.86738593e-01, 9.86064284e-01, 9.85839514e-01,
       9.85165206e-01, 9.84490897e-01, 9.83367049e-01, 9.82467970e-01,
       9.80669813e-01, 9.80445044e-01, 9.79545965e-01, 9.78646887e-01,
       9.77747808e-01, 9.76399191e-01, 9.75500112e-01, 9.73926725e-01,
       9.72353338e-01, 9.71229490e-01, 9.69656102e-01, 9.69206563e-01,
       9.68307485e-01, 9.68082715e-01, 9.66958867e-01, 9.64261632e-01,
       9.62013936e-01, 9.58867161e-01, 9.56169926e-01, 9.54596539e-01,
       9.51224994e-01, 9.49651607e-01, 9.46055293e-01, 9.42234210e-01,
       9.37289278e-01, 9.35041582e-01, 9.32793886e-01, 9.28972803e-01,
       9.25826028e-01, 9.21780175e-01, 9.17284783e-01, 9.13463700e-01,
       9.09867386e-01, 9.06495842e-01, 9.03349067e-01, 8.98404136e-01,
       8.91211508e-01, 8.86940886e-01, 8.80872106e-01, 8.74803327e-01,
       8.67385929e-01, 8.59294223e-01, 8.52551135e-01, 8.46931895e-01,
       8.37042032e-01, 8.29849404e-01, 8.20633850e-01, 8.10519218e-01,
       8.01977973e-01, 7.93436727e-01, 7.85569791e-01, 7.76354237e-01,
       7.65790065e-01, 7.56349742e-01, 7.47358957e-01, 7.38368173e-01,
       7.24432457e-01, 7.12969207e-01, 7.01730726e-01, 6.88244549e-01,
       6.71611598e-01, 6.55203416e-01, 6.41717240e-01, 6.30029220e-01,
       6.15419195e-01, 5.99460553e-01, 5.81478984e-01, 5.64396494e-01,
       5.49112160e-01, 5.31804900e-01, 5.14722410e-01, 4.98763767e-01,
       4.81906046e-01, 4.66621713e-01, 4.49988762e-01, 4.33355810e-01,
       4.13351315e-01, 3.94920207e-01, 3.74690942e-01, 3.52888290e-01,
       3.37603956e-01, 3.18498539e-01, 2.98943583e-01, 2.79388627e-01,
       2.63205215e-01, 2.42076871e-01, 2.24769611e-01, 2.09934817e-01,
       1.93077096e-01, 1.76893684e-01, 1.64306586e-01, 1.51045179e-01,
       1.38458080e-01, 1.30141605e-01, 1.18453585e-01, 1.08788492e-01,
       9.86738593e-02, 8.74353787e-02, 8.09170600e-02, 7.37244325e-02,
       6.76556530e-02, 6.15868735e-02, 5.39447067e-02, 4.74263880e-02,
       4.18071477e-02, 3.73117555e-02, 3.14677456e-02, 2.71971229e-02,
       2.24769611e-02, 1.79815689e-02, 1.52843336e-02, 1.39357159e-02,
       1.23623286e-02, 1.05641717e-02, 9.88986289e-03, 8.99078445e-03,
       8.09170600e-03, 4.04585300e-03, 3.82108339e-03, 3.59631378e-03,
       3.37154417e-03, 2.69723533e-03, 1.57338728e-03, 1.57338728e-03,
       1.34861767e-03, 6.74308833e-04, 6.74308833e-04, 6.74308833e-04])
    fprs = np.array([0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.70212766, 0.70212766, 0.70212766, 0.70212766, 0.70212766,
       0.69503546, 0.68794326, 0.68085106, 0.65957447, 0.65248227,
       0.65248227, 0.63829787, 0.63829787, 0.63829787, 0.62411348,
       0.60992908, 0.60992908, 0.60992908, 0.60992908, 0.60283688,
       0.60283688, 0.59574468, 0.59574468, 0.58865248, 0.58156028,
       0.56737589, 0.53900709, 0.53191489, 0.5106383 , 0.5035461 ,
       0.5035461 , 0.5035461 , 0.4964539 , 0.4751773 , 0.46808511,
       0.46099291, 0.44680851, 0.43971631, 0.43262411, 0.43262411,
       0.42553191, 0.41134752, 0.39007092, 0.39007092, 0.38297872,
       0.32624113, 0.32624113, 0.31205674, 0.31205674, 0.29787234,
       0.28368794, 0.27659574, 0.26950355, 0.21985816, 0.21985816,
       0.21985816, 0.21276596, 0.19148936, 0.12765957, 0.12765957,
       0.12056738, 0.10638298, 0.09219858, 0.08510638, 0.03546099,
       0.03546099, 0.03546099, 0.03546099, 0.03546099, 0.03546099,
       0.03546099, 0.0212766 , 0.0212766 , 0.0212766 , 0.0212766 ,
       0.0212766 , 0.0212766 , 0.0070922 , 0.0070922 , 0.0070922 ,
       0.0070922 , 0.0070922 , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])
    thresholds = np.arange(0, 1, 0.005)

    tools.plt_roc([fprs], [tprs], 'xxoo', thresholds=[thresholds], interpret_label='', save_path=None)

    print('end!')


if __name__ == '__main__2':  # 使用CelebA等数据集测试并绘制所有模型的ROC曲线
    distance_metric = 0
    if RELEASE:
        # data_path = '/disk1/home/xiaj/res/face/gcface/gc_together/origin_align160_margin32'
        data_path = '/disk1/home/xiaj/res/face/CelebA/Experiment/facenet_mtcnn_align160x160_margin32'
        faceid_models = ['/disk1/home/xiaj/dev/alg_verify/face/facenet/pretrained_model/20180402-114759/20180402-114759.pb']
        models_path = '/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model'
    else:
        data_path = '/home/xiajun/dataset/gc_together/origin_gen90_align160_margin32'
        faceid_models = ['/home/xiajun/dev/facerec/facenet/mydataset/models/20180402-114759']
        models_path = '/home/xiajun/dev/alg_verify/face/facenet/save_model'

    image_pairs, image_pairs_label = get_evaluate_pairs(data_path)
    models = get_models(models_path, rmblank=False)
    # models = models[0:1]
    faceid_models.extend(models)

    eval_infos = model_evaluate(faceid_models, image_pairs, distance_metric)

    eval_pairs_info = ''
    cls_label = set(image_pairs_label)
    eval_pairs_info = ['{}:{}'.format(cls, len(np.where(image_pairs_label == cls)[0])) for cls in cls_label]
    eval_pairs_info = tools.strcat(eval_pairs_info, cat_mark=', ')

    tprs = []
    fprs = []
    show_labels = []
    for info in eval_infos:
        tprs.append(info['tpr'])
        fprs.append(info['fpr'])
        val = '%.4f±%.3f' % (info['val'], info['val_std'])
        acc = info['acc']
        acc = '%.4f±%.3f' % (acc.mean(), acc.std())
        auc = '%.4f' % info['auc']
        far = '%.6f' % info['far']
        show_label = tools.strcat([info['model_name'], acc, auc, val, far], cat_mark=' : ')
        show_labels.append(show_label)
        print(show_label)
    tools.plt_roc(fprs, tprs, show_labels, interpret_label='model:acc±acc_std:auc:val±val_std:far\n'+eval_pairs_info, save_path='./roc.jpg')


if __name__ == '__main__':  # 测试各个模型在事先生成好的样本pair数据集上的准确率，可以选择度量方式(欧式距离、余弦)，是否使用集成方法
    distance_metric = 1
    assemble = False
    metric_model = '1vsN'
    which_model = 'not all'
    use_second_verify = True  # 仅在1vsN模式下有效
    second_verify_threshold = 0.9
    if RELEASE:
        # data_path = '/disk1/home/xiaj/res/face/gcface/gc_together/origin_align160_margin32'
        # data_path = '/disk1/home/xiaj/res/face/CelebA/Experiment/facenet_mtcnn_align160x160_margin32'
        data_path = '/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32'
        faceid_models = [
            # '/disk1/home/xiaj/dev/alg_verify/face/facenet/pretrained_model/20180402-114759/20180402-114759.pb',
        ]
        models_path = '/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model'
        # pair_save_path = '/disk1/home/xiaj/res/face/gcface/gc_together/origin_align160_margin32_pairs.txt'
        pair_save_path = '/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32_pairs.txt'
    else:
        data_path = '/home/xiajun/dataset/gc_together/origin_gen90_align160_margin32'
        faceid_models = ['/home/xiajun/dev/facerec/facenet/mydataset/models/20180402-114759']
        models_path = '/home/xiajun/dev/alg_verify/face/facenet/save_model'

    if metric_model == '1vs1':
        image_pairs, image_pairs_label = get_evaluate_pairs(data_path, pair_save_path)
    elif metric_model == '1vsN':
        base_dir = '/disk1/home/xiaj/res/face/gcface/faceid_1toN_test_base-mtcnn_align160x160_margin32'
        bases_path, bases_name = _get_images_path(base_dir, filter_cb=base_filter_cb)

        test_dir = '/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest'
        tests_path, tests_name = _get_images_path(test_dir, filter_cb=test_filter_cb, max_nrof_cls=999999999)
    else:
        raise Exception('暂不支持')

    if which_model == 'all':
        models = get_models(models_path, rmblank=False)
        faceid_models.extend(models)
    else:
        models = [#('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20190719-062455', 'model-20190719-062455.ckpt-498'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191016-203214', 'model-20191016-203214.ckpt-275'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191110-222038', 'model-20191110-222038-acc0.996833-val0.992000-prec0.998999.ckpt-107'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191030-083945', 'model-20191030-083945.ckpt-142'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191122-044343', 'model-20191122-044343-acc0.996000-val0.991667-prec0.998999.ckpt-235'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191104-045636', 'model-20191104-045636-acc0.996667-val0.991000-prec0.999336.ckpt-171'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191127-005344', 'model-20191127-005344-acc0.996167-val0.993333-prec0.998671.ckpt-341'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191127-005344', 'model-20191127-005344-acc0.994333-val0.991667-prec0.999001.ckpt-391'),  # 0.5279
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/TrainGcFace/20190517-231453', 'model-20190517-231453.ckpt-4'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/Reuse20190517-021344/20190517-035733', 'model-20190517-035733.ckpt-4'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/TrainGcFace/20190517-224828-CenterLossAlfa095Factor00003', 'model-20190517-224828.ckpt-1'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/20190518-034602', 'model-20190518-034602.ckpt-150'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/20190616-224319', 'model-20190616-224319.ckpt-275'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191203-083355', 'model-20191203-083355-acc0.996667-val0.992667-prec0.998668.ckpt-330'),  # 0.6003
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191203-083355', 'model-20191203-083355-final.ckpt-400'),  # 0.5985
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191204-001947', 'model-20191204-001947-acc0.996500-val0.994000-prec0.998668.ckpt-314'),  # 0.615
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191204-001947', 'model-20191204-001947-acc0.996333-val0.993000-prec0.999336.ckpt-393'),  # 0.4861, but 通过率下降了点
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191204-050907', 'model-20191204-050907-acc0.996333-val0.993000-prec0.999003.ckpt-387'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191208-195740', 'model-20191208-195740-acc0.997000-val0.993667-prec0.999336.ckpt-333'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191208-195740', 'model-20191208-195740-acc0.996167-val0.991667-prec0.999005.ckpt-396'),
                  #-('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191208-201809', 'model-20191208-201809-acc0.996833-val0.989000-prec0.999338.ckpt-352'),
                  ('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191208-201809', 'model-20191208-201809-acc0.996167-val0.991333-prec0.999004.ckpt-383'),
                  #('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191212-213848', 'model-20191212-213848-acc0.997000-val0.989000-prec0.999338.ckpt-371'),
                  ('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191212-213848', 'model-20191212-213848-acc0.996833-val0.991667-prec0.998670.ckpt-386'),
                  #-('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191212-213848', 'model-20191212-213848-acc0.993833-val0.991667-prec0.999001.ckpt-391'),
                  #-('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191213-034255', 'model-20191213-034255-final.ckpt-400'),
                  ('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191216-000340', 'model-20191216-000340-acc0.996500-val0.994000-prec0.999001.ckpt-339'),
                  ('/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191217-210306', 'model-20191217-210306-acc0.996333-val0.993333-prec0.998673.ckpt-397'),
                  ]
        faceid_models.extend(models)

    if metric_model == '1vs1':
        eval_infos = model_evaluate(faceid_models, image_pairs, distance_metric, assemble=assemble)
    elif metric_model == '1vsN':
        eval_infos = model_evaluate_1vsN(faceid_models, tests_name, tests_path, bases_name, bases_path, distance_metric, is_similarity=True, assemble=assemble,
                                         use_second_verify=use_second_verify, second_verify_threshold=second_verify_threshold)

    eval_pairs_info = ''
    if metric_model == '1vs1':
        cls_label = set(image_pairs_label)
        eval_pairs_info = ['{}:{}'.format(cls, len(np.where(image_pairs_label == cls)[0])) for cls in cls_label]
        eval_pairs_info = tools.strcat(eval_pairs_info, cat_mark=', ')
    elif metric_model == '1vsN':
        pass


    tprs = []
    fprs = []
    thresholds = []
    show_labels = []
    for info in eval_infos:
        tprs.append(info['tpr'])
        fprs.append(info['fpr'])
        if metric_model == '1vsN':
            thresholds.append(info['thresholds'])
        val = '%.4f±%.3f' % (info['val'], info['val_std'])
        acc = info['acc']
        acc = '%.4f±%.3f' % (acc.mean(), acc.std())
        auc = '%.4f' % info['auc']
        far = '%.6f' % info['far']
        show_label = tools.strcat([info['model_name'], acc, auc, val, far], cat_mark=' : ')
        show_labels.append(show_label)
        print(show_label)
    tools.plt_roc(fprs, tprs, show_labels, thresholds=thresholds, interpret_label='model:acc±acc_std:auc:val±val_std:far\n'+eval_pairs_info, save_path='./roc.jpg')


"""

"""

"""
# pair_save_path = '/disk1/home/xiaj/res/face/gcface/gc_together/origin_align160_margin32_pairs.txt'
distance_metric: 1
20180402-114759 : 0.9793±0.009 : 0.9920 : 0.8933±0.164 : 0.000000
20190719-062455 : 0.9856±0.005 : 0.9928 : 0.9412±0.023 : 0.000794
20191016-203214 : 0.9820±0.010 : 0.9917 : 0.9435±0.021 : 0.001620

distance_metric: 0
20180402-114759 : 0.9762±0.009 : 0.9908 : 0.8244±0.044 : 0.001660
20190719-062455 : 0.9801±0.006 : 0.8511 : 0.8275±0.051 : 0.000794
20191016-203214 : 0.9766±0.009 : 0.9784 : 0.8887±0.033 : 0.001503

20180402-114759 : 0.9941±0.004 : 0.9997 : 0.8933±0.164 : 0.000000
20190719-062455 : 0.9898±0.006 : 0.9994 : 0.9412±0.023 : 0.000794
20191016-203214 : 0.9926±0.004 : 0.9980 : 0.9427±0.020 : 0.001620


distance_metric: 1
20180402-114759 : 0.9941±0.004 : 0.9997 : 0.8933±0.164 : 0.000000
20190719-062455 : 0.9898±0.006 : 0.9994 : 0.9412±0.023 : 0.000794
20191016-203214 : 0.9930±0.003 : 0.9988 : 0.9435±0.021 : 0.001620
20191110-222038 : 0.9863±0.005 : 0.9992 : 0.6833±0.056 : 0.000000
20191030-083945 : 0.9961±0.003 : 0.9998 : 0.9746±0.017 : 0.001613
20191122-044343 : 0.9930±0.004 : 0.9997 : 0.9222±0.123 : 0.000000
20191104-045636 : 0.9875±0.005 : 0.9986 : 0.9452±0.021 : 0.000794


distance_metric: 0
20180402-114759 : 0.9770±0.011 : 0.9949 : 0.8244±0.044 : 0.001660
20190719-062455 : 0.9762±0.006 : 0.9165 : 0.8275±0.051 : 0.000794
20191016-203214 : 0.9754±0.010 : 0.9903 : 0.8887±0.033 : 0.001503
20191110-222038 : 0.9774±0.007 : 0.9800 : 0.8557±0.056 : 0.000000
20191030-083945 : 0.9859±0.007 : 0.9041 : 0.9533±0.021 : 0.001613
20191122-044343 : 0.9801±0.009 : 0.9830 : 0.8887±0.036 : 0.001503
20191104-045636 : 0.9774±0.008 : 0.9779 : 0.8866±0.039 : 0.001647


distance_metric: 1
'/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32_pairs.txt'
20180402-114759 : 0.9456±0.003 : 0.9915 : 0.6544±0.017 : 0.001061
20190719-062455 : 0.9614±0.002 : 0.9941 : 0.7705±0.010 : 0.000868
20191016-203214 : 0.9641±0.005 : 0.9946 : 0.7362±0.020 : 0.000870
20191110-222038 : 0.9616±0.003 : 0.9938 : 0.7445±0.017 : 0.001065
20191030-083945 : 0.9693±0.002 : 0.9959 : 0.7933±0.014 : 0.000967
20191122-044343 : 0.9645±0.003 : 0.9949 : 0.7731±0.013 : 0.001163
20191104-045636 : 0.9622±0.003 : 0.9936 : 0.7238±0.014 : 0.000865
20191127-005344 : 0.9706±0.003 : 0.9963 : 0.8094±0.010 : 0.001256
20191127-005344 : 0.9706±0.003 : 0.9961 : 0.8042±0.015 : 0.000959
20190517-231453 : 0.9706±0.003 : 0.9961 : 0.8042±0.015 : 0.000959
20190517-035733 : 0.9286±0.004 : 0.9795 : 0.5559±0.020 : 0.001159
20190517-224828 : 0.9391±0.003 : 0.9840 : 0.6803±0.013 : 0.000977
20190518-034602 : 0.9629±0.003 : 0.9937 : 0.8086±0.009 : 0.001072
20190616-224319 : 0.9577±0.003 : 0.9920 : 0.6848±0.019 : 0.001061
20191203-083355 : 0.9694±0.004 : 0.9959 : 0.8024±0.006 : 0.000966
20191203-083355 : 0.9658±0.003 : 0.9951 : 0.7737±0.008 : 0.001062
20191204-001947 : 0.9618±0.002 : 0.9940 : 0.7453±0.019 : 0.000961
20191204-001947 : 0.9604±0.004 : 0.9929 : 0.7686±0.015 : 0.000868
20191204-050907 : 0.9641±0.003 : 0.9942 : 0.7599±0.013 : 0.001072
20191208-195740-ckpt333 : 0.9704±0.004 : 0.9961 : 0.8026±0.012 : 0.001062
20191208-195740-ckpt396 : 0.9721±0.002 : 0.9964 : 0.8322±0.011 : 0.001062
20191208-201809-ckpt383 : 0.9727±0.003 : 0.9969 : 0.8023±0.014 : 0.001059
20191208-201809-ckpt352 : 0.9730±0.003 : 0.9964 : 0.7932±0.011 : 0.001051
20191212-213848-ckpt371 : 0.9863±0.003 : 0.9991 : 0.9270±0.006 : 0.001060
20191212-213848-ckpt386 : 0.9876±0.002 : 0.9992 : 0.9216±0.007 : 0.000964
20191212-213848-ckpt391 : 0.9865±0.003 : 0.9991 : 0.9223±0.006 : 0.001062
20191213-034255-ckpt400 : 0.9957±0.001 : 0.9999 : 0.9908±0.003 : 0.001164

模型集成测试结果：
assemble:_20191127-005344_20191127 :                                                        0.9725±0.003 : 0.9964 : 0.8083±0.009 : 0.001060
assemble:_20191127-005344-ckpt341_20191208-201809-ckpt383 :                                 0.9763±0.003 : 0.9973 : 0.8402±0.008 : 0.000964   star
assemble:_20191127-005344-ckpt391_20191208-201809 :                                         0.9760±0.003 : 0.9973 : 0.8421±0.007 : 0.000865
assemble:_20191127-005344_20191208-195740_20191208-201809 :                                 0.9749±0.003 : 0.9974 : 0.8424±0.009 : 0.000961
assemble:_20191127-005344_20191127-005344_20191208-201809 :                                 0.9755±0.003 : 0.9972 : 0.8412±0.006 : 0.000865
assemble:_20191208-195740-ckpt396_20191208-201809-ckpt352 :                                 0.9754±0.002 : 0.9972 : 0.8355±0.011 : 0.000966
assemble:_20191127-005344-ckpt341_20191208-201809-ckpt352 :                                 0.9750±0.002 : 0.9971 : 0.8383±0.008 : 0.000864
assemble:_20191208-195740-ckpt333_20191208-201809-ckpt383 :                                 0.9743±0.003 : 0.9973 : 0.8242±0.011 : 0.001155
assemble:_20191208-195740-ckpt396_20191208-201809-ckpt383 :                                 0.9762±0.003 : 0.9974 : 0.8413±0.009 : 0.000965
assemble:_20191127-005344_20191127-005344_20191208-195740_20191208-201809 :                 0.9749±0.002 : 0.9973 : 0.8449±0.015 : 0.000870
assemble:_20191127-005344_20191127-005344_20190517-231453_20191208-195740_20191208-201809 : 0.9749±0.003 : 0.9972 : 0.8420±0.015 : 0.000866
assemble:_20191208-201809-ckpt352_20191208-201809-ckpt383 :                                 0.9744±0.003 : 0.9969 : 0.8112±0.013 : 0.001060
assemble:_20191208-201809-ckpt352_20191212-213848-ckpt386 :                                 0.9850±0.003 : 0.9988 : 0.9064±0.008 : 0.001062
assemble:_20191208-201809-ckpt383_20191212-213848-ckpt386 :                                 0.9839±0.003 : 0.9989 : 0.9200±0.010 : 0.001055
"""



# 这部分注释已废弃，可以删除
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 冯艳晓	 tar=0.9923@far=0.0000	 false predict name=待定
test name: 云轶舟	 tar=0.7802@far=0.0000	 false predict name=待定
test name: 徐骋远	 tar=0.8447@far=0.0000	 false predict name=待定
test name: 龚翔	 tar=0.9202@far=0.0000	 false predict name=待定
test name: 吴张勇	 tar=0.5150@far=0.0054	 false predict name=待定
test name: 张振宇	 tar=0.9412@far=0.0000	 false predict name=待定
test name: 徐黎明	 tar=0.8133@far=0.0000	 false predict name=待定
test name: 胡成楠	 tar=0.9500@far=0.0000	 false predict name=待定
test name: 朱见平	 tar=0.4735@far=0.0066	 false predict name=待定
test name: 夏俊	 tar=0.8306@far=0.0018	 false predict name=待定
test name: 郑华晨	 tar=0.9205@far=0.0000	 false predict name=待定
test name: 陈贤波	 tar=0.7788@far=0.0000	 false predict name=待定
test name: 赵成伟	 tar=1.0000@far=0.0000	 false predict name=待定
test name: 戚佳宇	 tar=0.5747@far=0.0000	 false predict name=待定
test name: 罗鹏	 tar=0.8709@far=0.0000	 false predict name=待定
test name: 纪书保	 tar=0.9404@far=0.0023	 false predict name=待定
total tar=0.8121@far=0.0017

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.6, is_similarity=True)
test name: 冯艳晓	 tar=0.8923@far=0.0000	 false predict name=待定
test name: 云轶舟	 tar=0.4231@far=0.0000	 false predict name=待定
test name: 徐骋远	 tar=0.2919@far=0.0000	 false predict name=待定
test name: 龚翔	 tar=0.6626@far=0.0000	 false predict name=待定
test name: 吴张勇	 tar=0.2480@far=0.0000	 false predict name=待定
test name: 张振宇	 tar=0.7255@far=0.0000	 false predict name=待定
test name: 徐黎明	 tar=0.6800@far=0.0000	 false predict name=待定
test name: 胡成楠	 tar=0.9000@far=0.0000	 false predict name=待定
test name: 朱见平	 tar=0.0497@far=0.0066	 false predict name=待定
test name: 夏俊	 tar=0.6041@far=0.0000	 false predict name=待定
test name: 郑华晨	 tar=0.7748@far=0.0000	 false predict name=待定
test name: 陈贤波	 tar=0.4087@far=0.0000	 false predict name=待定
test name: 赵成伟	 tar=0.8710@far=0.0000	 false predict name=待定
test name: 戚佳宇	 tar=0.2184@far=0.0000	 false predict name=待定
test name: 罗鹏	 tar=0.6901@far=0.0000	 false predict name=待定
test name: 纪书保	 tar=0.8014@far=0.0000	 false predict name=待定
total tar=0.5746@far=0.0005

# 朱见平本来只有一张手机拍摄的照片，添加一张机器人展厅拍摄的照片再次测试。
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 张振宇	 tar=48/51=0.9412@far=0/51=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 罗鹏	 tar=607/697=0.8709@far=0/697=0.0000
test name: 龚翔	 tar=150/163=0.9202@far=0/163=0.0000
test name: 冯艳晓	 tar=129/130=0.9923@far=0/130=0.0000
test name: 云轶舟	 tar=142/182=0.7802@far=0/182=0.0000
test name: 朱见平	 tar=243/301=0.8073@far=2/301=0.0066
test name: 郑华晨	 tar=139/151=0.9205@far=0/151=0.0000
test name: 陈贤波	 tar=164/208=0.7885@far=0/208=0.0000
test name: 纪书保	 tar=807/856=0.9428@far=2/856=0.0023
test name: 吴张勇	 tar=192/367=0.5232@far=2/367=0.0054
test name: 徐骋远	 tar=136/161=0.8447@far=0/161=0.0000
test name: 徐黎明	 tar=61/75=0.8133@far=0/75=0.0000
test name: 夏俊	 tar=453/543=0.8343@far=1/543=0.0018
test name: 戚佳宇	 tar=50/87=0.5747@far=0/87=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
total tar=0.8389@far=0.0017

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.6, is_similarity=True)
test name: 张振宇	 tar=37/51=0.7255@far=0/51=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 罗鹏	 tar=481/697=0.6901@far=0/697=0.0000
test name: 龚翔	 tar=108/163=0.6626@far=0/163=0.0000
test name: 冯艳晓	 tar=116/130=0.8923@far=0/130=0.0000
test name: 云轶舟	 tar=77/182=0.4231@far=0/182=0.0000
test name: 朱见平	 tar=212/301=0.7043@far=2/301=0.0066
test name: 郑华晨	 tar=117/151=0.7748@far=0/151=0.0000
test name: 陈贤波	 tar=85/208=0.4087@far=0/208=0.0000
test name: 纪书保	 tar=689/856=0.8049@far=0/856=0.0000
test name: 吴张勇	 tar=92/367=0.2507@far=0/367=0.0000
test name: 徐骋远	 tar=47/161=0.2919@far=0/161=0.0000
test name: 徐黎明	 tar=51/75=0.6800@far=0/75=0.0000
test name: 夏俊	 tar=329/543=0.6059@far=0/543=0.0000
test name: 戚佳宇	 tar=19/87=0.2184@far=0/87=0.0000
test name: 赵成伟	 tar=54/62=0.8710@far=0/62=0.0000
total tar=0.6246@far=0.0005


# 罗鹏增加两张手机拍摄照片，一张戴眼镜，一张不戴眼镜
test name: 冯艳晓	 tar=129/130=0.9923@far=0/130=0.0000
test name: 吴张勇	 tar=178/367=0.4850@far=5/367=0.0136
test name: 云轶舟	 tar=143/182=0.7857@far=0/182=0.0000
test name: 陈贤波	 tar=161/208=0.7740@far=0/208=0.0000
test name: 朱见平	 tar=243/301=0.8073@far=2/301=0.0066
test name: 纪书保	 tar=802/856=0.9369@far=2/856=0.0023
test name: 龚翔	 tar=152/163=0.9325@far=0/163=0.0000
test name: 徐骋远	 tar=136/161=0.8447@far=0/161=0.0000
test name: 罗鹏	 tar=607/697=0.8709@far=0/697=0.0000
test name: 胡成楠	 tar=19/20=0.9500@far=0/20=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 徐黎明	 tar=61/75=0.8133@far=0/75=0.0000
test name: 郑华晨	 tar=140/151=0.9272@far=0/151=0.0000
test name: 张振宇	 tar=48/51=0.9412@far=0/51=0.0000
test name: 夏俊	 tar=458/543=0.8435@far=1/543=0.0018
test name: 戚佳宇	 tar=50/87=0.5747@far=0/87=0.0000
total tar=0.8360@far=0.0025

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.6, is_similarity=True)
test name: 冯艳晓	 tar=116/130=0.8923@far=0/130=0.0000
test name: 吴张勇	 tar=83/367=0.2262@far=1/367=0.0027
test name: 云轶舟	 tar=77/182=0.4231@far=0/182=0.0000
test name: 陈贤波	 tar=84/208=0.4038@far=0/208=0.0000
test name: 朱见平	 tar=212/301=0.7043@far=2/301=0.0066
test name: 纪书保	 tar=686/856=0.8014@far=0/856=0.0000
test name: 龚翔	 tar=111/163=0.6810@far=0/163=0.0000
test name: 徐骋远	 tar=50/161=0.3106@far=0/161=0.0000
test name: 罗鹏	 tar=508/697=0.7288@far=0/697=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 赵成伟	 tar=54/62=0.8710@far=0/62=0.0000
test name: 徐黎明	 tar=51/75=0.6800@far=0/75=0.0000
test name: 郑华晨	 tar=125/151=0.8278@far=0/151=0.0000
test name: 张振宇	 tar=39/51=0.7647@far=0/51=0.0000
test name: 夏俊	 tar=336/543=0.6188@far=0/543=0.0000
test name: 戚佳宇	 tar=19/87=0.2184@far=0/87=0.0000
total tar=0.6337@far=0.0007

"""


# 20180402-114759 底库手机拍摄，底库15个人，共28张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.64, is_similarity=True)
test name: 云轶舟	 tar=174/182=0.9560@far=0/182=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/罗鹏/20190708-1118594808_0.png
test name: 罗鹏	 tar=478/697=0.6858@far=1/697=0.0014
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
test name: 张振宇	 tar=46/51=0.9020@far=0/51=0.0000
test name: 陈贤波	 tar=197/208=0.9471@far=0/208=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190813-0833056542_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190813-0833092943_0.png
test name: 纪书保	 tar=788/856=0.9206@far=2/856=0.0023
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/龚翔/20190817-1852089228_0.png
test name: 龚翔	 tar=140/164=0.8537@far=1/164=0.0061
test name: 郑华晨	 tar=140/151=0.9272@far=0/151=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/徐黎明/20190827-1149167616_0.png
test name: 徐黎明	 tar=64/75=0.8533@far=1/75=0.0133
test name: 朱见平	 tar=53/302=0.1755@far=0/302=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1429425587_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1528087492_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1440275035_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536438521_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190725-0906455383_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1039456899_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526485500_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526523218_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190725-0905001050_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536451530_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190725-0905037693_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1131368622_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1531029272_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1045276139_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1935133986_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1432228929_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1045316456_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1432219029_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1037562599_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1116112854_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1122415746_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1528034797_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430528729_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1151533205_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536445251_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1530568845_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190725-0906477952_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1037575319_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536464526_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1042479259_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1529205791_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1042582513_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1528046375_0_1.png
test name: 吴张勇	 tar=146/367=0.3978@far=33/367=0.0899
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/徐骋远/20190823-1948297513_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/徐骋远/20190823-1941491256_0.png
test name: 徐骋远	 tar=33/161=0.2050@far=2/161=0.0124
test name: 赵成伟	 tar=59/62=0.9516@far=0/62=0.0000
total tar=2466/3426=0.7198@far=40/3426=0.0117

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.64, is_similarity=True)
test name: 云轶舟	 tar=124/182=0.6813@far=0/182=0.0000
test name: 罗鹏	 tar=231/697=0.3314@far=0/697=0.0000
test name: 冯艳晓	 tar=107/130=0.8231@far=0/130=0.0000
test name: 张振宇	 tar=37/51=0.7255@far=0/51=0.0000
test name: 陈贤波	 tar=161/208=0.7740@far=0/208=0.0000
test name: 纪书保	 tar=587/856=0.6857@far=0/856=0.0000
test name: 龚翔	 tar=77/164=0.4695@far=0/164=0.0000
test name: 郑华晨	 tar=118/151=0.7815@far=0/151=0.0000
test name: 徐黎明	 tar=30/75=0.4000@far=0/75=0.0000
test name: 朱见平	 tar=17/302=0.0563@far=0/302=0.0000
test name: 胡成楠	 tar=6/20=0.3000@far=0/20=0.0000
test name: 吴张勇	 tar=55/367=0.1499@far=0/367=0.0000
test name: 徐骋远	 tar=5/161=0.0311@far=0/161=0.0000
test name: 赵成伟	 tar=53/62=0.8548@far=0/62=0.0000
total tar=1608/3426=0.4694@far=0/3426=0.0000
"""


# 20180402-114759 底库为机器人摄像头拍摄，底库15个人，共17张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 张振宇	 tar=44/51=0.8627@far=0/51=0.0000
test name: 胡成楠	 tar=19/20=0.9500@far=0/20=0.0000
test name: 徐骋远	 tar=101/161=0.6273@far=0/161=0.0000
test name: 冯艳晓	 tar=128/130=0.9846@far=0/130=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190813-1153194900_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1503563340_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190813-1153202415_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1504129965_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1504049446_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1508026594_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1504109102_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-0944083122_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1504097775_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1508079250_0.png
test name: 纪书保	 tar=759/856=0.8867@far=10/856=0.0117
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190829-1358566264_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1530592719_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526495256_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190829-1355098815_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1037562599_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1056092729_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430528729_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1105362494_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190829-1358595651_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1118209323_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1528517796_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1529205791_0_1.png
test name: 吴张勇	 tar=114/367=0.3106@far=12/367=0.0327
test name: 郑华晨	 tar=142/151=0.9404@far=0/151=0.0000
test name: 徐黎明	 tar=55/75=0.7333@far=0/75=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/陈贤波/20190910-1224345209_0_1.png
test name: 陈贤波	 tar=158/208=0.7596@far=1/208=0.0048
test name: 龚翔	 tar=97/164=0.5915@far=0/164=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/朱见平/20190811-1806552995_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/朱见平/20190811-1806492035_0.png
test name: 朱见平	 tar=234/302=0.7748@far=2/302=0.0066
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/罗鹏/20190704-2025394183_0.png
test name: 罗鹏	 tar=575/697=0.8250@far=1/697=0.0014
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 云轶舟	 tar=53/182=0.2912@far=0/182=0.0000
total tar=2536/3426=0.7402@far=26/3426=0.0076

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.62, is_similarity=True)
test name: 张振宇	 tar=33/51=0.6471@far=0/51=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 徐骋远	 tar=36/161=0.2236@far=0/161=0.0000
test name: 冯艳晓	 tar=119/130=0.9154@far=0/130=0.0000
test name: 纪书保	 tar=510/856=0.5958@far=0/856=0.0000
test name: 吴张勇	 tar=47/367=0.1281@far=0/367=0.0000
test name: 郑华晨	 tar=123/151=0.8146@far=0/151=0.0000
test name: 徐黎明	 tar=36/75=0.4800@far=0/75=0.0000
test name: 陈贤波	 tar=83/208=0.3990@far=0/208=0.0000
test name: 龚翔	 tar=39/164=0.2378@far=0/164=0.0000
test name: 朱见平	 tar=215/302=0.7119@far=0/302=0.0000
test name: 罗鹏	 tar=301/697=0.4319@far=0/697=0.0000
test name: 赵成伟	 tar=38/62=0.6129@far=0/62=0.0000
test name: 云轶舟	 tar=10/182=0.0549@far=0/182=0.0000
total tar=1608/3426=0.4694@far=0/3426=0.0000
"""


# 20191127-005344-ckpt-341 底库为手机拍摄，底库15个人，共28张照片；测试样本共14人，共3422张图片
"""
test name: 徐骋远	 tar=42/161=0.2609@far=0/161=0.0000
test name: 张振宇	 tar=44/51=0.8627@far=0/51=0.0000
test name: 罗鹏	 tar=477/697=0.6844@far=0/697=0.0000
test name: 赵成伟	 tar=58/62=0.9355@far=0/62=0.0000
test name: 郑华晨	 tar=133/151=0.8808@far=0/151=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536451530_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536292613_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1042243050_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536445251_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1536464526_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190812-1042582513_0.png
test name: 吴张勇	 tar=94/367=0.2561@far=6/367=0.0163
test name: 冯艳晓	 tar=127/130=0.9769@far=0/130=0.0000
test name: 龚翔	 tar=139/164=0.8476@far=0/164=0.0000
test name: 云轶舟	 tar=155/182=0.8516@far=0/182=0.0000
test name: 纪书保	 tar=650/855=0.7602@far=0/855=0.0000
test name: 朱见平	 tar=122/299=0.4080@far=0/299=0.0000
test name: 陈贤波	 tar=150/208=0.7212@far=0/208=0.0000
test name: 徐黎明	 tar=56/75=0.7467@far=0/75=0.0000
total tar=2265/3422=0.6619@far=6/3422=0.0018

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.56, is_similarity=True)
test name: 徐骋远	 tar=29/161=0.1801@far=0/161=0.0000
test name: 张振宇	 tar=40/51=0.7843@far=0/51=0.0000
test name: 罗鹏	 tar=323/697=0.4634@far=0/697=0.0000
test name: 赵成伟	 tar=53/62=0.8548@far=0/62=0.0000
test name: 郑华晨	 tar=123/151=0.8146@far=0/151=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 吴张勇	 tar=31/367=0.0845@far=0/367=0.0000
test name: 冯艳晓	 tar=118/130=0.9077@far=0/130=0.0000
test name: 龚翔	 tar=116/164=0.7073@far=0/164=0.0000
test name: 云轶舟	 tar=113/182=0.6209@far=0/182=0.0000
test name: 纪书保	 tar=485/855=0.5673@far=0/855=0.0000
test name: 朱见平	 tar=41/299=0.1371@far=0/299=0.0000
test name: 陈贤波	 tar=108/208=0.5192@far=0/208=0.0000
test name: 徐黎明	 tar=29/75=0.3867@far=0/75=0.0000
total tar=1627/3422=0.4755@far=0/3422=0.0000
"""


# 20191127-005344-ckpt-341 底库为机器人摄像头拍摄，底库15个人，共17张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.50, is_similarity=True)
test name: 陈贤波	 tar=145/208=0.6971@far=0/208=0.0000
test name: 徐骋远	 tar=114/161=0.7081@far=0/161=0.0000
test name: 胡成楠	 tar=19/20=0.9500@far=0/20=0.0000
test name: 吴张勇	 tar=103/367=0.2807@far=0/367=0.0000
test name: 云轶舟	 tar=87/182=0.4780@far=0/182=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190813-1153202415_0.png
test name: 纪书保	 tar=717/855=0.8386@far=1/855=0.0012
test name: 龚翔	 tar=115/164=0.7012@far=0/164=0.0000
test name: 徐黎明	 tar=35/75=0.4667@far=0/75=0.0000
test name: 郑华晨	 tar=140/151=0.9272@far=0/151=0.0000
test name: 张振宇	 tar=36/51=0.7059@far=0/51=0.0000
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 冯艳晓	 tar=127/130=0.9769@far=0/130=0.0000
test name: 朱见平	 tar=230/299=0.7692@far=0/299=0.0000
test name: 罗鹏	 tar=542/697=0.7776@far=0/697=0.0000
total tar=2467/3422=0.7209@far=1/3422=0.0003

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.52, is_similarity=True)
test name: 陈贤波	 tar=133/208=0.6394@far=0/208=0.0000
test name: 徐骋远	 tar=98/161=0.6087@far=0/161=0.0000
test name: 胡成楠	 tar=19/20=0.9500@far=0/20=0.0000
test name: 吴张勇	 tar=90/367=0.2452@far=0/367=0.0000
test name: 云轶舟	 tar=76/182=0.4176@far=0/182=0.0000
test name: 纪书保	 tar=673/855=0.7871@far=0/855=0.0000
test name: 龚翔	 tar=100/164=0.6098@far=0/164=0.0000
test name: 徐黎明	 tar=33/75=0.4400@far=0/75=0.0000
test name: 郑华晨	 tar=132/151=0.8742@far=0/151=0.0000
test name: 张振宇	 tar=31/51=0.6078@far=0/51=0.0000
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 冯艳晓	 tar=127/130=0.9769@far=0/130=0.0000
test name: 朱见平	 tar=225/299=0.7525@far=0/299=0.0000
test name: 罗鹏	 tar=528/697=0.7575@far=0/697=0.0000
total tar=2322/3422=0.6786@far=0/3422=0.0000
"""


# 20191208-201809-ckpt-352 底库为手机拍摄，底库15个人，共28张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 朱见平	 tar=162/299=0.5418@far=0/299=0.0000
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 纪书保	 tar=731/855=0.8550@far=0/855=0.0000
test name: 徐骋远	 tar=59/161=0.3665@far=0/161=0.0000
test name: 张振宇	 tar=44/51=0.8627@far=0/51=0.0000
test name: 郑华晨	 tar=134/151=0.8874@far=0/151=0.0000
test name: 云轶舟	 tar=166/182=0.9121@far=0/182=0.0000
test name: 龚翔	 tar=134/164=0.8171@far=0/164=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1440133929_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526125760_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430528729_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1440185351_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430505173_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430485965_0.png
test name: 吴张勇	 tar=48/367=0.1308@far=6/367=0.0163
test name: 徐黎明	 tar=50/75=0.6667@far=0/75=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 罗鹏	 tar=469/697=0.6729@far=0/697=0.0000
test name: 陈贤波	 tar=176/208=0.8462@far=0/208=0.0000
test name: 冯艳晓	 tar=124/130=0.9538@far=0/130=0.0000
total tar=2372/3422=0.6932@far=6/3422=0.0018

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.56, is_similarity=True)
test name: 朱见平	 tar=55/299=0.1839@far=0/299=0.0000
test name: 赵成伟	 tar=55/62=0.8871@far=0/62=0.0000
test name: 纪书保	 tar=621/855=0.7263@far=0/855=0.0000
test name: 徐骋远	 tar=37/161=0.2298@far=0/161=0.0000
test name: 张振宇	 tar=39/51=0.7647@far=0/51=0.0000
test name: 郑华晨	 tar=127/151=0.8411@far=0/151=0.0000
test name: 云轶舟	 tar=145/182=0.7967@far=0/182=0.0000
test name: 龚翔	 tar=107/164=0.6524@far=0/164=0.0000
test name: 吴张勇	 tar=13/367=0.0354@far=0/367=0.0000
test name: 徐黎明	 tar=23/75=0.3067@far=0/75=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 罗鹏	 tar=344/697=0.4935@far=0/697=0.0000
test name: 陈贤波	 tar=146/208=0.7019@far=0/208=0.0000
test name: 冯艳晓	 tar=114/130=0.8769@far=0/130=0.0000
total tar=1844/3422=0.5389@far=0/3422=0.0000
"""


# 20191208-201809-ckpt-352 底库为机器人摄像头拍摄，底库15个人，共17张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 张振宇	 tar=41/51=0.8039@far=0/51=0.0000
test name: 纪书保	 tar=658/855=0.7696@far=0/855=0.0000
test name: 吴张勇	 tar=115/367=0.3134@far=0/367=0.0000
test name: 赵成伟	 tar=58/62=0.9355@far=0/62=0.0000
test name: 郑华晨	 tar=129/151=0.8543@far=0/151=0.0000
test name: 冯艳晓	 tar=126/130=0.9692@far=0/130=0.0000
test name: 朱见平	 tar=235/299=0.7860@far=0/299=0.0000
test name: 徐骋远	 tar=103/161=0.6398@far=0/161=0.0000
test name: 云轶舟	 tar=89/182=0.4890@far=0/182=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 龚翔	 tar=94/164=0.5732@far=0/164=0.0000
test name: 陈贤波	 tar=152/208=0.7308@far=0/208=0.0000
test name: 罗鹏	 tar=525/697=0.7532@far=0/697=0.0000
test name: 徐黎明	 tar=52/75=0.6933@far=0/75=0.0000
total tar=2395/3422=0.6999@far=0/3422=0.0000

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.43, is_similarity=True)
test name: 张振宇	 tar=43/51=0.8431@far=0/51=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1503563340_0.png
test name: 纪书保	 tar=767/855=0.8971@far=1/855=0.0012
test name: 吴张勇	 tar=177/367=0.4823@far=0/367=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 郑华晨	 tar=145/151=0.9603@far=0/151=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
test name: 朱见平	 tar=252/299=0.8428@far=0/299=0.0000
test name: 徐骋远	 tar=144/161=0.8944@far=0/161=0.0000
test name: 云轶舟	 tar=123/182=0.6758@far=0/182=0.0000
test name: 胡成楠	 tar=19/20=0.9500@far=0/20=0.0000
test name: 龚翔	 tar=140/164=0.8537@far=0/164=0.0000
test name: 陈贤波	 tar=176/208=0.8462@far=0/208=0.0000
test name: 罗鹏	 tar=590/697=0.8465@far=0/697=0.0000
test name: 徐黎明	 tar=60/75=0.8000@far=0/75=0.0000
total tar=2828/3422=0.8264@far=1/3422=0.0003
"""


# 20191208-201809-ckpt-383 底库为手机拍摄，底库15个人，共28张照片；测试样本共14人，共3422张图片
"""
test name: 徐骋远	 tar=76/161=0.4720@far=0/161=0.0000
test name: 徐黎明	 tar=63/75=0.8400@far=0/75=0.0000
test name: 罗鹏	 tar=522/697=0.7489@far=0/697=0.0000
test name: 陈贤波	 tar=177/208=0.8510@far=0/208=0.0000
test name: 云轶舟	 tar=158/182=0.8681@far=0/182=0.0000
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 冯艳晓	 tar=119/130=0.9154@far=0/130=0.0000
test name: 朱见平	 tar=232/299=0.7759@far=0/299=0.0000
test name: 龚翔	 tar=140/164=0.8537@far=0/164=0.0000
test name: 郑华晨	 tar=132/151=0.8742@far=0/151=0.0000
test name: 张振宇	 tar=47/51=0.9216@far=0/51=0.0000
test name: 纪书保	 tar=757/855=0.8854@far=0/855=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1528595165_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526125760_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1529205791_0_1.png
test name: 吴张勇	 tar=149/367=0.4060@far=3/367=0.0082
total tar=2647/3422=0.7735@far=3/3422=0.0009

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.60, is_similarity=True)
test name: 徐骋远	 tar=32/161=0.1988@far=0/161=0.0000
test name: 徐黎明	 tar=33/75=0.4400@far=0/75=0.0000
test name: 罗鹏	 tar=327/697=0.4692@far=0/697=0.0000
test name: 陈贤波	 tar=110/208=0.5288@far=0/208=0.0000
test name: 云轶舟	 tar=94/182=0.5165@far=0/182=0.0000
test name: 赵成伟	 tar=53/62=0.8548@far=0/62=0.0000
test name: 胡成楠	 tar=14/20=0.7000@far=0/20=0.0000
test name: 冯艳晓	 tar=91/130=0.7000@far=0/130=0.0000
test name: 朱见平	 tar=133/299=0.4448@far=0/299=0.0000
test name: 龚翔	 tar=92/164=0.5610@far=0/164=0.0000
test name: 郑华晨	 tar=122/151=0.8079@far=0/151=0.0000
test name: 张振宇	 tar=41/51=0.8039@far=0/51=0.0000
test name: 纪书保	 tar=554/855=0.6480@far=0/855=0.0000
test name: 吴张勇	 tar=29/367=0.0790@far=0/367=0.0000
total tar=1725/3422=0.5041@far=0/3422=0.0000
"""


# 20191208-201809-ckpt-383 底库为机器人摄像头拍摄，底库15个人，共17张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/纪书保/20190809-1503563340_0.png
test name: 纪书保	 tar=649/855=0.7591@far=1/855=0.0012
test name: 徐骋远	 tar=98/161=0.6087@far=0/161=0.0000
test name: 云轶舟	 tar=98/182=0.5385@far=0/182=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 徐黎明	 tar=48/75=0.6400@far=0/75=0.0000
test name: 郑华晨	 tar=126/151=0.8344@far=0/151=0.0000
test name: 张振宇	 tar=42/51=0.8235@far=0/51=0.0000
test name: 吴张勇	 tar=121/367=0.3297@far=0/367=0.0000
test name: 罗鹏	 tar=543/697=0.7791@far=0/697=0.0000
test name: 赵成伟	 tar=59/62=0.9516@far=0/62=0.0000
test name: 龚翔	 tar=80/164=0.4878@far=0/164=0.0000
test name: 陈贤波	 tar=133/208=0.6394@far=0/208=0.0000
test name: 朱见平	 tar=242/299=0.8094@far=0/299=0.0000
test name: 冯艳晓	 tar=124/130=0.9538@far=0/130=0.0000
total tar=2381/3422=0.6958@far=1/3422=0.0003

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.56, is_similarity=True)
test name: 纪书保	 tar=506/855=0.5918@far=0/855=0.0000
test name: 徐骋远	 tar=46/161=0.2857@far=0/161=0.0000
test name: 云轶舟	 tar=76/182=0.4176@far=0/182=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 徐黎明	 tar=39/75=0.5200@far=0/75=0.0000
test name: 郑华晨	 tar=115/151=0.7616@far=0/151=0.0000
test name: 张振宇	 tar=37/51=0.7255@far=0/51=0.0000
test name: 吴张勇	 tar=56/367=0.1526@far=0/367=0.0000
test name: 罗鹏	 tar=455/697=0.6528@far=0/697=0.0000
test name: 赵成伟	 tar=55/62=0.8871@far=0/62=0.0000
test name: 龚翔	 tar=27/164=0.1646@far=0/164=0.0000
test name: 陈贤波	 tar=89/208=0.4279@far=0/208=0.0000
test name: 朱见平	 tar=223/299=0.7458@far=0/299=0.0000
test name: 冯艳晓	 tar=111/130=0.8538@far=0/130=0.0000
total tar=1853/3422=0.5415@far=0/3422=0.0000
"""


# 20191212-213848-ckpt-386 底库为机器人摄像头拍摄，底库15个人，共17张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 徐黎明	 tar=69/75=0.9200@far=0/75=0.0000
test name: 云轶舟	 tar=152/182=0.8352@far=0/182=0.0000
test name: 郑华晨	 tar=147/151=0.9735@far=0/151=0.0000
test name: 徐骋远	 tar=160/161=0.9938@far=0/161=0.0000
test name: 吴张勇	 tar=162/367=0.4414@far=0/367=0.0000
test name: 纪书保	 tar=828/855=0.9684@far=0/855=0.0000
test name: 陈贤波	 tar=169/208=0.8125@far=0/208=0.0000
test name: 朱见平	 tar=281/299=0.9398@far=0/299=0.0000
test name: 罗鹏	 tar=587/697=0.8422@far=0/697=0.0000
test name: 张振宇	 tar=44/51=0.8627@far=0/51=0.0000
test name: 龚翔	 tar=142/164=0.8659@far=0/164=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
total tar=2953/3422=0.8629@far=0/3422=0.0000

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.41, is_similarity=True)
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 徐黎明	 tar=72/75=0.9600@far=0/75=0.0000
test name: 云轶舟	 tar=170/182=0.9341@far=0/182=0.0000
test name: 郑华晨	 tar=151/151=1.0000@far=0/151=0.0000
test name: 徐骋远	 tar=161/161=1.0000@far=0/161=0.0000
test name: 吴张勇	 tar=263/367=0.7166@far=0/367=0.0000
test name: 纪书保	 tar=843/855=0.9860@far=0/855=0.0000
test name: 陈贤波	 tar=185/208=0.8894@far=0/208=0.0000
test name: 朱见平	 tar=296/299=0.9900@far=0/299=0.0000
test name: 罗鹏	 tar=637/697=0.9139@far=0/697=0.0000
test name: 张振宇	 tar=46/51=0.9020@far=0/51=0.0000
test name: 龚翔	 tar=162/164=0.9878@far=0/164=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
total tar=3198/3422=0.9345@far=0/3422=0.0000

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.4, is_similarity=True)
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 徐黎明	 tar=73/75=0.9733@far=0/75=0.0000
test name: 云轶舟	 tar=171/182=0.9396@far=0/182=0.0000
test name: 郑华晨	 tar=151/151=1.0000@far=0/151=0.0000
test name: 徐骋远	 tar=161/161=1.0000@far=0/161=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1429449084_0.png
test name: 吴张勇	 tar=273/367=0.7439@far=1/367=0.0027
test name: 纪书保	 tar=845/855=0.9883@far=0/855=0.0000
test name: 陈贤波	 tar=186/208=0.8942@far=0/208=0.0000
test name: 朱见平	 tar=296/299=0.9900@far=0/299=0.0000
test name: 罗鹏	 tar=639/697=0.9168@far=0/697=0.0000
test name: 张振宇	 tar=46/51=0.9020@far=0/51=0.0000
test name: 龚翔	 tar=162/164=0.9878@far=0/164=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
total tar=3215/3422=0.9395@far=1/3422=0.0003
"""


# 20191212-213848-ckpt-386 底库为手机拍摄，底库15个人，共28张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 徐骋远	 tar=27/161=0.1677@far=0/161=0.0000
test name: 徐黎明	 tar=59/75=0.7867@far=0/75=0.0000
test name: 罗鹏	 tar=654/697=0.9383@far=0/697=0.0000
test name: 龚翔	 tar=154/164=0.9390@far=0/164=0.0000
test name: 冯艳晓	 tar=124/130=0.9538@far=0/130=0.0000
test name: 郑华晨	 tar=134/151=0.8874@far=0/151=0.0000
test name: 纪书保	 tar=840/855=0.9825@far=0/855=0.0000
test name: 张振宇	 tar=48/51=0.9412@far=0/51=0.0000
test name: 陈贤波	 tar=185/208=0.8894@far=0/208=0.0000
test name: 云轶舟	 tar=114/182=0.6264@far=0/182=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1429449084_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526523218_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430016414_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1429432080_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430517105_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1529134396_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1432228929_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1526125760_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430004254_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430528729_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430505173_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1432205878_0_1.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1529205791_0_1.png
test name: 吴张勇	 tar=164/367=0.4469@far=13/367=0.0354
test name: 赵成伟	 tar=60/62=0.9677@far=0/62=0.0000
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 朱见平	 tar=260/299=0.8696@far=0/299=0.0000
total tar=2843/3422=0.8308@far=13/3422=0.0038

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.59, is_similarity=True)
test name: 徐骋远	 tar=8/161=0.0497@far=0/161=0.0000
test name: 徐黎明	 tar=37/75=0.4933@far=0/75=0.0000
test name: 罗鹏	 tar=609/697=0.8737@far=0/697=0.0000
test name: 龚翔	 tar=137/164=0.8354@far=0/164=0.0000
test name: 冯艳晓	 tar=73/130=0.5615@far=0/130=0.0000
test name: 郑华晨	 tar=109/151=0.7219@far=0/151=0.0000
test name: 纪书保	 tar=810/855=0.9474@far=0/855=0.0000
test name: 张振宇	 tar=44/51=0.8627@far=0/51=0.0000
test name: 陈贤波	 tar=145/208=0.6971@far=0/208=0.0000
test name: 云轶舟	 tar=17/182=0.0934@far=0/182=0.0000
test name: 吴张勇	 tar=67/367=0.1826@far=0/367=0.0000
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 胡成楠	 tar=18/20=0.9000@far=0/20=0.0000
test name: 朱见平	 tar=177/299=0.5920@far=0/299=0.0000
total tar=2308/3422=0.6745@far=0/3422=0.0000
"""


# 20191213-034255-final.ckpt-400 底库为手机拍摄，底库15个人，共28张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.55, is_similarity=True)
test name: 朱见平	 tar=295/299=0.9866@far=0/299=0.0000
test name: 赵成伟	 tar=57/62=0.9194@far=0/62=0.0000
test name: 张振宇	 tar=51/51=1.0000@far=0/51=0.0000
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 陈贤波	 tar=205/208=0.9856@far=0/208=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
test name: 吴张勇	 tar=50/367=0.1362@far=0/367=0.0000
test name: 纪书保	 tar=852/855=0.9965@far=0/855=0.0000
test name: 郑华晨	 tar=151/151=1.0000@far=0/151=0.0000
test name: 云轶舟	 tar=168/182=0.9231@far=0/182=0.0000
test name: 龚翔	 tar=163/164=0.9939@far=0/164=0.0000
test name: 徐黎明	 tar=75/75=1.0000@far=0/75=0.0000
test name: 徐骋远	 tar=3/161=0.0186@far=0/161=0.0000
test name: 罗鹏	 tar=694/697=0.9957@far=0/697=0.0000
total tar=2914/3422=0.8515@far=0/3422=0.0000

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 朱见平	 tar=298/299=0.9967@far=0/299=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 张振宇	 tar=51/51=1.0000@far=0/51=0.0000
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 陈贤波	 tar=207/208=0.9952@far=0/208=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1429449084_0.png
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1430528729_0.png
test name: 吴张勇	 tar=254/367=0.6921@far=2/367=0.0054
test name: 纪书保	 tar=852/855=0.9965@far=0/855=0.0000
test name: 郑华晨	 tar=151/151=1.0000@far=0/151=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/云轶舟/20190813-1759417695_0.png
test name: 云轶舟	 tar=176/182=0.9670@far=1/182=0.0055
test name: 龚翔	 tar=164/164=1.0000@far=0/164=0.0000
test name: 徐黎明	 tar=75/75=1.0000@far=0/75=0.0000
test name: 徐骋远	 tar=13/161=0.0807@far=0/161=0.0000
test name: 罗鹏	 tar=697/697=1.0000@far=0/697=0.0000
total tar=3150/3422=0.9205@far=3/3422=0.0009
"""


# 20191213-034255-final.ckpt-400 底库为机器人摄像头拍摄，底库15个人，共17张照片；测试样本共14人，共3422张图片
"""
metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.5, is_similarity=True)
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 罗鹏	 tar=697/697=1.0000@far=0/697=0.0000
test name: 郑华晨	 tar=151/151=1.0000@far=0/151=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/吴张勇/20190809-1429449084_0.png
test name: 吴张勇	 tar=259/367=0.7057@far=1/367=0.0027
test name: 纪书保	 tar=853/855=0.9977@far=0/855=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
test name: 徐骋远	 tar=161/161=1.0000@far=0/161=0.0000
test name: 云轶舟	 tar=179/182=0.9835@far=0/182=0.0000
test name: 张振宇	 tar=51/51=1.0000@far=0/51=0.0000
/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest/陈贤波/20190809-0919327259_0_1.png
test name: 陈贤波	 tar=205/208=0.9856@far=1/208=0.0048
test name: 龚翔	 tar=163/164=0.9939@far=0/164=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 朱见平	 tar=298/299=0.9967@far=0/299=0.0000
test name: 徐黎明	 tar=75/75=1.0000@far=0/75=0.0000
total tar=3304/3422=0.9655@far=2/3422=0.0006

metric(tests_name, tests_path, tests_emb, bases_name, bases_path, bases_emb, subtract_mean=subtract_mean, distance_metric=distance_metric, threshold=0.52, is_similarity=True)
test name: 胡成楠	 tar=20/20=1.0000@far=0/20=0.0000
test name: 罗鹏	 tar=694/697=0.9957@far=0/697=0.0000
test name: 郑华晨	 tar=151/151=1.0000@far=0/151=0.0000
test name: 吴张勇	 tar=234/367=0.6376@far=0/367=0.0000
test name: 纪书保	 tar=851/855=0.9953@far=0/855=0.0000
test name: 冯艳晓	 tar=130/130=1.0000@far=0/130=0.0000
test name: 徐骋远	 tar=161/161=1.0000@far=0/161=0.0000
test name: 云轶舟	 tar=178/182=0.9780@far=0/182=0.0000
test name: 张振宇	 tar=51/51=1.0000@far=0/51=0.0000
test name: 陈贤波	 tar=205/208=0.9856@far=0/208=0.0000
test name: 龚翔	 tar=163/164=0.9939@far=0/164=0.0000
test name: 赵成伟	 tar=62/62=1.0000@far=0/62=0.0000
test name: 朱见平	 tar=298/299=0.9967@far=0/299=0.0000
test name: 徐黎明	 tar=75/75=1.0000@far=0/75=0.0000
total tar=3273/3422=0.9565@far=0/3422=0.0000
"""





















"""
model-20190517-221518.ckpt-1
best_threshold: 2.48 ~ 2.72

model-20190517-225714.ckpt-2
best_threshold: 2.52 ~ 2.72

model-20190518-034602.ckpt-150
best_threshold: 1.59
"""


# 20180402-114759 : 0.9556±0.003 : 0.9915 : 0.6544±0.017 : 0.001061


# TODO:
"""
1. 加入底库不是同一个摄像头
2. 是否有必要加入手机等高清设备拍摄的照片
3. 不同光照背景下的阈值是否不相同，逆光
4. 
"""


