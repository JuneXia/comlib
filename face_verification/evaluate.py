#! /usr/bin/python
import os
import sys
import argparse
import math
import numpy as np

RELEASE = False
if RELEASE:
    sys.path.append('/disk1/home/xiaj/dev/FlaskFace_debug')
else:
    sys.path.append('/home/xiajun/dev/FlaskFace')

from face_identification import faceid_pipeline
from sklearn.model_selection import KFold
from scipy import interpolate
from utils import dataset as datset
from utils import tools
import shutil
from multiprocessing import Process
from multiprocessing import Manager


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


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
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(best_threshold, dist[test_set],
                                                      actual_issame[test_set])
        # print('[calculate_roc]: best_threshold: ', best_threshold)

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
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
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    s_same = max(1e-10, float(n_same))
    n_diff = max(1e-10, float(n_diff))
    val = float(true_accept) / s_same
    far = float(false_accept) / n_diff
    return val, far


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds_roc = np.arange(0, 4, 0.2)  # TODO: 这个阈值的取值范围可以写到全局的地方去。
    #embeddings1 = embeddings[0::2]
    #embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds_roc, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds_val = np.arange(0, 4, 0.01)
    val, val_std, far = calculate_val(thresholds_val, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, thresholds_roc, accuracy, val, val_std, far


def extract_embeddings(faceid_model_path, image_pairs, emb_dict):
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

    faceid_model = faceid_pipeline.FaceID(faceid_model_path)
    embeddings1 = faceid_model.embedding(image_pairs[:, 0],
                                         use_fixed_image_standardization=use_fixed_image_standardization,
                                         random_rotate=random_rotate, random_crop=random_crop, random_flip=random_flip,
                                         fixed_contract=fixed_contract)
    embeddings2 = faceid_model.embedding(image_pairs[:, 1],
                                         use_fixed_image_standardization=use_fixed_image_standardization,
                                         random_rotate=random_rotate, random_crop=random_crop, random_flip=random_flip,
                                         fixed_contract=fixed_contract)
    emb_dict['embeddings1'] = embeddings1
    emb_dict['embeddings2'] = embeddings2
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


def model_evaluate(faceid_models, image_pairs, distance_metric):
    eval_infos = []
    emb_dict = Manager().dict()
    for i, faceid_model in enumerate(faceid_models):
        p = Process(target=extract_embeddings, args=(faceid_model, image_pairs, emb_dict))
        p.start()
        p.join()

        embeddings1, embeddings2 = emb_dict['embeddings1'], emb_dict['embeddings2']
        tpr, fpr, thresholds_roc, accuracy, val, val_std, far = evaluate(embeddings1, embeddings2, image_pairs_label,
                                                                         nrof_folds=10,
                                                                         distance_metric=distance_metric,
                                                                         subtract_mean=True)
        auc = tools.compute_auc(fpr, tpr)
        model_name = faceid_model.split('/')[-1]
        model_name = model_name.split('.')[0]
        model_name = tools.strcat(model_name.split('-')[0:2], '-')

        eval_info = {'tpr': tpr, 'fpr': fpr, 'acc': accuracy, 'auc': auc,
                     'val': val, 'val_std': val_std, 'far': far,
                     'model_name': model_name}
        eval_infos.append(eval_info)

        tools.view_bar('[model_evaluate]:: loading: ', i + 1, len(faceid_models))
    print('')

    return eval_infos


def get_evaluate_pairs(data_path):
    validation_images_path, validation_images_label = datset.load_dataset(data_path, shuffle=True)
    validation_dataset = datset.SiameseDataset(validation_images_path, validation_images_label, is_train=False)

    image_pairs = []
    image_pairs_label = []
    for batch, (images, labels) in enumerate(validation_dataset):
        image_pairs.append(images)
        image_pairs_label.append(labels)
    image_pairs = np.array(image_pairs)
    image_pairs_label = np.array(image_pairs_label)

    return image_pairs, image_pairs_label


if __name__ == '__main__':
    distance_metric = 0
    if RELEASE:
        data_path = '/disk1/home/xiaj/res/face/gcface/gc_together/origin_align160_margin32'
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



"""
model-20190517-221518.ckpt-1
best_threshold: 2.48 ~ 2.72

model-20190517-225714.ckpt-2
best_threshold: 2.52 ~ 2.72

model-20190518-034602.ckpt-150
best_threshold: 1.59
"""


