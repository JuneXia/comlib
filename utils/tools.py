import os
import math
import sys
import skimage
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
import shutil
from sklearn.manifold import TSNE
import datetime

DEBUG = False


def view_bar(message, num, total):
    """
    :param message:
    :param num:
    :param total:
    :return:

    Example:
    view_bar('loading: ', i + 1, len(list))
    """
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def strcat(strlist, cat_mark=','):
    '''
    :param strlist: a list for store string.
    :param cat_mark:
    :return:
    '''
    line = ''
    for ln in strlist:
        line += str(ln) + cat_mark
    line = line[0:line.rfind(cat_mark)]
    return line


def steps_per_epoch(num_samples, batch_size, allow_less_batsize=True):
    """
    :param num_samples: 样本总量
    :param batch_size:
    :param allow_less_batsize: True: 允许最后一个step的样本数量小于batch_size;
                               False: 如果最后一个step的样本数量小于batch_size，则丢弃这一step的样本。
    :return:
    """
    steps = 0
    steps += num_samples // batch_size
    if allow_less_batsize:
        offset = 0 if num_samples % batch_size == 0 else 1
        steps += offset

    return steps


def get_strtime():
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return time_str


def imcrop(img, bbox, scale_ratio=2.0):
    '''
    :param img: ndarray or image_path
    :param bbox: bounding box
    :param scale_ratio: the scale_ratio used to control margin size around bbox.
    :return:
    '''
    if type(img) == str:
        img = cv2.imread(img)

    xmin, ymin, xmax, ymax = bbox
    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale_ratio
    h = (ymax - ymin) * scale_ratio

    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))

    face = img[ymin:ymax, xmin:xmax, :]
    return face


def named_standard(data_path, mark='', replace=''):
    class_list = os.listdir(data_path)
    for i, cls in enumerate(class_list):
        if cls.count(mark):
            new_cls = cls.replace(mark, replace)
            class_path = os.path.join(data_path, cls)
            new_class_path = os.path.join(data_path, new_cls)
            os.rename(class_path, new_class_path)
            print('[tools.named_standard]:: change {} to {}'.format(class_path, new_class_path))

        view_bar('[tools.load_image]:: loading: ', i + 1, len(class_list))
    print('')


def load_image(data_path, subdir='', min_num_image_per_class=1, del_under_min_num_class=False, min_area4del=0):
    '''
    :param data_path:
    :param subdir: 如果每个类别又子目录，则应该指定。
    :param min_num_image_per_class: 每个类至少min_num_image_per_class张图片，少于这个数量的类不会被加载。
    :param del_under_min_num_class: if True, 如果每个类别少于min_num_image_per_class张图片，则永久删除这个类。
    :param min_area4del: 如果图片尺寸小于该值则会被删除。设置为0即不适用该操作。
    :return:
    '''
    del_count = 0

    images_info = []
    class_list = os.listdir(data_path)
    if DEBUG:
        class_list = class_list[0:100]
    class_list.sort()
    for i, cls in enumerate(class_list):
        class_images_info = []
        class_path = os.path.join(data_path, cls)
        image_list = os.listdir(class_path)
        image_list.sort()
        for image in image_list:
            image_path = os.path.join(class_path, image)
            if os.path.isdir(image_path):
                if image == subdir:
                    sub_image_list = os.listdir(image_path)
                    sub_image_list.sort()
                    for sub_image in sub_image_list:
                        sub_image_path = os.path.join(image_path, sub_image)
                        if os.path.isfile(sub_image_path):
                            if min_area4del > 0:
                                img = cv2.imread(sub_image_path)
                                img_area = img.shape[0] * img.shape[1]
                                if img_area < min_area4del:
                                    os.remove(sub_image_path)
                                    del_count += 1

                            sub_image = os.path.join(image, sub_image)
                            class_images_info.append([cls, sub_image])
                        else:
                            # raise Exception("Error, Shouldn't exist! Please check your param or trying to reason for {}!".format(sub_image_path))
                            print(image_path)
                            if os.path.exists(class_path):
                                shutil.rmtree(class_path)
                else:
                    # raise Exception("Error, Shouldn't exist! Please check your param or trying to reason for {}!".format(image_path))
                    print(image_path)
                    if os.path.exists(class_path):
                        shutil.rmtree(class_path)
            else:
                if min_area4del > 0:
                    img = cv2.imread(image_path)
                    img_area = img.shape[0] * img.shape[1]
                    if img_area < min_area4del:
                        os.remove(image_path)
                        del_count += 1

                class_images_info.append([cls, image])

        if del_under_min_num_class and len(class_images_info) < min_num_image_per_class:
            if os.path.exists(class_path):
                del_count += 1
                shutil.rmtree(class_path)
            print('[tools.load_image]:: delete: {} image, {}'.format(len(class_images_info), class_path))
            continue
        elif del_under_min_num_class and len(cls) > 30:
            # del_count += 1
            # shutil.rmtree(class_path)
            print('[tools.load_image]:: delete: {} image, {}'.format(len(class_images_info), class_path))
        else:
            images_info.extend(class_images_info)

        view_bar('[tools.load_image]:: loading: ', i + 1, len(class_list))
    print('')
    print('[tools.load_image]:: delete {} images!'.format(del_count))
    return images_info


def load_csv(csv_file, start_idx=None, end_idx=None):
    """
    读取CSV文件
    :param csv_file:
    :param start_idx: 从start_idx这一行开始取数据
    :param end_idx: 取到end_idx这一行为止
    :return:
    """
    print('\n[load_csv]:: load {} '.format(csv_file))
    with open(csv_file) as fid:
        lines = fid.readlines()
        lines = lines[start_idx:end_idx]
        csv_info = []
        for i, line in enumerate(lines):
            csv_info.append(line.strip().split(","))
            view_bar('loading:', i + 1, len(lines))
        print('')

        csv_info = np.array(csv_info)
    return csv_info


def save_csv(templates, csv_file, title_line='\n'):
    assert len(templates.shape) >= 2
    assert len(title_line) >= 1
    assert title_line[-1] == '\n'

    print('\n[save_csv]:: save {} '.format(csv_file))
    with open(csv_file, "w") as fid:
        fid.write(title_line)
        for i in range(len(templates)):
            text = strcat(templates[i]) + "\n"
            fid.write(text)
            view_bar('saving:', i + 1, len(templates))
        print('')
    fid.close()


def config_feedata(images_info, label_axis=0, dup_base=(), validation_ratio=0.0, filter_cb=None):
    """
    有的时候为了样本均衡，需要对样本量较少的类别进行重复采样。
    注意：按照本函数思路，仅仅是对原样本进行copy来重复采样。所以在训练之前还应当做数据增强。
    :param images_info:
    :param label_axis: 指定images_info中的哪一列是类别标签信息。
    :param dup_base: 该元组长度只能是1或2，当len(dup_base)=1时，dup2size=dup_base[0];
        当len(dup_base)=2时，dup2size根据每类样本的数量线性计算得到，计算公司详见代码.
    :param validation_ratio: 0.0, 不拆分数据集；大于0小于1时，表示按该比例拆分；-1，按代码中的公式线性拆分
    :param filter_cb: 自定义对每个类别样本的回调函数
        最终每个类别的数据均衡应该同时满足dup2range和filter_cb这两个限定条件。
    :return:
    """
    VAL_STATE = 1
    TRAIN_STATE = 0
    if len(dup_base) == 0:
        dup2size_func = None
    elif len(dup_base) == 1:
        dup2size_func = lambda x: dup_base[0]
    elif len(dup_base) == 2:
        # np.random.seed(666)
        # dup2size_func = lambda x: np.random.randint(dup_base[0], dup_base[1])  # 随机重复抽样到dup2range范围内

        assert (60 > dup_base[0]) and (100 > dup_base[1])
        dupk = (100-60)/(dup_base[1]-dup_base[0])
        dupb = 60-dup_base[0]*dupk
        dup2size_func = lambda x: min(100, int(dupk*x+dupb))
    else:
        raise Exception('len(dup_base) just in range [0, 2]!')

    if validation_ratio == 0:
        val_ratio_func = None
    elif validation_ratio == -1:
        valk = (10 - 3) / (350 - 20)
        valb = 3 - 20 * valk
        val_ratio_func = lambda x: min(10, int(valk * x + valb))
    elif (validation_ratio > 0) and (validation_ratio < 1):
        val_ratio_func = lambda x: min(10, int(x*validation_ratio))
    else:
        raise Exception('validation_ratio just equal to -1 or [0, 1)!')

    extend_images_info = []
    cls_names = set(images_info[:, label_axis])
    for i, cls in enumerate(cls_names):
        info = images_info[np.where(images_info[:, label_axis] == cls)]
        np.random.shuffle(info)

        val_info = []
        if val_ratio_func is not None:
            val_size = val_ratio_func(len(info))
            val_info = info[0:val_size]
            info = info[val_size:]

        if (dup2size_func is not None) and filter_cb(info):
            dup2size = dup2size_func(len(info))
            if len(info) < dup2size:
                extend_info = []
                multiple = dup2size // len(info) - 1
                for m in range(multiple):
                    extend_info.extend(info)

                extend_info.extend(info[0:dup2size % len(info)])
                extend_info = np.array(extend_info)
                info = np.concatenate((info, extend_info))

        if len(val_info) > 0:
            info = np.concatenate((val_info, info))

        mark = np.zeros((len(info), 1), dtype=np.int32)
        mark[0:len(val_info)] = VAL_STATE
        label = np.ones([len(info), 1], dtype=np.int32) * i
        info = np.hstack((mark, label, info))
        extend_images_info.extend(info)
        view_bar('config feed data:', i + 1, len(cls_names))
    print('')
    images_info = np.array(extend_images_info)

    return images_info


def concat_dataset(root_path, images, label_names):
    """
    数据路径拼接思想：root_path/images[i]/label_names[i]
    :param root_path: 数据根路径
    :param images: 特定类别的图片子目录，如：201912341.jpg、 subdir/201912341.jpg
    :param label_names: 在路径中的类别名
    :return:
    """
    print('\n[concat_dataset]:: load {} '.format(root_path))
    images_info = []
    imexist_count = 0
    imexts = ['jpg', 'png']
    for i, (image, label_name) in enumerate(zip(images, label_names)):
        imname, imext = image.rsplit('.', maxsplit=1)
        if imext not in imexts:
            raise Exception('Only support [jpg, png] currently!')

        imexist = False
        for ext in imexts:
            image_path = os.path.join(root_path, imname) + '.' + ext
            if os.path.exists(image_path):
                image_info = [label_name, image_path]
                images_info.append(image_info)
                imexist = True
                break
        if not imexist:
            imexist_count -= 1
            print('\t{}/{}.{} is not exist!'.format(root_path, imname, imexts))

        view_bar('loading:', i + 1, len(images))
    print('')

    images_info = np.array(images_info)

    print('\n***********************************************')
    print('From {}, not existed images count: {}'.format(root_path, abs(imexist_count)))
    print('***********************************************\n')

    return images_info


def compute_auc(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    return roc_auc


def _get_plt_color(index=None):
    # cnames = list(matplotlib.colors.cnames.keys())
    # cnames.sort()
    cnames = [
        # 'aliceblue',
        'antiquewhite',
        'aqua',
        'aquamarine',
        ##'azure',
        'beige',
        'bisque',
        'black',
        'blanchedalmond',
        'blue',
        'blueviolet',
        'brown',
        'burlywood',
        'cadetblue',
        'chartreuse',
        'chocolate',
        'coral',
        'cornflowerblue',
        # 'cornsilk',
        'crimson',
        'cyan',
        'darkblue',
        'darkcyan',
        'darkgoldenrod',
        # 'darkgray',  # 和 darkgrey 有点像
        'darkgreen',
        'darkgrey',
        'darkkhaki',
        'darkmagenta',
        'darkolivegreen',
        'darkorange',
        'darkorchid',
        'darkred',
        'darksalmon',
        'darkseagreen',
        'darkslateblue',
        'darkslategray',
        'darkslategrey',
        'darkturquoise',
        'darkviolet',
        'deeppink',
        'deepskyblue',
        'dimgray',
        'dimgrey',
        'dodgerblue',
        'firebrick',
        # 'floralwhite',
        'forestgreen',
        'fuchsia',
        'gainsboro',
        # 'ghostwhite',
        'gold',
        'goldenrod',
        'gray',
        'green',
        'greenyellow',
        'grey',
        # 'honeydew',
        'hotpink',
        'indianred',
        'indigo',
        # 'ivory',
        'khaki',
        'lavender',
        'lavenderblush',
        'lawngreen',
        'lemonchiffon',
        'lightblue',
        'lightcoral',
        'lightcyan',
        'lightgoldenrodyellow',
        'lightgray',
        'lightgreen',
        'lightgrey',
        'lightpink',
        'lightsalmon',
        'lightseagreen',
        'lightskyblue',
        'lightslategray',
        'lightslategrey',
        'lightsteelblue',
        # 'lightyellow',
        'lime',
        'limegreen',
        'linen',
        'magenta',
        'maroon',
        'mediumaquamarine',
        'mediumblue',
        'mediumorchid',
        'mediumpurple',
        'mediumseagreen',
        'mediumslateblue',
        'mediumspringgreen',
        'mediumturquoise',
        'mediumvioletred',
        # 'midnightblue',
        'mintcream',
        'mistyrose',
        'moccasin',
        'navajowhite',
        'navy',
        # 'oldlace',
        'olive',
        'olivedrab',
        'orange',
        'orangered',
        'orchid',
        'palegoldenrod',
        'palegreen',
        'paleturquoise',
        'palevioletred',
        'papayawhip',
        'peachpuff',
        'peru',
        'pink',
        'plum',
        'powderblue',
        'purple',
        'rebeccapurple',
        'red',
        'rosybrown',
        'royalblue',
        'saddlebrown',
        'salmon',
        'sandybrown',
        'seagreen',
        # 'seashell',
        'sienna',
        'silver',
        'skyblue',
        'slateblue',
        # 'slategray',  # 和slategrey有点像
        'slategrey',
        'snow',
        'springgreen',
        'steelblue',
        'tan',
        'teal',
        'thistle',
        'tomato',
        'turquoise',
        'violet',
        'wheat',
        # 'white',
        # 'whitesmoke',
        'yellow',
        'yellowgreen']

    if (index is None) or (index >= len(cnames)):
        index = np.random.randint(0, len(cnames))

    return cnames[index]


def _get_plt_linestyle(index=None):
    # linestyle = matplotlib.style.available
    linestyle = [#'.',
                 #',',
                 #'o',
                 #'v',
                 #'^',
                 #'<',
                 #'>',
                 #'1',
                 #'2',
                 #'3',
                 #'4',
                 #'s',
                 #'p',
                 #'*',
                 #'h',
                 #'H',
                 #'+',
                 #'x',
                 #'D',
                 #'d',
                 #'|',
                 #'_'
                 ]
    linestyle = [
        '-',
        '--',
        '-.',
        ':',
        # 'None', draw nothing, 相当于画点
        # ' ', draw nothing, 相当于画点
        # '', draw nothing, 相当于画点
    ]

    if (index is None) or (index >= len(linestyle)):
        index = np.random.randint(0, len(linestyle))

    return linestyle[index]


def _get_plt_markers(index=None):
    markers = matplotlib.markers.MarkerStyle.markers
    markers = list(markers.keys())
    markers_cp = markers.copy()
    for mark in markers_cp:
        if type(mark) in (int,):
            markers.remove(mark)
        elif mark is None:
            markers.remove(mark)
    markers.sort()

    if (index is None) or (index >= len(markers)):
        index = np.random.randint(0, len(markers))

    return markers[index]


def _get_plt_style(index=None):
    def _test():
        y = np.arange(1, 5)
        for i in range(0, 20):
            color = _get_plt_color()
            linestyle = _get_plt_linestyle()
            marker = _get_plt_markers()

            plt.plot(y + i, linewidth=1, color=color, linestyle=linestyle, marker=marker)
            y_pt = y + i
            plt.text(3, y_pt[-1], color, color=color, fontsize=16)
            print('{}, {}'.format(i, color))
        plt.show()

    color = _get_plt_color(index)
    linestyle = _get_plt_linestyle(index)
    marker = _get_plt_markers(index)

    #_test()

    return color, linestyle, marker


def plt_roc(fprs, tprs, show_labels, interpret_label='', save_path=None):
    plt.figure(figsize=(10, 10))

    for i, (fpr, tpr, show_label) in enumerate(zip(fprs, tprs, show_labels)):
        color, linestyle, marker = _get_plt_style(i)

        plt.plot(fpr, tpr, color=color, linewidth=1, linestyle=linestyle, marker=marker, label=show_label)

    plt.xlabel(r'$\rm{fpr}$', fontsize=16)
    plt.ylabel(r'$\rm{tpr}$', fontsize=16)
    plt.title(r'ROC', fontsize=16)
    plt.text(np.max(fprs)*0.6, np.max(tprs)*1.06, r'{}'.format(interpret_label), fontsize=10)
    plt.legend(loc="lower right")  # 显示图例
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_embedding(data, label, title, save_path=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    num_classes = len(set(label))

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # fontsize = min(len(data)/1200, 16)
    fontsize = 16
    print('[plot_embedding]: fontsize={}'.format(fontsize))
    for i in range(data.shape[0]):
        label_name = str(label[i])
        # color = plt.cm.Set(label[i] / num_classes)  # TODO: 有的时候会出错，暂不明什么原因。
        color = _get_plt_color(label[i])
        plt.text(data[i, 0], data[i, 1], label_name, color=color,
                 fontdict={'weight': 'bold', 'size': fontsize})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.legend()  # 显示图例
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def dim_reduct(features, rslt_dim=2):
    tsne = TSNE(n_components=rslt_dim, init='pca', random_state=1000, verbose=True)
    result = tsne.fit_transform(features)

    return result


if __name__ == '__main__':  # plt_roc
    from sklearn import svm, datasets
    from sklearn import cross_validation

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    ##变为2分类
    X, y = X[y != 2], y[y != 2]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)

    # Learn to predict each class against the other
    svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    y_score = svm.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    # plt_roc(fpr, tpr, roc_auc, save_path='./tmp.jpg')
    plt_roc([fpr, fpr], [tpr, tpr], [roc_auc, roc_auc], interpret_label='test', save_path='./tmp.jpg')

