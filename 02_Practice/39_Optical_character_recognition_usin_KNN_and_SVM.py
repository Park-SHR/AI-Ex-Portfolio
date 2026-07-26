import cv2
import numpy as np

CELL_SIZE = 20
NCLASSES = 10
TRAIN_RATIO = 0.8

digits_img = cv2.imread('../03_data/digits.png', 0)
digits = [
    np.hsplit(row, digits_img.shape[1] // CELL_SIZE)
    for row in np.vsplit(digits_img, digits_img.shape[0] // CELL_SIZE)
]
digits = np.array(digits).reshape(-1, CELL_SIZE, CELL_SIZE)
nsamples = digits.shape[0]
labels = np.repeat(np.arange(NCLASSES), nsamples // NCLASSES)

for i in range(nsamples):
    m = cv2.moments(digits[i])
    if m['mu02'] > 1e-3:
        s = m['mu11'] / m['mu02']
        M = np.float32([
            [1, s, -0.5 * CELL_SIZE * s],
            [0, 1, 0]
        ])
        digits[i] = cv2.warpAffine(digits[i], M, (CELL_SIZE, CELL_SIZE))

perm = np.random.permutation(nsamples)
digits = digits[perm]
labels = labels[perm]

ntrain = int(TRAIN_RATIO * nsamples)
ntest = nsamples - ntrain


def calc_hog(digits):
    win_size = (20, 20)
    block_size = (10, 10)
    block_stride = (10, 10)
    cell_size = (10, 10)
    nbins = 9

    hog = cv2.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins
    )

    samples = []
    for d in digits:
        samples.append(hog.compute(d))

    return np.array(samples, np.float32)


fea_hog_train = calc_hog(digits[:ntrain])
fea_hog_test = calc_hog(digits[ntrain:])
labels_train, labels_test = labels[:ntrain], labels[ntrain:]

K = 3
knn_model = cv2.ml.KNearest_create()
knn_model.train(fea_hog_train, cv2.ml.ROW_SAMPLE, labels_train)

svm_model = cv2.ml.SVM_create()
svm_model.setGamma(2)
svm_model.setC(1)
svm_model.setKernel(cv2.ml.SVM_RBF)
svm_model.setType(cv2.ml.SVM_C_SVC)
svm_model.train(fea_hog_train, cv2.ml.ROW_SAMPLE, labels_train)


def eval_model(fea, labels, f_pred):
    pred = f_pred(fea).astype(np.int32)
    acc = (pred.T == labels).mean() * 100

    conf_mat = np.zeros((NCLASSES, NCLASSES), np.int32)
    for c_gt, c_pred in zip(labels, pred):
        conf_mat[c_gt, c_pred] += 1

    return acc, conf_mat


knn_acc, knn_conf_mat = eval_model(
    fea_hog_test,
    labels_test,
    lambda fea: knn_model.findNearest(fea, K)[1]
)

print('KNN accuracy (%):', knn_acc)
print('KNN confusion matrix:')
print(knn_conf_mat)

svm_acc, svm_conf_mat = eval_model(
    fea_hog_test,
    labels_test,
    lambda fea: svm_model.predict(fea)[1]
)

print('SVM accuracy (%):', svm_acc)
print('SVM confusion matrix:')
print(svm_conf_mat)