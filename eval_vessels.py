import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from skimage import filters
import torch

from csl_common.utils.nn import to_numpy
import vlight
import predict_vessels


def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    assert np.max(pred_vessels) <= 1.0 and np.min(pred_vessels) >= 0.0
    assert np.max(true_vessels) == 1.0 and np.min(true_vessels) == 0.0
    assert np.max(masks) == 1.0 and np.min(masks) == 0.0
    assert pred_vessels.shape[0] == true_vessels.shape[0] and masks.shape[0] == true_vessels.shape[0]
    assert pred_vessels.shape[1] == true_vessels.shape[1] and masks.shape[1] == true_vessels.shape[1]
    assert pred_vessels.shape[2] == true_vessels.shape[2] and masks.shape[2] == true_vessels.shape[2]

    if split_by_img:
        n = pred_vessels.shape[0]
        return (np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]),
                np.array([pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]))
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()


def _f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def f1_score(true_vessels, pred_vessels, masks=None):
    if masks is None:
        vessels_in_mask, preds_in_mask = true_vessels, pred_vessels
    else:
        vessels_in_mask, preds_in_mask = pixel_values_in_mask(true_vessels, pred_vessels, masks)

    t = time.time()
    precision, recall, thresholds = precision_recall_curve(
        vessels_in_mask.flatten().astype(bool), preds_in_mask.flatten(), pos_label=1)
    print(f'pr curve: {int(1000 * (time.time() - t))}ms')

    return best_f1_threshold(precision, recall, thresholds)


def best_f1_threshold(precision, recall, thresholds):
    best_f1, best_threshold = -1., None
    for index in np.linspace(0, len(precision), num=5000, endpoint=False, dtype=int):
        curr_f1 = _f1_score(precision[index], recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]
    return best_f1, best_threshold


def misc_measures_evaluation(true_vessels, pred_vessels_bin):
    TP = np.count_nonzero(true_vessels & pred_vessels_bin)
    FN = np.count_nonzero(true_vessels & ~pred_vessels_bin)
    TN = np.count_nonzero(~true_vessels & ~pred_vessels_bin)
    FP = np.count_nonzero(~true_vessels & pred_vessels_bin)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = _f1_score(precision, sensitivity)
    return acc, sensitivity, specificity, f1


def threshold_vessel_probs(probs, threshold, masks, flatten=True):
    pred_vessels_bin = np.zeros(probs.shape, dtype=np.uint8)
    pred_vessels_bin[probs >= threshold] = 1
    if flatten:
        return pred_vessels_bin[masks == 1].flatten()
    else:
        return pred_vessels_bin


def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)

    precision, recall, thresholds = precision_recall_curve(
        vessels_in_mask.flatten().astype(bool), generated_in_mask.flatten(), pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin[masks == 1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin


def difference_map(ori_vessel, pred_vessel, mask):
    # ori_vessel : an RGB image
    thresholded_vessel = threshold_by_f1(np.expand_dims(ori_vessel, axis=0),
                                         np.expand_dims(pred_vessel, axis=0),
                                         np.expand_dims(mask, axis=0),
                                         flatten=False)

    thresholded_vessel = np.squeeze(thresholded_vessel, axis=0)
    diff_map = np.zeros((ori_vessel.shape[0], ori_vessel.shape[1], 3))

    # Green (overlapping)
    diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)] = (0, 255, 0)
    # Red (false negative, missing in pred)
    diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)] = (255, 0, 0)
    # Blue (false positive)
    diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)] = (0, 0, 255)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)])
    fn = len(diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)])
    fp = len(diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)])

    return diff_map, 2. * overlap / (2 * overlap + fn + fp)


def show_segmentation_results(orig_image, recon, preds, gt_mask=None, foreground_mask=None, threshold=0.5):
    """ Show results for one image """

    fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
    if torch.is_tensor(preds):
        preds = preds.squeeze().squeeze()
    else:
        preds = preds.squeeze()

    if foreground_mask is None:
        foreground_mask = np.ones_like(preds).astype(np.uint8)

    if gt_mask is not None:
        diff_map, _ = difference_map(gt_mask, preds, foreground_mask)
    else:
        diff_map = np.zeros_like(preds)

    if gt_mask is None:
        gt_mask = np.zeros_like(preds).astype(np.uint8)

    pred_mask = to_numpy((preds > threshold).squeeze())
    gt_mask = to_numpy(gt_mask)

    # probs = np.clip(probs, a_min=0, a_max=1)
    imgfname = f'./outputs/results/{modelname}/hrf_{idx + 1:02d}_probs.png'
    io_utils.makedirs(imgfname)
    cv2.imwrite(imgfname, (preds * 255).astype(np.uint8))
    imgfname = f'./outputs/results/{modelname}/hrf_{idx + 1:02d}_diff.png'
    cv2.imwrite(imgfname, cv2.cvtColor((diff_map).astype(np.uint8), cv2.COLOR_RGB2BGR))
    imgfname = f'./outputs/results/{modelname}/hrf_{idx + 1:02d}_orig.png'
    cv2.imwrite(imgfname, cv2.cvtColor((orig_image).astype(np.uint8), cv2.COLOR_RGB2BGR))

    ax[0,0].imshow(orig_image)
    ax[0,1].imshow(gt_mask)
    # ax[0,2].imshow(errors.astype(np.uint8))
    ax[0,2].imshow(diff_map.astype(np.uint8))

    ax[1,0].imshow(vis.to_disp_image(recon.squeeze(), denorm=True))
    # ax[1,1].imshow(preds, vmin=-1, vmax=1)
    ax[1,1].imshow(preds, vmax=1)
    ax[1,2].imshow(pred_mask.astype(np.uint8))
    plt.tight_layout()


def calculate_metrics(preds, gt_vessels, fov_masks=None, full_eval=False, verbose=False):

    assert len(preds) == len(gt_vessels)

    if not isinstance(preds, np.ndarray):
        preds = to_numpy(preds)

    if not isinstance(gt_vessels, np.ndarray):
        gt_vessels = to_numpy(gt_vessels)

    assert isinstance(gt_vessels, np.ndarray)

    if len(gt_vessels.shape) == 2:
        gt_vessels, preds  = gt_vessels[np.newaxis], preds[np.newaxis]

    if fov_masks is not None:
        if not isinstance(fov_masks, np.ndarray):
            fov_masks = np.array(fov_masks)
        if len(fov_masks.shape) == 2:
            fov_masks = fov_masks[np.newaxis]
        gt_vessels_in_mask, pred_vessels_in_mask = pixel_values_in_mask(gt_vessels, preds, fov_masks)
    else:
        gt_vessels_in_mask, pred_vessels_in_mask = gt_vessels, preds

    y_true = to_numpy(gt_vessels_in_mask).ravel() >= 1
    y_score = to_numpy(pred_vessels_in_mask).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    thresholds = np.fliplr([thresholds])[0]
    AUC_prec_rec = np.trapz(precision, recall)
    average_precision = AUC_prec_rec

    results = {}
    results['PR'] = average_precision

    if full_eval:

        best_f1, best_f1_th = best_f1_threshold(precision, recall, thresholds)
        results['F1'] = best_f1
        results['F1_th'] = best_f1_th

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc = auc(fpr, tpr)
        results['ROC'] = roc

        otsu_threshold = filters.threshold_otsu(pred_vessels_in_mask)
        y_pred_bin = pred_vessels_in_mask >= otsu_threshold
        acc, se, sp, f1 = misc_measures_evaluation(y_true, y_pred_bin)
        results['otsu_th'] = otsu_threshold
        results['otsu_SE'] = se
        results['otsu_SP'] = sp
        results['otsu_ACC'] = acc
        results['otsu_F1'] = f1

        fixed_threshold = 0.5
        y_pred_bin = pred_vessels_in_mask >= fixed_threshold
        acc, se, sp, f1 = misc_measures_evaluation(y_true, y_pred_bin)
        results['th_SE'] = se
        results['th_SP'] = sp
        results['th_ACC'] = acc
        results['th_F1'] = f1

        if verbose:
            print(f"F1 score : {best_f1:.4f} (th={best_f1_th:.3f})")
            print(f"F1 score : {f1:.4f} (th={fixed_threshold:.3f})")
            print(f"SE/SP/ACC: {se:.4f}, {sp:.4f}, {acc:.4f} (th={fixed_threshold:.3f})")
            print('AUC PR: {0:0.4f}'.format(average_precision))
            print('AUC ROC: {0:0.4f}'.format(roc))

    return results


if __name__ == '__main__':

    import os
    import cv2
    import time
    import config as cfg
    from csl_common import vis
    from csl_common.utils import io_utils
    from training import bool_str
    import retinadataset
    import unet

    np.set_printoptions(linewidth=np.inf)

    scales = {
        'drive': [2,3,4],
        'chase': [1,1.5,2],
        'hrf': [1]
    }

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='drive', choices=['drive', 'chase', 'hrf'], type=str)
    parser.add_argument('--model', default='vlight', choices=['unet', 'vlight'], type=str)
    parser.add_argument('--modelname', default='drive_vlight', type=str)
    parser.add_argument('--patch-size', default=512, type=int)
    parser.add_argument('--show', type=bool_str, default=False, help='show results')
    parser.add_argument('--gpu', type=bool_str, default=True)
    args = parser.parse_args()

    model = args.model
    dsname = args.dataset
    modelname = args.modelname

    root, cache_root = cfg.get_dataset_paths(dsname)
    dataset_cls = cfg.get_dataset_class(dsname)
    dataset = dataset_cls(root=root, train=False)
    print(dataset)

    model_dir = './'

    if model == 'unet':
        net = unet.load_net(os.path.join(model_dir, modelname)).cuda()
    elif model == 'vlight':
        net = vlight.load_net(os.path.join(model_dir, modelname))
        if args.gpu:
            net = net.cuda()
    else:
        raise ValueError

    net.eval()

    results_probs = []
    gt_masks = []
    fov_masks = []

    results = []
    t_tot = 0

    for idx in range(len(dataset))[:]:

        data = dataset[idx]
        image_id = data['fname']
        full_image = to_numpy(data['image'])
        gt_mask = to_numpy(data['mask'])//255
        fov_mask = dataset.fov_masks[image_id]

        print(f'\n---- Testing image {idx+1}: {image_id} ---- ')

        t = time.perf_counter()
        recon, probs = predict_vessels.segment_image(net, full_image, patch_size=args.patch_size, scales=scales[dsname])#, gpu=args.gpu)

        # probs_lr = np.fliplr(predict_vessels.segment_image(net, np.fliplr(full_image), scales=scales[dsname], patch_size=args.patch_size)[1])
        # probs_ud = np.flipud(predict_vessels.segment_image(net, np.flipud(full_image), scales=scales[dsname], patch_size=args.patch_size)[1])
        # probs = (probs + probs_lr + probs_ud) / 3

        t_image = time.perf_counter() - t
        print(f'prediction time: {int(1000*t_image)}ms')
        t_tot += t_image

        # print('Calculating metrics...')
        res = calculate_metrics(probs, gt_mask, fov_masks=fov_mask, full_eval=True, verbose=False)
        res['idx'] = idx
        res['fname'] = image_id
        results.append(res)

        if args.show:
            show_segmentation_results(full_image, full_image, probs, gt_mask, fov_mask)
            plt.show()

        # probs = np.clip(probs, a_min=0, a_max=1.0)
        results_probs.append(probs)
        gt_masks.append(gt_mask)
        fov_masks.append(fov_mask)

        # save as png to disk
        probs = np.clip(probs, a_min=0, a_max=1)
        imgfname = f'./outputs/results/{dsname}/{modelname}/{idx+1:02d}_test.png'
        io_utils.makedirs(imgfname)
        cv2.imwrite(imgfname, (probs*255).astype(np.uint8))

    print(f'Total time: {t_tot}s')
    print(f"\nModelname: {modelname}")
    print(f"Test on  : {dsname}")
    print(f'\n==== Results per image ==== \n')
    import pandas as pd
    df = pd.DataFrame(results)
    with pd.option_context('display.max_rows', None, 'display.max_columns', 500, 'display.width', 1000):
        print(df)
        print("-"*100)
        print("")
        print(df.describe())
        print("")

    print(f'\n==== Totals ==== \n')
    calculate_metrics(results_probs, gt_masks, fov_masks=fov_masks, full_eval=True, verbose=True)
    print("")



