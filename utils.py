
from monai.inferers import sliding_window_inference
import os
import random
import torch.nn.functional as F
import numpy as np
import torch
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
import pandas as pd


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return (left, dim - right)
    else:
        return (0, dim)


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    else:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def pad_image_and_label(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    pad_todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    pad_list = [0, 0]
    for to_pad in pad_todos:
        if to_pad[0]:
            pad_list.insert(0, to_pad[1])
            pad_list.insert(0, to_pad[2])
        else:
            pad_list.insert(0, 0)
            pad_list.insert(0, 0)
    if np.sum(pad_list) != 0:
        image = F.pad(image, pad_list, 'constant')
    if seg is not None:
        if np.sum(pad_list) != 0:
            seg = F.pad(seg, pad_list, 'constant')
        return image, seg, pad_list
    return image, seg, pad_list


def pad_or_crop_image(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    crop_list = [z_slice, y_slice, x_slice]
    image = image[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    if seg is not None:
        seg = seg[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    image, seg, pad_list = pad_image_and_label(image, seg)

    return image, seg, pad_list, crop_list


def normalize(image):
    min_ = torch.min(image)
    max_ = torch.max(image)
    scale_ = max_ - min_
    image = (image - min_) / scale_
    return image


def minmax(image, low_perc=1, high_perc=99):
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = torch.clip(image, low, high)
    image = normalize(image)
    return image


def regulate_data(image, label, target_size=(128, 128, 128), mode='train'):
    nonzero_index = torch.nonzero(torch.sum(image, axis=0) != 0)
    z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
    zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    image = image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()

    for i in range(image.shape[0]):
        image[i] = minmax(image[i])

    label = label[:, zmin:zmax, ymin:ymax, xmin:xmax]

    if mode == 'train':
        image, label, pad_list, crop_list = pad_or_crop_image(image, label, target_size)
        return image, label
    else:
        d, h, w = image.shape[1:]
        pad_d = (128 - d) if 128 - d > 0 else 0
        pad_h = (128 - h) if 128 - h > 0 else 0
        pad_w = (128 - w) if 128 - w > 0 else 0
        image, label, pad_list = pad_image_and_label(image, label, target_size=(d + pad_d, h + pad_h, w + pad_w))
        return image, label


def save_best_model(model, args):
    path = args.checkpoint_path
    path = os.path.join(path, 'best.pt')
    torch.save(model.state_dict(), path)
    return


def save_checkpoint(model, optimizer, epoch, scheduler, path, best_loss):
    path = os.path.join(path, 'checkpoint.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
    }, path)
    return


def load_checkpoint(model, optimizer, scheduler, args):
    path = args.checkpoint_path
    path = os.path.join(path, 'checkpoint.pt')
    if not os.path.exists(path):
        return 0, np.inf
    else:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['best_loss']


def cal_confuse(preds, targets, patient=0):
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"
    labels = ["ET", "TC", "WT"]
    confuse_list = []
    for i, label in enumerate(labels):
        if torch.sum(targets[i]) == 0 and torch.sum(targets[i] == 0):
            tp = tn = fp = fn = 0
            sens = spec = 1
        elif torch.sum(targets[i]) == 0:
            print(f'{patient} did not have {label}')
            sens = tp = fn = 0
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            spec = tn / (tn + fp)
        else:
            tp = torch.sum(torch.logical_and(preds[i], targets[i]))
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            fn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
        confuse_list.append([sens, spec])
    return confuse_list


def cal_dice_confuse(predict, target):
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    dice = DiceMetric(include_background=True)
    p_et = predict[0]
    p_tc = predict[1]
    p_wt = predict[2]
    t_et = target[0]
    t_tc = target[1]
    t_wt = target[2]
    p_et, p_tc, p_wt, t_et, t_tc, t_wt = p_et.unsqueeze(0).unsqueeze(0), p_tc.unsqueeze(0).unsqueeze(0), p_wt.unsqueeze(
        0).unsqueeze(0), t_et.unsqueeze(0).unsqueeze(0), t_tc.unsqueeze(0).unsqueeze(0), t_wt.unsqueeze(0).unsqueeze(0)

    if torch.sum(p_et) != 0 and torch.sum(t_et) != 0:
        et_dice = float(dice(p_et, t_et).cpu().numpy())
        et_hd = float(haussdor(p_et, t_et).cpu().numpy())
    elif torch.sum(p_et) == 0 and torch.sum(t_et) == 0:
        et_dice = 1
        et_hd = 0
    elif (torch.sum(p_et) == 0 and torch.sum(t_et) != 0) or (torch.sum(p_et) != 0 and torch.sum(t_et) == 0):
        et_dice = 0
        et_hd = 347
    if torch.sum(p_tc) != 0 and torch.sum(t_tc) != 0:
        tc_dice = float(dice(p_tc, t_tc).cpu().numpy())
        tc_hd = float(haussdor(p_tc, t_tc).cpu().numpy())
    elif torch.sum(p_tc) == 0 and torch.sum(t_tc) == 0:
        tc_dice = 1
        tc_hd = 0
    elif (torch.sum(p_tc) == 0 and torch.sum(t_tc) != 0) or (torch.sum(p_tc) != 0 and torch.sum(t_tc) == 0):
        tc_dice = 0
        tc_hd = 347
    if torch.sum(p_wt) != 0 and torch.sum(t_wt) != 0:
        wt_dice = float(dice(p_wt, t_wt).cpu().numpy())
        wt_hd = float(haussdor(p_wt, t_wt).cpu().numpy())
    elif torch.sum(p_wt) == 0 and torch.sum(t_wt) == 0:
        wt_dice = 1
        wt_hd = 0
    elif (torch.sum(p_wt) == 0 and torch.sum(t_wt) != 0) or (torch.sum(p_wt) != 0 and torch.sum(t_wt) == 0):
        wt_dice = 0
        wt_hd = 347

    confuse_metric = cal_confuse(predict, target)
    et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
    et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
    return dict(id='0', et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice,
                et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
                et_sens=float(et_sens), tc_sens=float(tc_sens), wt_sens=float(wt_sens),
                et_spec=float(et_spec), tc_spec=float(tc_spec), wt_spec=float(wt_spec))


def post_processing(model, image):
    def inference(model, image):
        return sliding_window_inference(inputs=image, roi_size=(128, 128, 128), sw_batch_size=2, predictor=model,
                                        overlap=0.6)

    predict = inference(model, image)
    predict += inference(model, image.flip(dims=(2,))).flip(dims=(2,))
    predict += inference(model, image.flip(dims=(3,))).flip(dims=(3,))
    predict += inference(model, image.flip(dims=(4,))).flip(dims=(4,))
    predict += inference(model, image.flip(dims=(2, 3))).flip(dims=(2, 3))
    predict += inference(model, image.flip(dims=(2, 4))).flip(dims=(2, 4))
    predict += inference(model, image.flip(dims=(3, 4))).flip(dims=(3, 4))
    predict += inference(model, image.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4))

    predict = predict / 8.0
    return predict


def Show_Param(args):
    print('exp-name:', args.exp_name)
    print('mode:', args.mode)
    print('learning_rate', args.lr)
    print('num_epoch:', args.end_epoch)
    print('batch-size:', args.batch_size)
    return


def write_log(writer, epoch, val_avg_loss, train_avg_loss, lr):
    writer.add_scalar(tag='train_avg_loss',
                      scalar_value=train_avg_loss,
                      global_step=epoch
                      )
    writer.add_scalar(tag='test/test_avg_Dice_loss',
                      scalar_value=val_avg_loss,
                      global_step=epoch
                      )
    writer.add_scalar(tag='test/test_avg_Dice',
                      scalar_value=1 - val_avg_loss,
                      global_step=epoch
                      )
    writer.add_scalar(tag='learn_rate',
                      scalar_value=lr,
                      global_step=epoch
                      )


def save_seg_csv(csv_path, name, csv):
    try:
        val_metrics = pd.DataFrame.from_records(csv)
        columns = ['id', 'et_dice', 'tc_dice', 'wt_dice', 'et_hd', 'tc_hd', 'wt_hd', 'et_sens', 'tc_sens', 'wt_sens',
                   'et_spec', 'tc_spec', 'wt_spec']
        val_metrics.to_csv(f'{str(csv_path)}/metrics.csv', index=False, columns=columns)
    except KeyboardInterrupt:
        print("Save CSV File Error!")


def get_ture_dice(predict, target):
    predict = predict.squeeze()
    target = target.squeeze()
    p_et = predict[0]
    p_tc = predict[1]
    p_wt = predict[2]
    t_et = target[0]
    t_tc = target[1]
    t_wt = target[2]
    p_et, p_tc, p_wt, t_et, t_tc, t_wt = p_et.unsqueeze(0).unsqueeze(0), p_tc.unsqueeze(0).unsqueeze(0), p_wt.unsqueeze(
        0).unsqueeze(0), t_et.unsqueeze(0).unsqueeze(0), t_tc.unsqueeze(0).unsqueeze(0), t_wt.unsqueeze(0).unsqueeze(0)
    dice = DiceMetric(include_background=True)
    et_dice = float(dice(p_et, t_et).cpu())
    tc_dice = float(dice(p_tc, t_tc).cpu())
    wt_dice = float(dice(p_wt, t_wt).cpu())

    if torch.sum(p_et) == 0 and torch.sum(t_et) == 0:
        et_dice = 1
    elif (torch.sum(p_et) == 0 and torch.sum(t_et) != 0) or (torch.sum(p_et) != 0 and torch.sum(t_et) == 0):
        et_dice = 0

    if torch.sum(p_tc) == 0 and torch.sum(t_tc) == 0:
        tc_dice = 1
    elif (torch.sum(p_tc) == 0 and torch.sum(t_tc) != 0) or (torch.sum(p_tc) != 0 and torch.sum(t_tc) == 0):
        tc_dice = 0

    if torch.sum(p_wt) == 0 and torch.sum(t_wt) == 0:
        wt_dice = 1
    elif (torch.sum(p_wt) == 0 and torch.sum(t_wt) != 0) or (torch.sum(p_wt) != 0 and torch.sum(t_wt) == 0):
        wt_dice = 0

    return (et_dice + tc_dice + wt_dice) / 3


def save_param(args):
    param = vars(args)
    txt_path = os.path.join(args.checkpoint_path,'param.txt')
    with open(txt_path,'w') as f:
        for X in param:
            f.write(X + ':' + str(param[X]) + '\n')

