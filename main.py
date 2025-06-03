import os.path
import torch.utils.data
import argparse
from PHtransBTS import PHtransBTS
import utils
import numpy as np
from loss import MyCriterion
from optimizer import LinearWarmupCosineAnnealingLR
from monai.inferers import sliding_window_inference
from torch.backends import cudnn
import random
from torch.utils.tensorboard import SummaryWriter
import BraTS
from BraTS import DataAugmenter
from timm.optim import optim_factory
import Deep_Supervision

parser = argparse.ArgumentParser(description='NET')
parser.add_argument('--lr', default=3e-4, type=float)

parser.add_argument('--exp-name', default="One", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--compute-batch-size', default=1, type=int)
parser.add_argument('--num-workers', default=2, type=int)
parser.add_argument('--end-epoch', default=400, type=int, help="Maximum iterations of the model")
parser.add_argument('--val', default=5, type=int, help="Validation frequency of the model")
parser.add_argument('--load-checkpoint', default=True, type=bool)
parser.add_argument('--seed', default=1)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--deep-supervision', default=True, type=float)
parser.add_argument('--Deepth', default=5, type=float)


folder_path = os.getcwd()
parser.add_argument('--js-path', default=os.path.join(folder_path, 'brats21_folds.json'), type=str)
parser.add_argument('--dt-path', default=r'E:\dataset\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021', type=str)

parser.add_argument('--checkpoint-path', default=os.path.join(folder_path, 'checkpoint'), type=str)
parser.add_argument('--log-path', default=os.path.join(folder_path, 'log'), type=str)
parser.add_argument('--csv-path', default=os.path.join(folder_path, 'csv'), type=str)


def init_randon(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def init_folder(args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.csv_path, exist_ok=True)
    os.makedirs(os.path.join(args.log_path, args.exp_name), exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.exp_name)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.csv_path = os.path.join(args.csv_path, args.exp_name)
    os.makedirs(args.csv_path, exist_ok=True)

def main(args):
    init_randon(args.seed)
    init_folder(args)
    utils.Show_Param(args)
    utils.save_param(args)

    device = torch.device('cuda:0')
    model = PHtransBTS(channels=(24, 80, 160, 320, 400, 400),
                   blocks=(1, 1, 1, 1, 1),
                   heads=(1, 2, 2, 4, 8),
                   r=(4, 2, 1, 1, 1),
                   deep_supervision=True,
                   branch_in=4, branch_out=3,
                   AgN=24, conv_proportion=0.8).to(device)
    if args.mode == 'train':
        train(model, args, device)
    if args.mode == 'test':
        test(model, args)


def train(model, args, device):
    writer = SummaryWriter(os.path.join(args.log_path, args.exp_name))

    if args.deep_supervision:
        criterion = Deep_Supervision.DeepCriterion(Deepth=args.Deepth).to(device)
    else:
        criterion = MyCriterion().to(device)

    optimizer = optim_factory.create_optimizer_v2(model, opt='adamw', weight_decay=args.weight_decay, lr=args.lr,
                                                  betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=args.end_epoch)

    data_aug = DataAugmenter().to(device)
    dataset = BraTS.BraTS2021(mode='train', json_path=args.js_path, data_path=args.dt_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.compute_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch, eta_min=1e-5)
    accumulation_steps = int(args.batch_size / args.compute_batch_size)
    start_epoch = 0
    best_avg_loss = np.inf

    if args.load_checkpoint:
        start_epoch, best_avg_loss = utils.load_checkpoint(model, optimizer, scheduler, args)

    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        loss_sum = 0
        torch.cuda.empty_cache()

        for i, data in enumerate(train_loader):
            image = data[0].to(device)
            label = data[1].to(device)
            image, label = data_aug(image, label)
            pred = model(image)
            loss = criterion(pred, label)
            loss = loss / accumulation_steps
            loss.backward()
            if ((i + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss
            print(f'epoch=={epoch}/{args.end_epoch}     iter=={i}/{train_loader.__len__()}      loss=={loss * accumulation_steps}')

        scheduler.step()
        utils.save_checkpoint(model, optimizer, epoch, scheduler, args.checkpoint_path, best_avg_loss)
        loss_sum = loss_sum * args.batch_size

        if epoch % args.val == 0:
            avg_loss = train_val(model, args, device)
            utils.write_log(writer, epoch, val_avg_loss=avg_loss, train_avg_loss=loss_sum / dataset.__len__(),
                            lr=scheduler.get_last_lr()[0])
            print(f'epoch == {epoch}    val_avg_loss == {avg_loss}      best_loss = {best_avg_loss}')

            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                utils.save_best_model(model, args)
        print(f'epoch == {epoch}    train_avg_loss == {loss_sum / dataset.__len__()}')

def train_val(model, args, device):
    dataset = BraTS.BraTS2021(mode='val', json_path=args.js_path, data_path=args.dt_path)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=args.num_workers)

    model.eval()
    if args.deep_supervision:
        model.do_ds = False
    loss_sum = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image = data[0].to(device)
            label = data[1].to(device)
            pred = sliding_window_inference(inputs=image, roi_size=(128, 128, 128), sw_batch_size=args.compute_batch_size, predictor=model,
                                            overlap=0.6)
            pred = (pred > 0)
            dice = utils.get_ture_dice(pred, label)
            loss = 1-dice
            loss_sum += loss
            # print(f'iter=={i}/{dataset.__len__()}      Dice_loss =={loss}       Dice=={dice}')
    if args.deep_supervision:
        model.do_ds = True
    return loss_sum / val_loader.__len__()

def test(model, args):
    metrics_dict = []
    dataset = BraTS.BraTS2021(mode='test', json_path=args.js_path, data_path=args.dt_path)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True,
                                              num_workers=args.num_workers)

    device = torch.device('cuda:0')
    args.best_model_path = os.path.join(args.checkpoint_path, 'best.pt')
    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()
    if args.deep_supervision:
        model.do_ds = False
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data[0].to(device)
            label = data[1].to(device)
            pred = utils.post_processing(model, image)
            prob = torch.sigmoid(pred)

            predict = (prob > 0.5).squeeze()
            label = label.squeeze()

            dice_metrics = utils.cal_dice_confuse(predict.cpu(), label.cpu())
            dice_metrics['id'] = data[2]
            print(dice_metrics)
            metrics_dict.append(dice_metrics)
    utils.save_seg_csv(args.csv_path, args.exp_name, metrics_dict)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
