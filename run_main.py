import argparse
import torch
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import evaluate
from utils.tools import get_parameter_number
from models import CPGRU, CPLSTM, CPMLP, CPBiGRU, CPBiLSTM, CPTransformer, PatchTST, iTransformer, Transformer, \
    DLinear, Autoformer, MLP, MICN, CNN,  \
    BiLSTM, BiGRU, GRU, LSTM
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_baseline
import time
import json
import random
import numpy as np
import os
import json
import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_baseline, load_content
parser = argparse.ArgumentParser(description='BatteryLife')

def set_seed(seed):
    accelerate.utils.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')
parser.add_argument('--dataset', type=str, default='HUST', help='dataset used for pretrained model')
parser.add_argument('--data', type=str, required=False, default='BatteryLife', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/HUST_dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--early_cycle_threshold', type=int, default=100, help='when to stop model training')
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--lstm_layers', type=int, default=1, help='num of LSTM layers')
parser.add_argument('--e_layers', type=int, default=2, help='num of intra-cycle layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of inter-cycle layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='relu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=10, help='patch length')
parser.add_argument('--stride', type=int, default=10, help='stride')
parser.add_argument('--patch_len2', type=int, default=10, help='patch length for inter-cycle patching')
parser.add_argument('--stride2', type=int, default=10, help='stride for inter-cycle patching')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--output_num', type=int, default=1, help='The number of prediction targets')
parser.add_argument('--class_num', type=int, default=8, help='The number of life classes')

# optimization
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
parser.add_argument('--weighted_sampling', action='store_true', default=False, help='use weighted sampling')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--least_epochs', type=int, default=5, help='The model is trained at least some epoches before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function, [MSE, BMSE, MAPE]')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--mlp', type=int, default=0)

# Evaluation alpha-accuracy
parser.add_argument('--alpha1', type=float, default=0.15, help='the 10 percent alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the 15 percent alpha for alpha-accuracy')



args = parser.parse_args()


nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
set_seed(args.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2_baseline.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, gradient_accumulation_steps=args.accumulation_steps)
accelerator.print(args.__dict__)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_lradj{}_dataset{}_loss{}_wd{}_wl{}_bs{}_s{}'.format(
        args.model,
        args.seq_len,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.lradj, args.dataset, args.loss, args.wd, args.weighted_loss, args.batch_size, args.seed)


    data_provider_func = data_provider_baseline
    if args.model == 'Transformer':
        model = Transformer.Model(args).float()
    elif args.model == 'CPBiLSTM':
        model = CPBiLSTM.Model(args).float()
    elif args.model == 'CPBiGRU':
        model = CPBiGRU.Model(args).float()
    elif args.model == 'CPGRU':
        model = CPGRU.Model(args).float()
    elif args.model == 'CPLSTM':
        model = CPLSTM.Model(args).float()
    elif args.model == 'BiLSTM':
        model = BiLSTM.Model(args).float()
    elif args.model == 'BiGRU':
        model = BiGRU.Model(args).float()
    elif args.model == 'LSTM':
        model = LSTM.Model(args).float()
    elif args.model == 'GRU':
        model = GRU.Model(args).float()
    elif args.model == 'PatchTST':
        model = PatchTST.Model(args).float()
    elif args.model == 'iTransformer':
        model = iTransformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'CPMLP':
        model = CPMLP.Model(args).float()
    elif args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'MLP':
        model = MLP.Model(args).float()
    elif args.model == 'MICN':
        model = MICN.Model(args).float()
    elif args.model == 'CNN':
        model = CNN.Model(args).float()
    elif args.model == 'MLP':
        model = MLP.Model(args).float()
    elif args.model == 'MICN':
        model = MICN.Model(args).float()
    elif args.model == 'CNN':
        model = CNN.Model(args).float()
    elif args.model == 'CPTransformer':
        model = CPTransformer.Model(args).float()
    else:
        raise Exception(f'The {args.model} is not an implemented baseline!')
        
    
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    
    
    accelerator.print("Loading training samples......")
    train_data, train_loader = data_provider_func(args, 'train', None, sample_weighted=args.weighted_sampling)
    label_scaler = train_data.return_label_scaler()  
    life_class_scaler = train_data.return_life_class_scaler()    
    accelerator.print("Loading vali samples......")
    vali_data, vali_loader = data_provider_func(args, 'val', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)
    accelerator.print("Loading test samples......")
    test_data, test_loader = data_provider_func(args, 'test', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling)

    if accelerator.is_local_main_process and os.path.exists(path):
        del_files(path)  # delete checkpoint files
        accelerator.print(f'success delete {path}')

    os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()
    joblib.dump(label_scaler, f'{path}/label_scaler')
    joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
    with open(path+'/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    if accelerator.is_local_main_process:
        wandb.init(
        # set the wandb project where this run will be logged
        project="new_LifeBaseline",
        
        # track hyperparameters and run metadata
        config=args.__dict__,
        name=nowtime
        )


    para_res = get_parameter_number(model)
    accelerator.print(para_res)
        
    for name, module in model._modules.items():
        accelerator.print(name," : ",module)
    
    
    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    trained_parameters_names = []
    for name, p in model.named_parameters():
        if p.requires_grad is True:
            trained_parameters_names.append(name)
            trained_parameters.append(p)

    accelerator.print(f'Trainable parameters are: {trained_parameters_names}')
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    elif args.lradj == 'TST':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate
        )
    else:
        # constant
        scheduler = None

    life_classes = json.load(open('data_provider/life_classes.json'))
    class_numbers = len(list(life_classes.keys()))

    criterion = nn.MSELoss(reduction='none') 


    life_class_criterion = nn.MSELoss() 

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    best_vali_loss = float('inf')
    best_vali_MAE, best_test_MAE = 0, 0
    best_vali_RMSE, best_test_RMSE = 0, 0
    best_vali_alpha_acc1, best_test_alpha_acc1 = 0, 0 
    best_vali_alpha_acc2, best_test_alpha_acc2 = 0, 0 

    best_seen_vali_alpha_acc1, best_seen_test_alpha_acc1 = 0, 0
    best_seen_vali_alpha_acc2, best_seen_test_alpha_acc2 = 0, 0
    best_unseen_vali_alpha_acc1, best_unseen_test_alpha_acc1 = 0, 0
    best_unseen_vali_alpha_acc2, best_unseen_test_alpha_acc2 = 0, 0

    best_vali_MAPE, best_test_MAPE = 0, 0
    best_seen_vali_MAPE, best_seen_test_MAPE = 0, 0
    best_unseen_vali_MAPE, best_unseen_test_MAPE = 0, 0

    for epoch in range(args.train_epochs):
        mae_metric = evaluate.load('./utils/mae')
        mape_metric = evaluate.load('./utils/mape')
        iter_count = 0
        total_loss = 0
        total_cl_loss = 0
        total_lc_loss = 0
        
        model.train()
        epoch_time = time.time()
        print_cl_loss = 0
        print_life_class_loss = 0
        std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
        total_preds, total_references = [], []
        for i, (cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids) in enumerate(train_loader):
            with accelerator.accumulate(model):
                model_optim.zero_grad()
                iter_count += 1
                
                life_class = life_class.to(accelerator.device)
                scaled_life_class = scaled_life_class.float().to(accelerator.device)
                cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
                labels = labels.float().to(accelerator.device)
                

                # encoder - decoder
                outputs = model(cycle_curve_data, curve_attn_mask)
                

                cut_off = labels.shape[0]
                    
                if args.loss == 'MSE':
                    loss = criterion(outputs[:cut_off], labels)
                    loss = torch.mean(loss * weights)
                elif args.loss == 'MAPE':
                    tmp_outputs = outputs[:cut_off] * std + mean_value
                    tmp_labels = labels * std + mean_value
                    loss = criterion(tmp_outputs/tmp_labels, tmp_labels/tmp_labels)
                    loss = torch.mean(loss * weights)
                    
                label_loss = loss.detach().float()
                
                
                print_loss = loss.detach().float()
                
                total_loss += loss.detach().float()
                total_cl_loss += print_cl_loss
                total_lc_loss += print_life_class_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off]  * std + mean_value
                all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))
                
                total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
                total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
                accelerator.backward(loss)
                model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                
                if (i + 1) % 5 == 0:
                    accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | lc_loss: {print_life_class_loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        train_rmse = root_mean_squared_error(total_references, total_preds)
        train_mape = mean_absolute_percentage_error(total_references, total_preds)
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False)
        test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, test_unseen_alpha_acc1, test_seen_alpha_acc1, test_unseen_alpha_acc2, test_seen_alpha_acc2 = vali_baseline(args, accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True)
        vali_loss = vali_mape

        
        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            best_vali_MAE = vali_mae_loss
            best_test_MAE = test_mae_loss
            best_vali_RMSE = vali_rmse
            best_test_RMSE = test_rmse
            best_vali_MAPE = vali_mape
            best_test_MAPE = test_mape

            # alpha-accuracy
            best_vali_alpha_acc1 = vali_alpha_acc1
            best_vali_alpha_acc2 = vali_alpha_acc2
            best_test_alpha_acc1 = test_alpha_acc1
            best_test_alpha_acc2 = test_alpha_acc2

            # seen, unseen
            best_seen_test_MAPE = test_seen_mape
            best_unseen_test_MAPE = test_unseen_mape
            best_seen_test_alpha_acc1 = test_seen_alpha_acc1
            best_unseen_test_alpha_acc1 = test_unseen_alpha_acc1
            best_seen_test_alpha_acc2 = test_seen_alpha_acc2
            best_unseen_test_alpha_acc2 = test_unseen_alpha_acc2
            
        train_loss = total_loss / len(train_loader)
        total_cl_loss = total_cl_loss / len(train_loader)
        total_lc_loss = total_lc_loss / len(train_loader)
        accelerator.print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {total_cl_loss:.5f}| Train lc loss: {total_lc_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
            f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")
        
        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_RMSE": vali_rmse, "vali_MAPE": vali_mape, "vali_acc1": vali_alpha_acc1, "vali_acc2": vali_alpha_acc2, 
                       "test_RMSE": test_rmse, "test_MAPE": test_mape, "test_acc1": test_alpha_acc1, "test_acc2": test_alpha_acc2})
        
        early_stopping(epoch+1, vali_loss, vali_mae_loss, test_mae_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            accelerator.set_trigger()
            
        if accelerator.check_trigger():
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Test 15%-accuracy: {best_test_alpha_acc1:.4f} | Test 10%-accuracy: {best_test_alpha_acc2:.4f} | Val MAE: {best_vali_MAE:.4f} | Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f} | Val 15%-accuracy: {best_vali_alpha_acc1:.4f} | Val 10%-accuracy: {best_vali_alpha_acc2:.4f} ')
accelerator.print(f'Best model performance: Test Seen MAPE: {best_seen_test_MAPE:.4f} | Test Unseen MAPE: {best_unseen_test_MAPE:.4f}')
accelerator.print(f'Best model performance: Test Seen 15%-accuracy: {best_seen_test_alpha_acc1:.4f} | Test Unseen 15%-accuracy: {best_unseen_test_alpha_acc1:.4f}')
accelerator.print(f'Best model performance: Test Seen 10%-accuracy: {best_seen_test_alpha_acc2:.4f} | Test Unseen 10%-accuracy: {best_unseen_test_alpha_acc2:.4f}')
accelerator.print(path)
accelerator.set_trigger()
if accelerator.check_trigger() and accelerator.is_local_main_process:
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "vali_RMSE": best_vali_RMSE, "vali_MAPE": best_vali_MAPE, "vali_acc1": best_vali_alpha_acc1, "vali_acc2": best_vali_alpha_acc2, 
               "test_RMSE": best_test_RMSE, "test_MAPE":best_test_MAPE, "test_acc1": best_test_alpha_acc1, "test_acc2": best_test_alpha_acc2})
    wandb.finish()
