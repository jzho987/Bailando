# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from dataset.motion_seq import MoSeq
# from models.vqvae import VQVAE

from utils.log import Logger
from utils.functional import str2bool, load_data, load_data_aist, check_data_distribution,visualizeAndWrite,load_test_data_aist,load_test_data
# from utils.metrics import quantized_metrics
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
warnings.filterwarnings('ignore')

# a, b, c, d = check_data_distribution('/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_train')

import matplotlib.pyplot as plt


class MoQ():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self, quick: bool):
        model = self.model.train()
        config = self.config

        training_data = self.training_data
        optimizer = self.optimizer
        
        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
            print('Use pretrained model!')
            print(config.init_weight)  
            checkpoint = torch.load(config.init_weight)
            model.load_state_dict(checkpoint['model'], strict=False)

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        writer = SummaryWriter()
        self.device = torch.device(config.device)


        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            train_epoch_total_loss = 0
            for batch in tqdm(training_data):
                pose_seq = batch.to(self.device)
                if config.rotmat:
                    pose_seq = pose_seq[:, :, 3:]
                elif config.global_vel:
                    pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
                    pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
                    pose_seq = pose_seq.clone().detach()

                else:
                    pose_seq[:, :, :3] = 0
                optimizer.zero_grad()

                _, loss, _ = model(pose_seq)

                train_epoch_total_loss += loss.clone().detach().cpu().item()
                loss.backward()
                optimizer.step()

            
            train_epoch_avg_loss = train_epoch_total_loss / len(training_data)
            writer.add_scalar("train_epoch_avg_loss", train_epoch_avg_loss, epoch_i)

            checkpoint = {
                'model': model.state_dict(),
                'config': config,
                'epoch': epoch_i
            }
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)

            if epoch_i % config.test_freq == 0:
                with torch.no_grad():
                    model.eval()
                    test_epoch_total_loss = 0
                    for i_eval, batch_eval in enumerate(tqdm(self.testing_data)):
                        # Prepare data
                        pose_seq_eval = batch_eval.to(self.device)
                        src_pos_eval = pose_seq_eval[:, :] #
                        global_shift = src_pos_eval[:, :, :3].clone()
                        if config.rotmat:
                            # trans = pose_seq[:, :, :3]
                            src_pos_eval = src_pos_eval[:, :, 3:]
                        elif config.global_vel:
                            src_pos_eval[:, :-1, :3] = src_pos_eval[:, 1:, :3] - src_pos_eval[:, :-1, :3]
                            src_pos_eval[:, -1, :3] = src_pos_eval[:, -2, :3]
                        else:
                            src_pos_eval[:, :, :3] = 0

                        _, loss, _ = model(src_pos_eval)  # first 20 secs
                        test_epoch_total_loss += loss.detach().cpu().item()


                test_epoch_avg_loss = test_epoch_total_loss / len(self.testing_data)
                writer.add_scalar("test_epoch_avg_loss", test_epoch_avg_loss, epoch_i)
                writer.flush()

                model.train()


    def eval(self):
        with torch.no_grad():
            print("Evaluation...")

            config = self.config
            model = self.model.eval()
            epoch_tested = config.testing.ckpt_epoch

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device(config.device)
            checkpoint = torch.load(ckpt_path, map_location=torch.device(self.config.device))
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()

            results = []
            quants = {}

            # the restored error
            tot_euclidean_error = 0
            tot_eval_nums = 0
            tot_body_length = 0
            euclidean_errors = []
            for i_eval, batch_eval in enumerate(tqdm(self.testing_data, desc='Generating Dance Poses')):
                # Prepare data
                pose_seq_eval = batch_eval.to(self.device)
                src_pos_eval = pose_seq_eval[:, :] #
                global_shift = src_pos_eval[:, :, :3].clone()
                if config.rotmat:
                    # trans = pose_seq[:, :, :3]
                    src_pos_eval = src_pos_eval[:, :, 3:]
                elif config.global_vel:
                    print('Using Global Velocity')
                    pose_seq_eval[:, :-1, :3] = pose_seq_eval[:, 1:, :3] - pose_seq_eval[:, :-1, :3]
                    pose_seq_eval[:, -1, :3] = pose_seq_eval[:, -2, :3]
                else:
                    src_pos_eval[:, :, :3] = 0

                b, t, c = src_pos_eval.size()
                pose_seq_out, _, _ = model(src_pos_eval)  
                
                diff = (src_pos_eval - pose_seq_out).view(b, t, c//3, 3)
                tot_euclidean_error += torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=3)))
                tot_eval_nums += 1
                euclidean_errors.append(torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=3))))
                
                body_len = (torch.sum((src_pos_eval[:, :, 0:3] - src_pos_eval[:, :, 9:12]) ** 2, dim=2).sqrt().mean() + \
                torch.sum((src_pos_eval[:, :, 9:12] - src_pos_eval[:, :, 18:21]) ** 2, dim=2).sqrt().mean() + \
                torch.sum((src_pos_eval[:, :, 27:30] - src_pos_eval[:, :, 18:21]) ** 2, dim=2).sqrt().mean() + \
                torch.sum((src_pos_eval[:, :, 36:39] - src_pos_eval[:, :, 27:30]) ** 2, dim=2).sqrt().mean() )

                tot_body_length += body_len
                
                if config.global_vel:
                    print('Using Global Velocity')
                    global_vel = pose_seq_out[:, :, :3].clone()
                    pose_seq_out[:, 0, :3] = 0
                    for iii in range(1, pose_seq_out.size(1)):
                        pose_seq_out[:, iii, :3] = pose_seq_out[:, iii-1, :3] + global_vel[:, iii-1, :]
                else:
                    pose_seq_out[:, :, :3] = global_shift
                

                if config.rotmat:
                    pose_seq_out = torch.cat([global_shift, pose_seq_out], dim=2)
                results.append(pose_seq_out)

                if config.structure.use_bottleneck:
                    quants_pred = model.module.encode(src_pos_eval)
                    if isinstance(quants_pred, tuple):
                        quants[str(i_eval)] = (model.module.encode(src_pos_eval)[0][0].cpu().data.numpy()[0], model.module.encode(src_pos_eval)[1][0].cpu().data.numpy()[0])
                    else:
                        quants[str(i_eval)] = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]
                else:
                    quants = None

            print(tot_euclidean_error / (tot_eval_nums * 1.0))
            print('avg body len', tot_body_length / tot_eval_nums)
            print(torch.mean(torch.stack(euclidean_errors)), torch.std(torch.stack(euclidean_errors)))
            dance_names = [str(x) for x in range(len(self.testing_data))]
            visualizeAndWrite(results, config, self.evaldir, dance_names, epoch_tested, quants)


    def visgt(self,):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.testing_data, desc='Generating Dance Poses')):
            pose_seq_eval = batch_eval
            results.append(pose_seq_eval)

        visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)

    def analyze_code(self,):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        training_data = self.training_data
        all_quants = None

        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.training_data, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            pose_seq_eval = batch_eval.to(self.device)

            quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()
            all_quants = np.append(all_quants, quants.reshape(-1)) if all_quants is not None else quants.reshape(-1)

        print(all_quants)
                    # exit()
        # visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)
        plt.hist(all_quants, bins=config.structure.l_bins, range=[0, config.structure.l_bins])

        #图片的显示及存储
        #plt.show()   #这个是图片显示
        log = datetime.datetime.now().strftime('%Y-%m-%d')
        plt.savefig(self.histdir1 + '/hist_epoch_' + str(epoch_tested)  + '_%s.jpg' % log)   #图片的存储
        plt.close()

    def sample(self,):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        quants = {}

        results = [] 

        if hasattr(config, 'analysis_array') and config.analysis_array is not None:
            # print(config.analysis_array)
            names = [str(ii) for ii in config.analysis_array]
            print(names)
            for ii in config.analysis_array:
                print(ii)
                zs =  [(ii * torch.ones((1, self.config.sample_code_length), device='cuda')).long()]
                print(zs[0].size())
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                if config.global_vel:
                    global_vel = pose_sample[:, :, :3].clone()
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)

        elif hasattr(config, 'analysis_sequence') and config.analysis_sequence is not None:
            # print(config.analysis_array)
            names = ['-'.join([str(jj) for jj in ii]) + '-rate' + str(config.sample_code_rate) for ii in config.analysis_sequence]
            print(names)
            for ii in config.analysis_sequence:
                print(ii)

                zs =  [torch.tensor(np.array(ii).repeat(self.config.sample_code_rate), device='cuda')[None].long()]
                print(zs[0].size())
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants['-'.join([str(jj) for jj in ii]) + '-rate' + str(config.sample_code_rate) ] = (zs[0].cpu().data.numpy()[0], zs[0].cpu().data.numpy()[0])

                if False:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)

        elif hasattr(config, 'analysis_pair') and config.analysis_pair is not None:
            print(config.analysis_pair)
            names = ['-'.join([str(jj) for jj in ii])  for ii in config.analysis_pair]
            print(names)
            for ii in config.analysis_pair:
                print(ii)
                zs =  ([torch.tensor(np.array(ii[:1]).repeat(self.config.sample_code_rate), device='cuda')[None].long()], [torch.tensor(np.array(ii[1:2]).repeat(self.config.sample_code_rate), device='cuda')[None].long()])
                print(zs[0][0].size())
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants['-'.join([str(jj) for jj in ii]) ] = (zs[0][0].cpu().data.numpy()[0], zs[1][0].cpu().data.numpy()[0])

                if False:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length), device='cuda')]
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants[str(ii)] = zs[0].cpu().data.numpy()[0]
                quants['rand_seq_' + str(ii)] = (zs[0].cpu().data.numpy()[0], zs[0].cpu().data.numpy()[0])

                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]
                        
                results.append(pose_sample)
        visualizeAndWrite(results, config, self.sampledir, names, epoch_tested, quants)


    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()

        if hasattr(config.data, 'train_dir'):
            self._build_train_loader()
            self._build_optimizer()

        if hasattr(config.data, 'test_dir'):      
            self._build_test_loader()



    def _build_model(self):
        """ Define Model """
        config = self.config 
        if hasattr(config.structure, 'name'):
            print(f'using {config.structure.name}')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        self.model = model.to(config.device)


    def _build_train_loader(self):
        data = self.config.data
        print("building training set")
        fnames = os.listdir(data.train_dir)
        train_dance_data = []
        for name in tqdm(fnames):
            path = os.path.join(data.train_dir, name)
            np_dance = np.load(path)
            train_dance_data.append(np_dance)
        print(f"data loaded: {len(train_dance_data)}")

        self.training_data = prepare_dataloader(train_dance_data, self.config.batch_size)



    def _build_test_loader(self):
        data = self.config.data
        print("building testing set")
        fnames = os.listdir(data.test_dir)
        test_dance_data = []
        for name in tqdm(fnames):
            path = os.path.join(data.test_dir, name)
            np_dance = np.load(path)[:896]
            test_dance_data.append(np_dance)
        print(f"data loaded: {len(test_dance_data)}")

        self.testing_data = prepare_dataloader(test_dance_data, self.config.batch_size)

    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model.module.parameters(),
                                             ),
                                             **config.kwargs)
        # self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("./", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

        self.histdir = os.path.join(self.visdir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir):
            os.mkdir(self.histdir)

        self.imgsdir = os.path.join(self.visdir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir):
            os.mkdir(self.imgsdir)

        self.videodir = os.path.join(self.visdir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir):
            os.mkdir(self.videodir)
        
        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        self.gtdir = os.path.join(self.expdir, "gt")
        if not os.path.exists(self.gtdir):
            os.mkdir(self.gtdir)

        self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.histdir1 = os.path.join(self.evaldir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir1):
            os.mkdir(self.histdir1)

        self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir1):
            os.mkdir(self.imgsdir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir1):
            os.mkdir(self.videodir1)

        self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)


def prepare_dataloader(dance_data, batch_size):
    modata = MoSeq(dance_data)
    sampler = torch.utils.data.RandomSampler(modata, replacement=True)
    data_loader = torch.utils.data.DataLoader(
        modata,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True
    )

    return data_loader

