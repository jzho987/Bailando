# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from dataset.motion_seq import MoSeq

from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import numpy as np
import models
import datetime
from utils.functional import visualizeAndWrite
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import matplotlib.pyplot as plt



class MCTall():
    def __init__(self, args):
        self.config = args
        self.device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        vqvae = self.model.eval()
        gpt = self.model2.train()

        config = self.config
        training_data = self.training_data
        testing_data = self.testing_data
        optimizer = self.optimizer
        writer = SummaryWriter()
        
        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'], strict=False)

        # if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
        #     print('Use pretrained model!')
        #     print(config.init_weight)  
        #     checkpoint = torch.load(config.init_weight)
        #     gpt.load_state_dict(checkpoint['model'], strict=False)
        # # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')


        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            train_epoch_total_loss = 0
            for _, batch in enumerate(tqdm(training_data, desc=f"[ TRAIN ] Epoch: {epoch_i}")):
                pose_seq  = batch 
                pose_seq = pose_seq.to(self.device)
                pose_seq[:, :, :3] = 0
                optimizer.zero_grad()
                
                with torch.no_grad():
                    quants_pred = vqvae.module.encode(pose_seq)
                    if isinstance(quants_pred, tuple):
                        quants_input = tuple(quants_pred[ii][0][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                        quants_target = tuple(quants_pred[ii][0][:, 1:].clone().detach() for ii in range(len(quants_pred)))
                    else:
                        quants = quants_pred[0]
                        quants_input = quants[:, :-1].clone().detach()
                        quants_target = quants[:, 1:].clone().detach()
                
                _, loss = gpt(quants_input, quants_target)
                loss.mean().backward()
                optimizer.step()

                train_epoch_total_loss += loss.clone().mean().detach().cpu().item()

            train_epoch_avg_loss = train_epoch_total_loss / len(training_data)
            writer.add_scalar("train_epoch_avg_loss", train_epoch_avg_loss, epoch_i)


            checkpoint = {
                'model': gpt.state_dict(),
                'config': config,
                'epoch': epoch_i
            }
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
                
            if epoch_i % config.test_freq == 0:
                with torch.no_grad():
                    print("Evaluation...")
                    gpt.eval()
                    test_epoch_total_loss = 0
                    for _, batch in enumerate(tqdm(testing_data, desc=f"[ EVAL  ] Epoch: {epoch_i}")):
                        pose_seq  = batch 
                        pose_seq = pose_seq.to(self.device)
                        pose_seq[:, :, :3] = 0
                        
                        quants_pred = vqvae.module.encode(pose_seq)
                        if isinstance(quants_pred, tuple):
                            quants_input = tuple(quants_pred[ii][0][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                            quants_target = tuple(quants_pred[ii][0][:, 1:].clone().detach() for ii in range(len(quants_pred)))
                        else:
                            quants = quants_pred[0]
                            quants_input = quants[:, :-1].clone().detach()
                            quants_target = quants[:, 1:].clone().detach()
                        
                        _, loss = gpt(quants_input, quants_target)
                        test_epoch_total_loss += loss.clone().mean().detach().cpu().item()

                    test_epoch_avg_loss = test_epoch_total_loss / len(testing_data)
                    writer.add_scalar("test_epoch_avg_loss", test_epoch_avg_loss, epoch_i)
                    writer.flush()
                gpt.train()


    def eval(self):
        with torch.no_grad():
            vqvae = self.model.eval()
            gpt = self.model2.eval()

            config = self.config
            epoch_tested = config.testing.ckpt_epoch

            checkpoint = torch.load(config.vqvae_weight, map_location=self.device)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gpt.load_state_dict(checkpoint['model'])
            gpt.eval()

            results = []
            random_id = 0  # np.random.randint(0, 1e4)
            # quants = {}
            quants_out = {}
            for i_eval, batch_eval in enumerate(tqdm(self.testing_data, desc='Generating Dance Poses')):
                pose_seq = batch_eval.to(self.device)
                quants = vqvae.module.encode(pose_seq)

                if isinstance(quants, tuple):
                    x = tuple(quants[i][0][:, :28].clone() for i in range(len(quants)))
                else:
                    x = quants[0][:, :28].clone()

                zs = gpt.module.sample(x, 120)

                pose_sample = vqvae.module.decode(zs)

                if config.global_vel:
                    print('!!!!!')
                    global_vel = pose_sample[:, :, :3].clone()
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)
                if isinstance(zs, tuple):
                    quants_out[str(i_eval)] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                else:
                    quants_out[str(i_eval)] = zs[0].cpu().data.numpy()[0]

            dance_names = [str(x) for x in range(len(self.testing_data))]
            visualizeAndWrite(results, config, self.evaldir, dance_names, epoch_tested, quants_out)


    def visgt(self,):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            _, pose_seq_eval = batch_eval
            # src_pos_eval = pose_seq_eval[:, :] #
            # global_shift = src_pos_eval[:, :, :3].clone()
            # src_pos_eval[:, :, :3] = 0

            # pose_seq_out, loss, _ = model(src_pos_eval)  # first 20 secs
            # quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()[0]
            # all_quants = np.append(all_quants, quants) if quants is not None else quants
            # pose_seq_out[:, :, :3] = global_shift
            results.append(pose_seq_eval)
            # moduel.module.encode

            # quants = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]

                    # exit()
        # weights = np.histogram(all_quants, bins=1, range=[0, config.structure.l_bins], normed=False, weights=None, density=None)
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
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length), device='cuda')]
                pose_sample = model.module.decode(zs)
                quants['rand_seq_' + str(ii)] = zs[0].cpu().data.numpy()[0]
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
        if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
            print(f'using {config.structure.name} and {config.structure_generate.name} ')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)

            model_class2 = getattr(models, config.structure_generate.name)
            model2 = model_class2(config.structure_generate)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
        self.model2 = model2.to(self.device)
        self.model = model.to(self.device)


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

        self.optimizer = optim(itertools.chain(self.model2.module.parameters(),
                                             ),
                                             **config.kwargs)


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
