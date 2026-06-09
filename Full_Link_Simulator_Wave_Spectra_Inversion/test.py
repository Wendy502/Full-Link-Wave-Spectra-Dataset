"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import cv2
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from scipy.io import savemat
from set_seed import set_seed
from tqdm import tqdm
import time

start_time = time.time()

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    set_seed()
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='Test', ncols=80):
        model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

        # 结果反归一化（real_B, fake_B）及保存mat文件作后续分析
        mat_path = "".join(img_path)
        mat_name = mat_path.split('\\')[-1]
        mat_folder = opt.mat_output_folder + '/' + opt.name
        mat_SavePath = mat_folder + '/' + mat_name

        if not os.path.exists(mat_folder):
            os.makedirs(mat_folder)

        real_B = visuals['real_B'].cpu().numpy()
        real_B = np.squeeze(real_B, axis=0)
        real_B = np.power(10, (real_B*0.5+0.5) * 4) - 1
        # real_B = cv2.normalize(real_B,None, 0, 1, cv2.NORM_MINMAX)
        real_B = np.squeeze(real_B, axis=0)

        fake_B = visuals['fake_B'].cpu().numpy()
        fake_B = np.squeeze(fake_B, axis=0)
        fake_B = np.power(10, (fake_B*0.5+0.5) * 4) - 1
        # fake_B = cv2.normalize(fake_B, None, 0, 1, cv2.NORM_MINMAX)
        fake_B = np.squeeze(fake_B, axis=0)

        real_A = visuals['real_A'].cpu().numpy()
        real_A = np.squeeze(real_A, axis=0)
        real_A = real_A[0,:,:]
        # real_A = np.power(10, (real_A*0.5+0.5) * 3) - 1
        real_A = real_A * 0.5 + 0.5

        savemat(mat_SavePath, {'real_A': real_A,'real_B': real_B, 'fake_B': fake_B})

    webpage.save()  # save the HTML


end_time = time.time()
# 计算耗时
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.4f} 秒")