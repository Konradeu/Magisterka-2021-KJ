"""Script used to load state dict, to a created discriminator object"""

# all the necessary imports for this script
import gc
from os import listdir

import torch
import hypergan as hg
from hypergan.configuration import Configuration
from hypergan.gan_component import GANComponent
import matplotlib.pyplot as plt
from hypergan.inputs.image_loader import ImageLoader
from hyperchamber.config import Config
from torchvision import datasets, transforms

import numpy as np
import pickle


if __name__ == '__main__':  # this if statement needs to be applied in windows OS

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    paths = ['Normal', 'Full_pictures', 'Cutted_pictures']
    batches = ['4 batches','8 batches', '12 batches', '16 batches']
    gC_it = 0
    for b in batches:
        for p in paths:
            path = r'D:\Cyfronet\unpacked'
            path += '\\' + b + '\\' + p

            for dir in listdir(path):
                final_path = path + '\\' + dir + '\\saves\\default\\discriminator0.save'
                image_path = r'D:\cyfronet images\Seperate' + '\\' + b + '\\' + p + '\\' + dir
                state_dict = torch.load(final_path)  # the "r" needs to be before the path in windows OS
                config = Configuration.load('default.json')
                # print("The config keys:", config.keys())
                inp_class = 'class:hypergan.inputs.image_loader.ImageLoader'  # the class for the lookup function
                klass = GANComponent.lookup_function(None, inp_class)
                config['blank'] = False  # taken from the HyperGAN library code fragment
                # config['directory'] = r'C:\Users\jakub\Desktop\Magisterka'  # path to the folder containing saves and samples DEEEWWWW IIIITTTTT
                batch_in_int = int(b.split()[0])
                config['batch_size'] = batch_in_int  # the batch size of trained model - necessary for the discriminator input

                #  config for the ImageLoader class (the input of the gan -> discriminator)
                if p == 'Normal':
                    img_loader_config = Config({'class': 'class:hypergan.inputs.image_loader.ImageLoader', 'batch_size': batch_in_int,
                                      'directories': [image_path], 'channels': 3, 'crop': False,
                                      'height': 150, 'random_crop': False, 'resize': True, 'shuffle': True, 'width': 250,
                                      'blank': False})
                elif p == 'Full_pictures':
                    img_loader_config = Config(
                        {'class': 'class:hypergan.inputs.image_loader.ImageLoader', 'batch_size': batch_in_int,
                         'directories': [image_path], 'channels': 3,
                         'crop': False,
                         'height': 256, 'random_crop': False, 'resize': True, 'shuffle': True, 'width': 256,
                         'blank': False})

                elif p == 'Cutted_pictures':
                    img_loader_config = Config(
                        {'class': 'class:hypergan.inputs.image_loader.ImageLoader', 'batch_size': batch_in_int,
                         'directories': [image_path], 'channels': 3,
                         'crop': False,
                         'height': 128, 'random_crop': False, 'resize': True, 'shuffle': True, 'width': 128,
                         'blank': False})

                img_loader = ImageLoader(img_loader_config)
                gan = hg.GAN(inputs=img_loader, config=config)  # creating the GAN object, from which we can take the discriminator
                discriminator = gan.discriminator

                missing_keys, unexpected_keys = discriminator.load_state_dict(state_dict)
                # the commented parts of code are previously used workaround to get both scores for real ang generated
                # images, without shuffling

                # print(missing_keys, unexpected_keys)  # checking if there are some missing or wrong wages
                # print(discriminator.state_dict().keys())

                transform = transforms.Compose([transforms.ToTensor()])  # image transforms for the PyTorch methods

                test_imgs_path = image_path  # relative path to the test images, not to the class dir
                dataset = datasets.ImageFolder(test_imgs_path, transform=transform)  # dateset object used for dataloader

                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_in_int, shuffle=True, num_workers=0)  # object for loading images
                # dL_iter = iter(dataloader)

                labels_map = {0: "Gen", 1: "Real"}
                j = 0
                discriminator_score = []
                dL_placeholder = []
                # for images, labels in dataloader:
                #     dL_placeholder.append([images, labels])

                for images, labels in dataloader:
                    nump_labels = labels.numpy()
                    images, labels = images.cuda(), labels.cuda()

                    i = 0
                    plt.figure()
                    disc_results = discriminator(images).detach().cpu().numpy()

                    for img in images:
                        discriminator_score.append({'label': nump_labels[i], 'score': disc_results[i]})
                        print(discriminator_score)
                        ax = plt.subplot(1, 4, i + 1)
                        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
                        ax.set_title("Ground.truth: {}".format(labels_map[nump_labels[i]]), fontsize=8)
                        ax.text(10.0, 190.0, 'Disc:{}'.format(disc_results[i]), fontsize=8)
                        plt.axis('off')

                        i += 1
                        if i == 4:
                            plt.show()
                            i=0
                    j += 1

                    print(j)
                    if j == 50:  # number, how many figures should the script create
                        file_name = b + '_' + p + '_' + '_' + dir + '_' + 'dict.pkl'
                        a_file = open(file_name, "wb")
                        pickle.dump(discriminator_score, a_file)
                        print(a_file, '--- Han been created!')
                        a_file.close()
                        print(a_file, '--- Han been closed!')
                        del klass
                        print('The: "klass" object has been deleted!')
                        del config
                        print('The: "config" object has been deleted!')
                        del gan
                        print('The: "gan" object has been deleted!')
                        del img_loader_config
                        print('The: "img_loader_config" object has been deleted!')
                        del img_loader
                        print('The: "img_loader" object has been deleted!')
                        del discriminator
                        print('The: "discriminator" object has been deleted!')
                        del dataloader
                        print('The: "dataloader" object has been deleted!')
                        del transform
                        print('The: "transform" object has been deleted!')
                        torch.cuda.empty_cache()
                        if gC_it == 1:
                            print('Garbage collection has started... ')
                            gc.collect()
                            print("Garbage collection has just finished!")
                            gC_it = 0
                        else:
                            gC_it += 1
                        break

'''In case if you need to run a HyperGAN comment'''
# import os
# os.system(r'hypergan train C:\Users\jakub\Desktop\Magisterka' + ' --resize --size 250x150x3 -b 4  --sampler batch_walk --steps 100 --sample_every 5')