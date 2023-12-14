import os
from typing import List, Optional, Union, Tuple
import click
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from torch_utils import gen_utils
import random
import scipy
import numpy as np
import PIL.Image
import torch

import legacy

from swim import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
import gen_utils
# ----------------------------------------------------------------------------

# TODO/hax: Use this for generation: https://huggingface.co/spaces/SIGGRAPH2022/Self-Distilled-StyleGAN/blob/main/model.py
# SGANXL uses it for generation w/L2 norm: https://github.com/autonomousvision/stylegan_xl/blob/4241ff9cfeb69d617427107a75d69e9d1c2d92f2/torch_utils/gen_utils.py#L428
# @click.group()
def main():
    praser = argparse.ArgumentParser()
    praser.add_argument('--network', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
    praser.add_argument('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
    praser.add_argument('--device', help='Device to use for image generation; using the CPU is slower than the GPU', default='cuda')
    # Centroids options
    praser.add_argument('--seed', type=int, help='Random seed to use', default=0)
    praser.add_argument('--num-latents', type=int, help='Number of latents to use for clustering; not recommended to change', default=60000)
    praser.add_argument('--num-clusters', type=int, help='Number of cluster centroids to find', default=8)
    # Extra parameters
    # praser.add_argument('--anchor-latent-space', '-anchor', =True, help='Anchor the latent space to w_avg to stabilize the video')
    praser.add_argument('--plot-pca', '-pca', default=True, help='Plot and save the PCA of the disentangled latent space W')
    praser.add_argument('--dim-pca', '-dim', type=int, help='Number of dimensions to use for the PCA', default=2)
    praser.add_argument('--verbose', type=bool, help='Verbose mode for KMeans (during centroids calculation)', default=False)
    praser.add_argument('--outdir', type=str, help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'clusters'), metavar='DIR')
    praser.add_argument('--description', '-desc', type=str, help='Description name for the directory path to save results', default='pure_centroids')

    args = praser.parse_args()
    get_centroids(network_pkl=args.network, device=args.device, seed=args.seed, num_latents=args.num_latents, num_clusters=args.num_clusters, plot_pca=args.plot_pca, dim_pca=3, 
                  verbose=args.verbose, outdir=args.outdir, description=args.description)

# ----------------------------------------------------------------------------


# @main.command(name='get-centroids')
# @click.pass_context
# @click.option('--network', 'network_pkl', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
# @click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# @click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
# # Centroids options
# @click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
# @click.option('--num-latents', type=int, help='Number of latents to use for clustering; not recommended to change', default=60000, show_default=True)
# @click.option('--num-clusters', type=click.Choice(['32', '64', '128']), help='Number of cluster centroids to find', default='32', show_default=True)
# # Extra parameters
# @click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# @click.option('--plot-pca', '-pca', is_flag=True, help='Plot and save the PCA of the disentangled latent space W')
# @click.option('--dim-pca', '-dim', type=click.IntRange(min=2, max=3), help='Number of dimensions to use for the PCA', default=3, show_default=True)
# @click.option('--verbose', type=bool, help='Verbose mode for KMeans (during centroids calculation)', show_default=True, default=False)
# @click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'clusters'), show_default=True, metavar='DIR')
# @click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='pure_centroids', show_default=True)
def get_centroids(
        # ctx: click.Context,
        network_pkl: str,
        device: Optional[str],
        seed: Optional[int],
        num_latents: Optional[int],
        num_clusters: Optional[str],
        plot_pca: Optional[bool],
        dim_pca: Optional[int],
        verbose: Optional[bool],
        outdir: Union[str, os.PathLike],
        description: Optional[str]=None,
        plot_samples=1000,
):
    """Find the cluster centers in the latent space of the selected model"""
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')
    G = Generator1(train=False, size=256, style_dim=512,channel_multiplier=1)
    state_dict = torch.load(network_pkl, map_location='cpu')['g_ema']
    print(G.load_state_dict(state_dict, strict=False))
    G.to(device=device)
    G.eval()
    # Load the network

    # Setup for using CPU
    # if device.type == 'cpu':
    #     gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    # if anchor_latent_space:
    #     gen_utils.anchor_latent_space(G)

    desc = f'multimodal-truncation-{num_clusters}clusters'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    # Create the run dir with the given name description
    run_dir = gen_utils.make_run_dir(outdir, desc)

    print('Generating all the latents...')
    torch.manual_seed(seed)
    z = torch.randn((num_latents, 512)).to(device)
    with torch.no_grad():
        w = G.style(z)

    # Get the centroids
    print('Finding the cluster centroids. Patience...')
    scaler = StandardScaler()
    scaler.fit(w.cpu())

    # Scale the dlatents and perform KMeans with the selected number of clusters
    w_scaled = scaler.transform(w.cpu())
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, init='random', verbose=int(verbose)).fit(w_scaled)
    cluster_centers_ = scaler.inverse_transform(kmeans.cluster_centers_)
    # Get the centroids and inverse transform them to the original space
    w_avg_multi = torch.Tensor(cluster_centers_).to(device)
    # w_avg_multi.dtype = torch.float32
    print('Success! Saving the centroids...')
    

    if plot_pca:
        print('Plotting the PCA of the disentangled latent space...')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=dim_pca)
        fit_pca = pca.fit(w_scaled)
        fit_pca = pca.fit_transform(w_scaled)
        kmeans_pca = KMeans(n_clusters=int(num_clusters), random_state=0, verbose=0, init='random').fit_predict(fit_pca)
        # colors = ['b', 'c', 'g', 'k', 'm', 'r', 'orange', 'purplr', 'yellow', 'green yellow', 'dark', 'red purple', 'pale teal', 'adobe', 'celery']
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d' if dim_pca == 3 else None)
        index = np.arange(num_latents)
        random.shuffle(index)
        index = np.arange(num_latents)
        labels = kmeans.labels_
        labels = labels[index].tolist()
        
        if dim_pca == 3:
            axes = fit_pca[:, 0], fit_pca[:, 1], fit_pca[:, 2]
             
        else:
            axes = fit_pca[:, 0], fit_pca[:, 1]
        ax.scatter(*axes, c=labels, cmap='tab20c', edgecolor='k', s=40, alpha=0.3)
        
        centroid_pcas = pca.fit_transform(cluster_centers_)
        classes = np.arange(num_clusters)
        if dim_pca == 3:
            axes_c = centroid_pcas[:, 0], centroid_pcas[:, 1], centroid_pcas[:, 2]
        else:
            axes_c = centroid_pcas[:, 0], centroid_pcas[:, 1]
        # axes = centroid_pcas[:, 0], centroid_pcas[:, 1], centroid_pcas[:, 2] if dim_pca == 3 else centroid_pcas[:, 0], centroid_pcas[:, 1]
        ax.scatter(*axes, c=classes, cmap='tab20c', edgecolor='k', s=40, alpha=0.8)
        
        ax.set_title(r"$| \mathcal{W} | \rightarrow $" + f'{dim_pca}')
        ax.axis('off')
        plt.savefig(os.path.join(run_dir, f'pca_{dim_pca}dim_{num_clusters}clusters.png'))
        
        
        fig = plt.figure(figsize=(5, 2.5))
        ax1 = fig.add_subplot(111, projection='3d' if dim_pca == 3 else None)
        
        axes_s = fit_pca[:plot_samples, 0],fit_pca[:plot_samples, 1]
        
        ax1.scatter(*axes_s, c=labels[:plot_samples], cmap='tab20c', edgecolor='k', s=20, alpha=0.4)
        ax1.scatter(*axes_c, c=classes, cmap='tab20c', edgecolor='k', s=20, alpha=0.8)
        ax1.axis('off')
        plt.savefig(os.path.join(run_dir, f'pca_{dim_pca}dim_{num_clusters}clusters1.png'))
        
        # calsses
        # fig = plt.figure(figsize=(20, 10))
        # ax = fig.add_subplot(111, projection='3d' if dim_pca == 3 else None)
        # axes = fit_pca[:, 0], fit_pca[:, 1], fit_pca[:, 2] if dim_pca == 3 else fit_pca[:, 0], fit_pca[:, 1]
        # ax.scatter(*axes, c=kmeans_pca1, cmap='inferno', edgecolor='k', s=40, alpha=0.5)
        # ax.set_title(r"$| \mathcal{W} | \rightarrow $" + f'{dim_pca}')
        # ax.axis('off')
        # plt.savefig(os.path.join(run_dir, f'pca_{dim_pca}dim_{num_clusters}clusters1.png'))
    for idx, w_avg in enumerate(w_avg_multi):
        # w_avg = torch.tile(w_avg, (1, G.num_latents, 1))
        synth_image = G.synthesis(w_avg)
        synth_image = tensor_transform_reverse(synth_image)[0]  # [-1.0, 1.0] -> [0.0, 255.0]
        # synth_image = synth_image.permute(1,2,0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        synth_image = synth_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    # im.save(fp, format=format)
        # Save image and dlatent/new centroid
        PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(run_dir, f'pure_centroid_no{idx+1:03d}-{num_clusters}clusters.jpg'))
        np.save(os.path.join(run_dir, f'centroid_{idx+1:03d}-{num_clusters}clusters.npy'), w_avg.cpu().numpy())
    # synth_image = G.synthesis(w_avg_multi)
    # synth_image = (synth_image + 1) * 255/2  # [-1.0, 1.0] -> [0.0, 255.0]
    # synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()  # NCWH => NWHC
    
    # for idx, img in enumerate(synth_image):
    #     # w_avg = torch.tile(w_avg, (1, G.mapping.num_ws, 1))
    #     # Save image and dlatent/new centroid
    #     PIL.Image.fromarray(img, 'RGB').save(os.path.join(run_dir, f'pure_centroid_no{idx+1:03d}-{num_clusters}clusters.jpg'))
    #     np.save(os.path.join(run_dir, f'centroid_{idx+1:03d}-{num_clusters}clusters.npy'), w_avg_multi[idx].unsqueeze(0).cpu().numpy())
    print('Done!')

def tensor_transform_reverse(image):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * 0.229 + 0.485
    moco_input[:,1,:,:] = image[:,1,:,:] * 0.224 + 0.456
    moco_input[:,2,:,:] = image[:,2,:,:] * 0.225 + 0.406
    return moco_input
# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
