import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import io
from ipywidgets import interact, IntSlider
from monai.visualize.utils import blend_images
import torch
pd.options.mode.chained_assignment = None  # default='warn'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


def plot_sigmoid(num_channels, current_epoch, all_logits, all_preds, all_targets, s_i, z_i):
    fig, axs = plt.subplots(num_channels, 4)
    fig.suptitle(f"Epoch: {current_epoch}, sample: {s_i}, z index: {z_i}")

    for i in range(num_channels):
        img_ref = axs[i, 1].imshow(all_logits[s_i][i, :, :, z_i], vmin=0, vmax=1)
        axs[i, 1].set_title(f'sigmoid_{i}')
        axs[i, 2].imshow(all_preds[s_i][i, :, :, z_i])
        axs[i, 2].set_title(f"pred_{i}")
        axs[i, 3].imshow(all_targets[s_i][i, :, :, z_i])
        axs[i, 3].set_title(f"gt_{i}")

    # Turn off the axes for all subplots
    for ax in axs.ravel():
        ax.axis('off')

    # Add a colorbar to the entire figure
    cbar = fig.colorbar(img_ref, ax=axs[:, 0], location='left')
    plt.tight_layout()        

    fig.savefig(f"sigmoid_imgs/3ch/{current_epoch}.png")
    return fig2img(fig)
    # plt.close(fig)
    
def plot_confusion(metrics, epoch, threshold = 0.5):

    df = pd.DataFrame({
                    "prediction": metrics.case_pred.values(),
                    "target": metrics.case_target.values()
                    })

    df["fn"] = (df["prediction"] < threshold) & (df["target"]  == 1)
    df["fp"] = (df["prediction"] > threshold) & (df["target"]  == 0)
    df["tp"] = (df["prediction"] > threshold) & (df["target"]  == 1)
    df["tn"] = (df["prediction"] < threshold) & (df["target"]  == 0)


    cf_matrix = np.array([[df.tn.sum(), df.fp.sum()], [df.fn.sum(), df.tp.sum()]])

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    g = sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues').set_title(f"Confusion matrix @ epoch {epoch}")

    img = fig2img(g.figure)
    plt.close(g.figure)
    return img

def plot_metrics(metrics, epoch):

    # aggregate metrics
    AP = round(metrics.AP, 4)
    auroc = round(metrics.auroc, 4)
    picai_score = metrics.score

    # Precision-Recall (PR) curve
    precision = metrics.precision
    recall = metrics.recall

    # Receiver Operating Characteristic (ROC) curve
    tpr = metrics.case_TPR
    fpr = metrics.case_FPR

    # Free-Response Receiver Operating Characteristic (FROC) curve
    sensitivity = metrics.lesion_TPR
    fp_per_case = metrics.lesion_FPR
    
    score = round(metrics.score, 4)

    fig, axs = plt.subplots(1,3,constrained_layout=True, subplot_kw=dict(box_aspect=1))# subplot_kw={'adjustable' : 'box', 'aspect' : 'equal'})
    fig.suptitle(f"Evaluation metrics @ epoch {epoch}\nScore:{score}")
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=axs[0])

    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name="None")
    disp.plot(ax=axs[1])
    axs[1].get_legend().remove()

    disp = RocCurveDisplay(fpr=fp_per_case, tpr=sensitivity, estimator_name="None")
    disp.plot(ax=axs[2])
    axs[2].set_xlim(0.001, 5.0); axs[2].set_xscale('log')
    axs[2].set_xlabel("False positives per case"); axs[2].set_ylabel("Sensitivity")
    # axs[2].set_aspect('equal', adjustable='box')
    axs[2].get_legend().remove()

    [ax.set_title(title) for ax, title in zip(axs, [f"AP: {AP}", f"AROC: {auroc}", "FROC"])]
    
    img = fig2img(fig)
    plt.close(fig)
    return img
    
def plot_difference(metrics, epoch, threshold=0.5):

    df = pd.DataFrame({
                        "prediction": metrics.case_pred.values(),
                        "target": metrics.case_target.values()
                        })

    df["fn"] = (df["prediction"] < threshold) & (df["target"]  == 1)
    df["fp"] = (df["prediction"] > threshold) & (df["target"]  == 0)
    df["tp"] = (df["prediction"] > threshold) & (df["target"]  == 1)
    df["tn"] = (df["prediction"] < threshold) & (df["target"]  == 0)

    df["difference"] = 0.0
    df.loc[df.fp | df.fn, "difference"] = threshold - df.prediction
    
    
    fig, ax = plt.subplots(1,2, constrained_layout=True, subplot_kw=dict(box_aspect=1))

    df.difference.plot(kind='bar', color="red", ax=ax[0]).set_title("FP & FN")
    ax[0].axhline(0, color="black")
    ax[0].set_ylim(bottom=-0.5, top=0.5)
    ax[0].locator_params(axis='x', nbins=10)
    ax[0].locator_params(axis='y', nbins=5)
    
    df["difference"] = 0.0
    df.loc[df.tp | df.tn, "difference"] = df.prediction - threshold
    
    df.difference.plot(kind='bar', color="green", ax=ax[1]).set_title("TP & TN")
    ax[1].axhline(0, color="black")
    ax[1].set_ylim(bottom=-0.5, top=0.5)
    ax[1].locator_params(axis='x', nbins=10)
    ax[1].locator_params(axis='y', nbins=5)
    fig.suptitle(f"Difference from threshold: {threshold} @ epoch {epoch} ")
    
    img = fig2img(fig)
    plt.close(fig)
    return img



def volume_slice_plotter(volume1, volume2):
    
    volume1 = volume1.detach().cpu().numpy() if "Tensor" in str(type(volume1)) else volume1
    volume2 = volume2.detach().cpu().numpy() if "Tensor" in str(type(volume2)) else volume2
    
    # Calculate min and max values for volume1
    vmin = np.min(volume1)
    vmax = np.max(volume1)

    # Visualization function
    @interact(slice_index=IntSlider(min=0, max=volume1.shape[-1]-1, description='Slice Index:'))
    def plot_slices(slice_index):
        slice1 = volume1[:, :, slice_index]
        slice2 = volume2[:, :, slice_index]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
        
        # Plot with a consistent colormap
        im1 = axes[0].imshow(slice1, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Volume 1, Slice {slice_index}')
        
        im2 = axes[1].imshow(slice2, cmap='viridis', interpolation='nearest') 
        axes[1].set_title(f'Volume 2, Slice {slice_index}')

        # Add a shared colorbar
        fig.colorbar(im1, ax=axes.ravel().tolist())  # Shared colorbar for both subplots

        plt.show()


def slice_comparison(image, labels, titles):
    blended_imgs = [blend_images(image=image, label=label, cmap="coolwarm") for label in labels]

    # Visualization function
    @interact(slice_index=IntSlider(min=0, max=labels[0].shape[-1]-1, description='Slice Index:'))
    def plot_slices(slice_index):
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, len(labels) + 1, width_ratios=[1] * len(labels) + [0.05])  # Add space for colorbar

        for i, (blended_img, title) in enumerate(zip(blended_imgs, titles)):
            ax = fig.add_subplot(gs[0, i])
            img_plot = ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap="coolwarm")
            ax.set_title(title, fontsize=20)
            ax.axis('off')

        cax = fig.add_subplot(gs[0, -1])  # Colorbar axes
        cbar = fig.colorbar(img_plot, cax=cax, aspect=5)  # Get the Colorbar object
        # cbar.ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

import matplotlib.ticker as ticker

def slice_comparison_multi(image, labels, titles):

    vmin = -1
    vmax = 1
    alpha = 0.5

    # Standard XAI heatmap: blue (low) -> green -> yellow -> red (high)
    cmap = LinearSegmentedColormap.from_list(
            #"RdWhGn", ["red", "white","white", "green"]
            #"xai_heatmap", ["blue", "cyan", "green", "yellow", "red"]
            "xai_heatmap", ["blue", "red"]
        )
    cmap = "inferno"

    blended_imgs_1 = [image[0:1].repeat(3, 1, 1, 1)]
    blended_imgs_2 = [image[1:2].repeat(3, 1, 1, 1)]
    blended_imgs_3 = [image[2:3].repeat(3, 1, 1, 1)]

    blended_imgs_1 += [blend_images(image=image[0:1], label=label[0:1], cmap=cmap, rescale_arrays=False, alpha=alpha) if label.shape[0] == 3 else blend_images(image=image[0:1], label=label[0:1], rescale_arrays=False) for label in labels ]
    blended_imgs_2 += [blend_images(image=image[1:2], label=label[1:2], cmap=cmap, rescale_arrays=False, alpha=alpha) if label.shape[0] == 3 else blend_images(image=image[1:2], label=label[0:1], rescale_arrays=False) for label in labels]
    blended_imgs_3 += [blend_images(image=image[2:3], label=label[2:3], cmap=cmap, rescale_arrays=False, alpha=alpha) if label.shape[0] == 3 else blend_images(image=image[2:3], label=label[0:1], cmap="coolwarm", rescale_arrays=False) for label in labels]


    

    # Visualization function
    @interact(slice_index=IntSlider(min=0, max=labels[0].shape[-1]-1, description='Slice Index:'))
    def plot_slices(slice_index):
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, len(titles), width_ratios=[1] * len(titles), wspace=0.01, hspace=-0.45)  # Add space for colorbar 0.01
        axs = []

        for i, (blended_img, title) in enumerate(zip(blended_imgs_1, titles)):
            ax = fig.add_subplot(gs[0, i])
            axs.append(ax)
            img_plot = ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap=cmap, vmin=vmin, vmax=vmax)
            
            ax.set_title(title, fontsize=10)
            # ax.axis('off')
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            
            if i == 0:
                ax.set_ylabel("T2W")
            
        for i, (blended_img, title) in enumerate(zip(blended_imgs_2, titles)):
            ax = fig.add_subplot(gs[1, i])
            axs.append(ax)
            img_plot = ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if i == 0:
                ax.set_ylabel("ADC")
            
            
        for i, (blended_img, title) in enumerate(zip(blended_imgs_3, titles)):
            ax = fig.add_subplot(gs[2, i])
            axs.append(ax)
            img_plot = ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if i == 0:
                ax.set_ylabel("HBV")
        
        # Share colorbar between subplots
        cbar = plt.colorbar(img_plot, ax=axs)
        # cbar.ax.set_ylim(vmin, vmax)
        # cbar.set_label('Colorbar')

        # Set colorbar maximum and minimum values for all subplots
        # cbar.set_clim(0, 1)
        
            

        # cax = fig.add_subplot(gs[0, -1])  # Colorbar axes
        # cbar = fig.colorbar(img_plot, cax=cax, aspect=5)  # Get the Colorbar object
        # cbar.ax.set_ylim(0, 1)

        # plt.tight_layout()
        plt.show()



def slice_comparison_multi_gif(image, labels, titles, save_path="slice_comparison.gif", duration=150, dpi=100):
    """
    Create a pendulum-style looping GIF from the multi-slice comparison plot.

    Renders each slice as a frame, then reverses the sequence (excluding
    endpoints) to create a smooth back-and-forth animation.

    Args:
        image: Input image tensor (C, H, W, D) with 3 channels (T2W, ADC, HBV)
        labels: List of attribution/label tensors to overlay
        titles: List of column titles (including raw image column)
        save_path: Output path for the GIF file
        duration: Frame duration in milliseconds
        dpi: Resolution of each frame
    """
    vmin = -1
    vmax = 1
    alpha = 0.3
    cmap = "jet"

    blended_imgs_1 = [image[0:1].repeat(3, 1, 1, 1)]
    blended_imgs_2 = [image[1:2].repeat(3, 1, 1, 1)]
    blended_imgs_3 = [image[2:3].repeat(3, 1, 1, 1)]

    blended_imgs_1 += [blend_images(image=image[0:1], label=label[0:1], cmap=cmap, rescale_arrays=False, alpha=alpha) if label.shape[0] == 3 else blend_images(image=image[0:1], label=label[0:1], rescale_arrays=False) for label in labels]
    blended_imgs_2 += [blend_images(image=image[1:2], label=label[1:2], cmap=cmap, rescale_arrays=False, alpha=alpha) if label.shape[0] == 3 else blend_images(image=image[1:2], label=label[0:1], rescale_arrays=False) for label in labels]
    blended_imgs_3 += [blend_images(image=image[2:3], label=label[2:3], cmap=cmap, rescale_arrays=False, alpha=alpha) if label.shape[0] == 3 else blend_images(image=image[2:3], label=label[0:1], cmap="coolwarm", rescale_arrays=False) for label in labels]

    n_slices = labels[0].shape[-1]
    frames = []

    for slice_index in range(n_slices):
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, len(titles), width_ratios=[1] * len(titles), wspace=0.01, hspace=-0.45)
        axs = []

        for i, (blended_img, title) in enumerate(zip(blended_imgs_1, titles)):
            ax = fig.add_subplot(gs[0, i])
            axs.append(ax)
            ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if i == 0:
                ax.set_ylabel("T2W")

        for i, (blended_img, title) in enumerate(zip(blended_imgs_2, titles)):
            ax = fig.add_subplot(gs[1, i])
            axs.append(ax)
            ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if i == 0:
                ax.set_ylabel("ADC")

        for i, (blended_img, title) in enumerate(zip(blended_imgs_3, titles)):
            ax = fig.add_subplot(gs[2, i])
            axs.append(ax)
            img_plot = ax.imshow(torch.moveaxis(blended_img[:, :, :, slice_index], 0, -1), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if i == 0:
                ax.set_ylabel("HBV")

        plt.colorbar(img_plot, ax=axs)
        fig.suptitle(f"Slice {slice_index}/{n_slices - 1}", fontsize=12)

        # Render figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # Create pendulum sequence: forward + reverse (excluding endpoints)
    pendulum_frames = frames + frames[-2:0:-1]

    pendulum_frames[0].save(
        save_path,
        save_all=True,
        append_images=pendulum_frames[1:],
        duration=duration,
        loop=0
    )

    print(f"Saved pendulum GIF ({len(pendulum_frames)} frames) to {save_path}")
    return save_path


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
