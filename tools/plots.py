import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import os
import matplotlib.gridspec as gridspec

from .backends import get_mag_fenicsx, get_mag

def make_mp4(path, video_path, fps=30, key = lambda x: int(x.split('_')[1].split('.')[0])):
    # Sort image files numerically based on the extracted time value
    image_files = [img for img in os.listdir(path) if img.endswith('.png')]
    image_files = sorted(image_files, key = key)

    # Create the MP4 video by loading each image and adding it to the array
    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            image = imageio.imread(img_path)
            writer.append_data(image)

def make_gif(path, gif_path, fps=30, key=lambda x: int(x.split('_')[1].split('.')[0])):
    # Sort image files numerically based on the extracted time value
    image_files = [img for img in os.listdir(path) if img.endswith('.png')]
    image_files = sorted(image_files, key=key)

    # Create the GIF by loading each image and appending it to the gif writer
    images = []
    for img_file in image_files:
        img_path = os.path.join(path, img_file)
        image = imageio.imread(img_path)
        images.append(image)

    # Save the images as a GIF
    imageio.mimsave(gif_path, images, fps=fps)

class PlotDYNASTY():
    def __init__(self, domain):
        self.domain = domain

        # Define DYNASTY coordinates
        data = {
            "CV_code": self.domain[:, 0],  # Example CV codes
            "Component": [
                *["GV1"]*2, *["Cooler"]*(26-5), *["GV2"]*(61-26), *["GO1"]*(92-61), *["GV1"]*(125-92)
            ]
        }
        df = pd.DataFrame(data)

        # Loop dimensions
        loop_dimension = 3.05
        half_dim = loop_dimension / 2

        # Segment lengths based on the number of control volumes
        num_gv1 = sum(df['Component'] == 'GV1')
        num_cooler = sum(df['Component'] == 'Cooler')
        num_gv2 = sum(df['Component'] == 'GV2')
        num_go1 = sum(df['Component'] == 'GO1')

        # Calculate x, y coordinates for each segment
        coords = []

        # GV1 (left leg) goes from bottom (-half_dim, -half_dim) to top (-half_dim, +half_dim)
        y_gv1 = np.linspace(-half_dim, half_dim, num_gv1)
        x_gv1 = np.full(num_gv1, -half_dim)
        coords.extend(zip(x_gv1, y_gv1))

        # Cooler (top leg) goes from left (-half_dim, +half_dim) to right (+half_dim, +half_dim)
        x_cooler = np.linspace(-half_dim, half_dim, num_cooler)
        y_cooler = np.full(num_cooler, half_dim)
        coords.extend(zip(x_cooler, y_cooler))

        # GV2 (right leg) goes from top (+half_dim, +half_dim) to bottom (+half_dim, -half_dim)
        y_gv2 = np.linspace(half_dim, -half_dim, num_gv2)
        x_gv2 = np.full(num_gv2, half_dim)
        coords.extend(zip(x_gv2, y_gv2))

        # GO1 (bottom leg) goes from right (+half_dim, -half_dim) to left (-half_dim, -half_dim)
        x_go1 = np.linspace(half_dim, -half_dim, num_go1)
        y_go1 = np.full(num_go1, -half_dim)
        coords.extend(zip(x_go1, y_go1))

        # Rearrange the coordinates to start from the second-to-last point of GV1
        coords = coords[(125-92):] + coords[:(125-92)]  # Move last two points of GV1 to the start

        # Assign coordinates to DataFrame
        df['x'], df['y'] = zip(*coords)

        self.coords = np.asarray(df['x'].to_numpy()), np.asarray(df['y'].to_numpy())

    def plot_contour(self, ax, snap, time = None, cmap = cm.jet, levels=40, show_ticks=False):
        if time is None:
            time = np.arange(0, snap.shape[1], 1)
        
        assert snap.shape[0] == self.domain.shape[0], 'Snapshots and domain do not have the same number of nodes'
        assert snap.shape[1] == len(time), 'Snapshots and time do not have the same number of time steps'

        plot = ax.contourf(time, self.domain[:, 0], snap, cmap=cmap, levels=levels)

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        return plot
    
    def  add_tubes(ax, _s, facecolors = ['none']*2):
        outer_rect = patches.Rectangle((-3.05/2 - _s, -3.05/2 - _s), 3.05 + 2 * _s, 3.05 + 2 * _s, linewidth=1, edgecolor='k', facecolor=facecolors[0])
        inner_rect = patches.Rectangle((-3.05/2 + _s, -3.05/2 + _s), 3.05 - 2 * _s, 3.05 - 2 * _s, linewidth=1, edgecolor='k', facecolor=facecolors[1])
        
        ax.add_patch(outer_rect)
        ax.add_patch(inner_rect)

    def plot_loop(self, ax, snap, cmap=cm.jet, s=100,
                  _s_coeff = None, show_ticks=False, vmin=None, vmax=None):

        # Create and add the rectangles
        if _s_coeff is not None:
            _s = s / _s_coeff
            self.add_tubes(ax, _s)
        
        # Plot the loop
        sc = ax.scatter(*self.coords, c=snap, cmap=cmap, s=s, marker='s', vmin=vmin, vmax=vmax)

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        return sc 
    
    def plotting_loop_rec(  self, params_to_plot, fom, params, fom_times, tt,
                            recons: list, labels: list, cmap=cm.RdYlBu_r, 
                            length_plot = 6, levels=None, fontsize=20, 
                            shrink=0.9, fraction=0.046, pad=0.04,
                            show_residuals = True, show_cb = True
                                ):

        if show_residuals:
            nrows = 1 + len(recons) * 2
        else:
            nrows = 1 + len(recons)
        ncols = len(params_to_plot)

        fig, axs = plt.subplots(nrows, ncols, figsize=(length_plot * ncols, nrows * (length_plot-0.5)))
        axs = axs.reshape(nrows, ncols)

        if levels is None:
            levels = [np.min(fom), np.max(fom)]

        for i, mu_i in enumerate(params_to_plot):

            fom_loop = self.plot_loop(axs[0, i], fom[mu_i, tt], cmap=cmap, vmin=levels[0], vmax=levels[1])

            for j, recon in enumerate(recons):
                recon = recon[i]
                self.plot_loop(axs[1 + j, i], recon[:, tt], cmap=cmap, vmin=levels[0], vmax=levels[1])

                if show_residuals:
                    resid = np.abs(fom[mu_i, tt] - recon[:, tt])
                    resid_loop = self.plot_loop(axs[1 + len(recons) + j, i], resid, cmap=cmap, vmin=resid.min(), vmax=resid.max())

            if show_residuals and show_cb:
                fig.colorbar(resid_loop, ax=axs[len(recons)+1:, i], shrink=shrink, fraction=fraction, pad = pad).set_label('Residual')
        if show_cb:
            fig.colorbar(fom_loop, ax=axs[:len(recons)+1, -1], shrink=shrink, fraction=fraction, pad = pad).set_label('Temperature')
        [axs[0, i].set_title('P = {:.2f}'.format(params[params_to_plot[i]]), fontsize=fontsize) for i, mu_i in enumerate(params_to_plot)]

        axs[0,0].set_ylabel('FOM', fontsize=fontsize-5)
        for j, recon in enumerate(recons):
            axs[1 + j, 0].set_ylabel(labels[j], fontsize=fontsize-5)
            if show_residuals:
                axs[1 + len(recons) + j, 0].set_ylabel('Residual - ' + labels[j], fontsize=fontsize-5)

        fig.suptitle('Time = {:.2f} s'.format(fom_times[tt]), y=.98, fontsize=fontsize+2)

        for ax in axs.flatten():
            ax.set_aspect('equal')

        return fig, axs


class PlotFlowCyl():
    def __init__(self, domain, centre = (0.5, 0.5), radius = 0.05, is_fenicsx = True):
        self.domain = domain

        width = np.max(domain[:,0]) - np.min(domain[:,0])
        height = np.max(domain[:,1]) - np.min(domain[:,1])

        self.aspect = height / width

        self.centre = centre
        self.radius = radius
        self.is_fenicsx = is_fenicsx

    def create_circle(self, ls=1):
        
        circle = patches.Circle(self.centre, self.radius, edgecolor='black', facecolor='white', linewidth=ls)
        return circle
    
    def plot_contour(self, ax, snap, cmap = cm.RdYlBu_r, levels=40, show_ticks=False):

        if snap.shape[0] == 2*self.domain.shape[0]:

            if self.is_fenicsx:
                snap = get_mag_fenicsx(snap)
            else:
                snap = get_mag(snap)

        plot = ax.tricontourf(self.domain[:,0], self.domain[:,1], snap, cmap=cmap, levels=levels)
        ax.add_patch(self.create_circle())
        
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return plot

    # def plotting_reconstruction(   self, params_to_plot, fom, Re_numbers, fom_times, tt,
    #                                 recons: list, labels: list, cmap=cm.RdYlBu_r, 
    #                                 length_plot = 6, levels=100, fontsize=20, shrink=0.9,
    #                                 show_residuals = True
    #                             ):

    #     if show_residuals:
    #         nrows = 1 + len(recons) * 2
    #     else:
    #         nrows = 1 + len(recons)
    #     ncols = len(params_to_plot)

    #     fig, axs = plt.subplots(nrows, ncols, figsize=(length_plot * ncols, nrows * (length_plot-0.5) * self.aspect))

    #     for i, mu_i in enumerate(params_to_plot):

    #         fom_cont = self.plot_contour(axs[0, i], fom[mu_i, tt], cmap=cmap, levels=levels)

    #         for j, recon in enumerate(recons):
    #             recon = recon[i]
    #             self.plot_contour(axs[1 + j, i], recon[:, tt], cmap=cmap, levels=levels)

    #             if show_residuals:
    #                 if self.is_fenicsx:
    #                     resid = np.abs(get_mag_fenicsx(fom[mu_i, tt]) - get_mag_fenicsx(recon[:, tt])) 
    #                 else:
    #                     resid = np.abs(get_mag(fom[mu_i, tt]) - get_mag(recon[:, tt]))

    #                 self.plot_contour(axs[1 + len(recons) + j, i], resid, cmap=cmap, levels=levels)

    #     fig.colorbar(fom_cont, ax=axs[:, -1], shrink=shrink).set_label('Velocity')

    #     [axs[0, i].set_title('Re = {:.2f}'.format(Re_numbers[params_to_plot[i]]), fontsize=fontsize) for i, mu_i in enumerate(params_to_plot)]

    #     axs[0,0].set_ylabel('FOM', fontsize=fontsize-5)
    #     for j, recon in enumerate(recons):
    #         axs[1 + j, 0].set_ylabel(labels[j], fontsize=fontsize-5)
    #         if show_residuals:
    #             axs[1 + len(recons) + j, 0].set_ylabel('Residual - ' + labels[j], fontsize=fontsize-5)

    #     fig.suptitle('Time = {:.2f} s'.format(fom_times[tt]), y=.98, fontsize=fontsize+2)

    #     return fig, axs

    def plotting_reconstruction(self, params_to_plot, fom, Re_numbers, fom_times, tt,
                                recons: list, labels: list, cmap=cm.RdYlBu_r, 
                                length_plot=6, levels=100, fontsize=20, show_residuals=True):

        if show_residuals:
            nrows = 1 + len(recons) * 2
        else:
            nrows = 1 + len(recons)
        ncols = len(params_to_plot) + 1  # Extra column for the colorbar

        fig = plt.figure(figsize=(length_plot * (ncols - 1), nrows * (length_plot - 0.5) * self.aspect))
        gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1] * (ncols - 1) + [0.05])  # Last column for colorbar

        axs = np.empty((nrows, ncols - 1), dtype=object)

        for i, mu_i in enumerate(params_to_plot):
            axs[0, i] = fig.add_subplot(gs[0, i])
            fom_cont = self.plot_contour(axs[0, i], fom[mu_i, tt], cmap=cmap, levels=levels)

            for j, recon in enumerate(recons):
                axs[1 + j, i] = fig.add_subplot(gs[1 + j, i])
                self.plot_contour(axs[1 + j, i], recon[i][:, tt], cmap=cmap, levels=levels)

                if show_residuals:
                    axs[1 + len(recons) + j, i] = fig.add_subplot(gs[1 + len(recons) + j, i])
                    if self.is_fenicsx:
                        resid = np.abs(get_mag_fenicsx(fom[mu_i, tt]) - get_mag_fenicsx(recon[i][:, tt])) 
                    else:
                        resid = np.abs(get_mag(fom[mu_i, tt]) - get_mag(recon[i][:, tt]))

                    self.plot_contour(axs[1 + len(recons) + j, i], resid, cmap=cmap, levels=levels)

        # Create a dedicated colorbar axis
        cax = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(fom_cont, cax=cax)
        cb.set_label('Velocity', fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize-4)
        
        # Titles and labels
        [axs[0, i].set_title(f'Re = {Re_numbers[params_to_plot[i]]:.2f}', fontsize=fontsize) for i, mu_i in enumerate(params_to_plot)]
        axs[0, 0].set_ylabel('FOM', fontsize=fontsize - 5)
        for j, recon in enumerate(recons):
            axs[1 + j, 0].set_ylabel(labels[j], fontsize=fontsize - 5)
            if show_residuals:
                axs[1 + len(recons) + j, 0].set_ylabel(f'Residual - {labels[j]}', fontsize=fontsize - 5)

        fig.suptitle(f'Time = {fom_times[tt]:.2f} s', y=0.98, fontsize=fontsize + 2)

        return fig, axs



class PlotTWIGL():
    def __init__(self, domain):
        self.domain = domain

        width = np.max(domain[:,0]) - np.min(domain[:,0])
        height = np.max(domain[:,1]) - np.min(domain[:,1])

        self.aspect = height / width

        self.Nh = domain.shape[0]

    def plot_contour(self, ax, snap, cmap = cm.plasma, levels=40, show_ticks=False):

        plot = ax.tricontourf(self.domain[:,0], self.domain[:,1], snap, cmap=cmap, levels=levels)
        
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return plot

    def plotting_reconstruction(   self, params_to_plot, fom, omegas, fom_times, tt,
                                    recons: list, labels: list, cmap=cm.RdYlBu_r, 
                                    field_to_plot = 0,
                                    length_plot = 6, levels=100, fontsize=20, shrink=0.9,
                                    show_residuals = True
                                ):

        if show_residuals:
            nrows = 1 + len(recons) * 2
        else:
            nrows = 1 + len(recons)
        ncols = len(params_to_plot)

        fig, axs = plt.subplots(nrows, ncols, figsize=(length_plot * ncols, nrows * (length_plot-0.5) * self.aspect))

        for i, mu_i in enumerate(params_to_plot):

            fom_cont = self.plot_contour(axs[0, i], fom[mu_i, tt, field_to_plot * self.Nh : (field_to_plot+1) * self.Nh], cmap=cmap, levels=levels)

            for j, recon in enumerate(recons):
                recon = recon[i]
                self.plot_contour(axs[1 + j, i], recon[field_to_plot * self.Nh : (field_to_plot+1) * self.Nh, tt], cmap=cmap, levels=levels)

                if show_residuals:
                    resid = np.abs(fom[mu_i, tt, field_to_plot * self.Nh : (field_to_plot+1) * self.Nh] - recon[field_to_plot * self.Nh : (field_to_plot+1) * self.Nh, tt])
                    self.plot_contour(axs[1 + len(recons) + j, i], resid, cmap=cmap, levels=levels)

        fig.colorbar(fom_cont, ax=axs[:, -1], shrink=shrink).set_label('Velocity')

        [axs[0, i].set_title('$\omega = {:.2f}$'.format(omegas[mu_i]), fontsize=fontsize) for i, mu_i in enumerate(params_to_plot)]

        axs[0,0].set_ylabel('FOM', fontsize=fontsize-5)
        for j, recon in enumerate(recons):
            axs[1 + j, 0].set_ylabel(labels[j], fontsize=fontsize-5)
            if show_residuals:
                axs[1 + len(recons) + j, 0].set_ylabel('Residual - ' + labels[j], fontsize=fontsize-5)

        fig.suptitle('Time = {:.2f} s'.format(fom_times[tt]), y=.98, fontsize=fontsize+2)

        return fig, axs
