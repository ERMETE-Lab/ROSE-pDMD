import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np

from .backends import get_mag_fenicsx

class PlotFlowCyl_Fenicsx():
    def __init__(self, domain):
        self.domain = domain

        width = np.max(domain[:,0]) - np.min(domain[:,0])
        height = np.max(domain[:,1]) - np.min(domain[:,1])

        self.aspect = height / width

    def create_circle(self, ls=1):
        # Add a circle centered at (0.5, 0.5) with radius 0.05
        circle = patches.Circle((0.5, 0.5), 0.05, edgecolor='black', facecolor='white', linewidth=ls)
        return circle
    
    def plot_contour(self, ax, snap, cmap = cm.RdYlBu_r, levels=40, show_ticks=False):

        if snap.shape[0] == 2*self.domain.shape[0]:
            snap = get_mag_fenicsx(snap)

        plot = ax.tricontourf(self.domain[:,0], self.domain[:,1], snap, cmap=cmap, levels=levels)
        ax.add_patch(self.create_circle())
        
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return plot

    def plotting_reconstruction(   self, params_to_plot, fom, Re_numbers, fom_times, tt,
                                    recons: list, labels: list, cmap=cm.RdYlBu_r, 
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

            fom_cont = self.plot_contour(axs[0, i], fom[mu_i, tt], cmap=cmap, levels=levels)

            for j, recon in enumerate(recons):
                recon = recon[i]
                self.plot_contour(axs[1 + j, i], recon[:, tt], cmap=cmap, levels=levels)

                if show_residuals:
                    resid = np.abs(get_mag_fenicsx(fom[mu_i, tt]) - get_mag_fenicsx(recon[:, tt])) 
                    self.plot_contour(axs[1 + len(recons) + j, i], resid, cmap=cmap, levels=levels)

        fig.colorbar(fom_cont, ax=axs[:, -1], shrink=shrink).set_label('Velocity')

        [axs[0, i].set_title('Re = {:.2f}'.format(Re_numbers[params_to_plot[i]]), fontsize=fontsize) for i, mu_i in enumerate(params_to_plot)]

        axs[0,0].set_ylabel('FOM', fontsize=fontsize-5)
        for j, recon in enumerate(recons):
            axs[1 + j, 0].set_ylabel(labels[j], fontsize=fontsize-5)
            if show_residuals:
                axs[1 + len(recons) + j, 0].set_ylabel('Residual - ' + labels[j], fontsize=fontsize-5)

        fig.suptitle('Time = {:.2f} s'.format(fom_times[tt]), y=.98, fontsize=fontsize+2)

        return fig, axs
