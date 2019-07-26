import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _save_fig(fig, file_name, pil_copy=None, colormap=None):
    if file_name != '':

        # Path => generate directories
        path = os.path.abspath(os.path.dirname(file_name))
        if not os.path.exists(path):
            os.makedirs(path)
        # Save
        fig.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0)

        if pil_copy is not None:
            try:
                if pil_copy.dtype != np.uint8:
                    scaler = MinMaxScaler()
                    pil_copy = scaler.fit_transform(pil_copy)
                    pil_copy = (pil_copy*255).astype(np.uint8)

                im = pil_copy
                if colormap is not None:
                    cm = mpl.cm.get_cmap(colormap)
                    im = cm(pil_copy)
                    im = np.uint8(im * 255)

                img = Image.fromarray(im)
                file_name_base = os.path.splitext(file_name)[0]
                img.save(file_name_base + '_pil_copy.png')

            except MemoryError as er:
                print('visualizer.py : _save_fig -> Unable to save PIL image copy..')
                print(er)


def display_image(image, file_name='', single_channel_cmap='jet', normalize_axes=False,
                  on_tensorboard_with_name=None, tensorboard_writer=None, hide_axes=False,
                    render_on_screen=True, min_max_rescale=False,
                  save_pil_copy=True):

    # Check if the input images are in RGB
    is_rgb = image.ndim == 3 and image.shape[2] == 3

    # Generate figure
    dpi = 150
    fig = plt.figure(figsize=(10,10), dpi=dpi, frameon=False)

    # Rescale?
    if min_max_rescale:
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max()
        image = image * 255
        image = image.astype(np.uint8)

    # Display figure
    cmap = None if is_rgb else single_channel_cmap
    plt.imshow(image, cmap=cmap)

    # Normalize axes?
    if normalize_axes:
        plt.axis('normal')

    # Disable axes
    if hide_axes:
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)

    # Render on screen?
    if render_on_screen:
        fig.show()

    # Save?
    _save_fig(fig, file_name, pil_copy=image if save_pil_copy else None, colormap=cmap)

    # Close figure
    plt.close(fig)


def _plot_with_range(x,y,plot_range,legend='',color='red',marker='o',markersize=5,linewidth=2,grid=True,logscale=False):

    # Plot range
    if plot_range is None:
        plot_range = range(len(y[0]))

    # Panda dataframe
    data = {'x': x[plot_range]}
    data.update({l: y[ii][plot_range] for ii, l in enumerate(legend)})

    df = pd.DataFrame(data)

    # Plot!
    # multiple line plot
    plt_fun = plt.plot
    if logscale:
        plt_fun = plt.semilogx
    for ii, column in enumerate(df.drop('x', axis=1)):
        plt_fun(df['x'], df[column], marker=marker[ii], color=color(ii), markersize=markersize,
                 linewidth=linewidth, alpha=0.9, label=column)

    plt.grid(grid)


def plot_cmc(cmc, color='blue', marker='o', linestyle='solid', legend='CMC', legend_location='best', linewidth=4, markersize=8, file_name='',
             grid=True, render_on_screen=True, title='Cumulative Matching Characteristic (CMC)', main_range=None, font_size=24,
             inset_range=None, inset_location=None, inset_font_size=22):
    dpi = 150
    fig = plt.figure(figsize=(10, 10), dpi=dpi, frameon=False)

    # style
    plt.style.use('default')
    color = plt.cm.get_cmap('Set1')

    if not (isinstance(cmc, list) or isinstance(cmc, tuple)):
        cmc = [cmc]
        marker = [marker]
        linestyle = [linestyle]
        legend = [legend]

    # Figure
    ax = plt.subplot(111, xlabel='Rank', ylabel='Recognition Accuracy [%]', title=title)

    if main_range is None:
        main_range = slice(0,len(cmc[0]))
    _plot_with_range(x=list(range(len(cmc[0]))), y=cmc, legend=legend, color=color, marker=marker, markersize=markersize,
                     linewidth=linewidth, plot_range=main_range, grid=grid)

    # Plot legend
    if legend != '':
        plt.legend(loc=legend_location, fontsize=font_size)

    # Inset figure?
    if inset_range is not None and inset_location is not None:
        # this is an inset axes over the main axes
        inset_ax = plt.axes(inset_location)
        _plot_with_range(x=list(range(len(cmc[0]))), y=cmc, legend=legend, color=color, marker=marker, markersize=markersize,
                         linewidth=linewidth, plot_range=inset_range, grid=grid)

        for item in ([inset_ax .title, inset_ax .xaxis.label, inset_ax .yaxis.label] + inset_ax .get_xticklabels() + inset_ax .get_yticklabels()):
            item.set_fontsize(inset_font_size)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

    # Need to render on screen
    if render_on_screen:
        plt.show()

    # Save?
    _save_fig(fig, file_name)

    # Close figure
    plt.close(fig)


def display_ranked_matching_images(matching_images, im_size=None, matching_ids=None, indexes=None,
                                   true_match_line_width=6, true_match_line_color='#00FF00',
                                   file_name='', render_on_screen=True):
    if indexes is None:
        indexes = range(0, len(matching_images))
    N = len(matching_images[0][1])
    if im_size is None:
        im_size = (matching_images[0][0].width, matching_images[0][0].height)

    fontspec = {'family': 'serif', 'color': 'black', 'fontweight': 'bold', 'fontsize': 12}

    # Init figure
    dpi = 150
    fig = plt.figure(figsize=(10, 10), dpi=dpi, frameon=False)

    # Create image grid
    gs = gridspec.GridSpec(len(indexes), 1, wspace=0.0, hspace=0.0, figure=fig)
    ax = [plt.subplot(gs[i]) for i in range(len(indexes))]

    # Draw probe/gallery text
    ax[0].set_title('Probe           Ranked Gallery', fontdict=fontspec, loc='left')

    # Plot each selected image
    for ii, idx in enumerate(indexes):

        # Transparent canvas image (N + 1 probe + 1 empty space)
        canvas_im = Image.new('RGBA', (im_size[0] * (N+2), im_size[1]), (0,0,0,0))

        # Probe image
        probe_image = matching_images[idx][0].resize(im_size)

        # True match location
        tm = []
        if matching_ids is not None:
            tm = np.where(np.array(matching_ids[idx][1]) == matching_ids[idx][0])[0].tolist()

        # Gallery images strip
        gallery_images = Image.new('RGB', (im_size[0]*N, im_size[1]))
        x_offset = 0
        for jj, im in enumerate(matching_images[idx][1]):
            im = im.resize(im_size)
            if jj in tm:
                _draw_rectangle(im, im_size, true_match_line_color, true_match_line_width)

            gallery_images.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        # Past the probe ad the gallery strip onto the canvas, then draw it
        canvas_im.paste(probe_image, (0,0))
        canvas_im.paste(gallery_images, (im_size[0]*2,0))
        ax[ii].imshow(canvas_im)
        ax[ii].axis('off')

    # Render on screen
    if render_on_screen:
        plt.show()

    # Remove (hide) axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)

    # Save?
    _save_fig(fig, file_name)

    # Close figure
    plt.close(fig)


def _draw_rectangle(im, im_size, color, width=1):
    draw = ImageDraw.Draw(im)
    for i in range(width):
        draw.rectangle([0+i, 0+i, im_size[0]-i, im_size[1]-i], outline=color)