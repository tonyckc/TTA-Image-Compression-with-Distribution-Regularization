import torch
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


def correlation_compute(data,means,scales,mask_type='point',window_size=7):
    '''
    data: feature map
    mask type

    '''



    data = torch.div((data-means),scales+1e-8)
    z = data.unfold(2, window_size, 1).unfold(3, window_size,
                                                1)  # extracting the respective window with a stride of 1
    batch, channel, row, col = data.shape
    i, j = z.shape[2], z.shape[3]
    z = z.reshape(batch, channel, i, j, -1, window_size * window_size)
    # print(z.shape)
    # torch.Size([1, 3, 506, 762, 1, 49])
    mid = int((window_size * window_size) / 2)
    # print(mid)
    tensor2 = z[:, :, :, :, :,
              mid]  # extracting the middle value from each respective window with which correlation would be calculated
    # print (tensor2)
    tensor2 = tensor2.reshape(batch, channel, i, j, 1, 1)
    tensor2 = z * tensor2
    # print(tensor2.size())

    tensor2 = tensor2.reshape(batch * channel * i * j, window_size * window_size)
    tensor2 = torch.mean(tensor2, dim=0)
    mid_point = tensor2[mid]
    tensor2 = tensor2 / mid_point
    if mask_type == 'point':
        tensor2[mid] = 0
    correlation_map = tensor2
    correlation_loss = (correlation_map**2).sum()
    return correlation_map,correlation_loss



def distribution_correlation_compute(data,means,scales,mask_type='point',window_size=7):
    '''
    data: feature map
    mask type

    '''
    #torch.exp(-0.5 * (mu_1 - mu_2) * ((sigma_1 ** 2 + sigma_2 ** 2 + (1 / gamma))) * (mu_1 - mu_2))


    #data = torch.div((data-means),scales+1e-8)
    z_mean = means.unfold(2, window_size, 1).unfold(3, window_size,
                                                1)  # extracting the respective window with a stride of 1
    z_scales = scales.unfold(2, window_size, 1).unfold(3, window_size,
                                                    1)  # extracting the respective window with a stride of 1


    batch, channel, row, col = data.shape
    i, j = z_mean.shape[2], z_mean.shape[3]
    z_mean = z_mean.reshape(batch, channel, i, j, -1, window_size * window_size)
    z_scales = z_scales.reshape(batch, channel, i, j, -1, window_size * window_size)


    # print(z.shape)
    # torch.Size([1, 3, 506, 762, 1, 49])
    mid = int((window_size * window_size) / 2)
    # print(mid)
    tensor2_mean = z_mean[:, :, :, :, :,
              mid]  # extracting the middle value from each respective window with which correlation would be calculated
    tensor2_scales = z_scales[:, :, :, :, :,
              mid]  # extracting the middle valu
    # print (tensor2)

    tensor2_mean = tensor2_mean.reshape(batch, channel, i, j, 1, 1)
    tensor2_mean = tensor2_mean.repeat(1, 1, 1, 1, 1, window_size * window_size)

    tensor2_scales = tensor2_scales.reshape(batch, channel, i, j, 1, 1)
    tensor2_scales = tensor2_scales.repeat(1, 1, 1, 1, 1, window_size * window_size)
    gamma = torch.tensor(0.01).cuda()
    tensor2 = torch.exp(-0.5 * (z_mean - tensor2_mean) * ((torch.exp(z_scales) + torch.exp(tensor2_scales) + (1 / gamma))) * (z_mean - tensor2_mean))

    # print(tensor2.size())

    tensor2 = tensor2.reshape(batch * channel * i * j, window_size * window_size)
    tensor2 = torch.mean(tensor2, dim=0)
    mid_point = tensor2[mid]
    tensor2 = tensor2 / mid_point
    if mask_type == 'point':
        tensor2[mid] = 0
    correlation_map = tensor2
    #print(correlation_map)
    #exit()
    correlation_loss = ((correlation_map)**2).sum()
    return correlation_map,correlation_loss




def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    #ax.set_xticklabels(col_labels)
    #ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts