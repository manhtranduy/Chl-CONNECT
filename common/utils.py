import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager
from scipy.stats import linregress
from matplotlib.colors import to_rgba
import pandas as pd
import matplotlib as mpl
import colorsys
import warnings
import os

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# Suppress specific warning by category
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

file_dir=os.path.dirname(os.path.abspath(__file__))
# Specify the path to your font file
font_bold_path = os.path.join(file_dir,'Gulliver Bold Regular.otf')
font_path = os.path.join(file_dir,'Gulliver Regular.otf')
fm.fontManager.addfont(font_path)
fm.fontManager.addfont(font_bold_path)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()
print("Font loaded with internal name:", font_name)
mpl.rcParams['font.family'] = font_name

# Add the font to the matplotlib font manager
fm.fontManager.addfont(font_path)

# Set mathtext to use custom fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = font_name  # regular math text
mpl.rcParams['mathtext.it'] = font_name  # italic math text (override default italic)
mpl.rcParams['mathtext.bf'] = font_name  # bold math text

# =============================================================================
# font configuration
# =============================================================================
# matplotlib.use('Qt5Agg')
font_manager.findfont("Gulliver")
font_manager.findfont("Gulliver")
# Set the custom font for non-math text
mpl.rcParams['font.family'] = 'Gulliver'
mpl.rcParams["legend.labelspacing"] = 0.3
mpl.rcParams["legend.columnspacing"] = 0.3

# Enable LaTeX rendering
# mpl.rcParams['text.usetex'] = True

# Set the LaTeX preamble to use the Gulliver Bold font
# Replace 'Gulliver Bold' with the actual name of the font package for LaTeX
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{Gulliver Bold}'

# mpl.rcParams['mathtext.default'] = 'rm'
# ['rm', 'cal', 'it', 'tt', 'sf', 'bf', 'default', 'bb', 'frak', 'scr', 'regular']


# =============================================================================
# functions
# =============================================================================
def lsqfitgm(X, Y):
    """
    Calculate a "MODEL-2" least squares fit using the geometric mean approach.

    The SLOPE of the line is determined by calculating the GEOMETRIC MEAN
    of the slopes from the regression of Y-on-X and X-on-Y.

    Parameters:
    - X: x data (vector)
    - Y: y data (vector)

    Returns:
    - m: slope
    - b: y-intercept
    - r: correlation coefficient
    - sm: standard deviation of the slope
    - sb: standard deviation of the y-intercept
    """

    # Helper function for Y-on-X regression
    def lsqfity(X, Y):
        n = len(X)
        Sx = np.sum(X)
        Sy = np.sum(Y)
        Sx2 = np.sum(X ** 2)
        Sxy = np.sum(X * Y)

        # Calculate slope for Y-on-X
        my = (n * Sxy - Sx * Sy) / (n * Sx2 - Sx ** 2)
        return my

    # Helper function for X-on-Y regression
    def lsqfitx(X, Y):
        n = len(X)
        Sx = np.sum(X)
        Sy = np.sum(Y)
        Sy2 = np.sum(Y ** 2)
        Sxy = np.sum(X * Y)

        # Calculate slope for X-on-Y
        mx = (n * Sxy - Sx * Sy) / (n * Sy2 - Sy ** 2)
        return mx

    # Determine slope of Y-on-X regression
    my = lsqfity(X, Y)

    # Determine slope of X-on-Y regression
    mx = lsqfitx(X, Y)

    # Calculate geometric mean slope
    m = np.sqrt(my * mx)
    if (my < 0) and (mx < 0):
        m = -m

    # Determine the size of the vector
    n = len(X)

    # Calculate sums and means
    Sx = np.sum(X)
    Sy = np.sum(Y)
    xbar = Sx / n
    ybar = Sy / n

    # Calculate geometric mean intercept
    b = ybar - m * xbar

    # Calculate more sums
    Sx2 = np.sum(X ** 2)

    # Calculate re-used expressions
    den = n * Sx2 - Sx ** 2

    # Calculate r, sm, sb and s2
    r = np.sqrt(my / mx)
    if (my < 0) and (mx < 0):
        r = -r

    diff = Y - b - m * X
    s2 = np.sum(diff * diff) / (n - 2)
    sm = np.sqrt(n * s2 / den)
    sb = np.sqrt(Sx2 * s2 / den)

    return m, b, r, sm, sb


def generate_rainbow_colors(num_colors, brightness=0.7, reverse=False):
    """
    Generate a list of colors with hues ranging from violet to red in rainbow order.

    Parameters:
    - num_colors: Number of colors to generate
    - brightness: Value (brightness) in HSV color space (0-1)
    - reverse: Whether to reverse the color order

    Returns:
    - List of RGB color tuples
    """
    colors = []

    for i in range(num_colors):
        # Map index to hue value from violet (0.8) to red (0)
        # Using a linear mapping
        hue = 0.8 - (0.8 * i / max(1, num_colors - 1))

        saturation = 1.0  # Maximum saturation
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        colors.append((r, g, b))

    if reverse:
        colors.reverse()

    return colors

def list_contains_string(lst):
    for element in lst:
        if isinstance(element, str):
            return True
    return False

def place_text(ax, text, preference, fontsize=14, fontfamily='Gulliver'):
    """
    Place text on the plot according to the preferred position provided.
    
    Parameters:
    - ax: The axis object of the plot.
    - text: The text to place.
    - preference: The preferred position for the text, e.g., 'top left'.
    - fontsize: Font size of the text.
    - fontfamily: Font family of the text.
    """
    positions = {
        'top left': (0.02, 0.98),
        'top right': (0.98, 0.98),
        'bottom left': (0.02, 0.02),
        'bottom right': (0.98, 0.02),
        'center': (0.5, 0.5)
    }
    
    # Adjusted bbox properties for tighter padding and black edge color
    bbox_props = dict(boxstyle="square,pad=0.05", fc="white", ec="black", lw=1, alpha=0.5)
    
    if preference in positions:
        x_pos, y_pos = positions[preference]
        ax.text(x_pos, y_pos, text, transform=ax.transAxes, fontsize=fontsize,
                va='top', horizontalalignment='left', fontfamily=fontfamily,linespacing=1)
                # bbox=bbox_props)


def ErrorNorm(X, Y, stat_area_norm=['R2_log', 'Slope_log', 'SSPB', 'MAPD', 'NV'], up=None):
    if up is None:
        up = np.ones_like(X, dtype=bool)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    X = X[up]
    Y = Y[up, :]

    # Handle invalid data points
    filter_invalid = np.any(np.isnan(Y), axis=1) & ~np.all(np.isnan(Y), axis=1)
    Y_invalid = Y[filter_invalid, :];
    Y_invalid[np.isnan(Y_invalid)] = -1
    Y[filter_invalid, :] = Y_invalid;

    # Initialize metrics
    metrics = {
        'Intercept': [],
        'Intercept_log': [],
        'MAE': [],
        'MAE_log': [],
        'MAPD': [],
        'MAPD_log': [],
        'MR': [],
        'MRAD': [],
        'MRAD_log': [],
        'MR_log': [],
        'N': [],
        'NV': [],
        'R2': [],
        'R2_log': [],
        'RMSD': [],
        'RMSD_log': [],
        'Slope': [],
        'Slope_log': [],
        'SSPB': []  # Added SSPB metric
    }

    for i in range(Y.shape[1]):
        y = Y[:, i]
        valid_idx = ~(np.isnan(X) | np.isnan(y))
        x_filtered, y_filtered = X[valid_idx], y[valid_idx]

        metrics['NV'].append(np.sum(y_filtered < 0))
        y_filtered[y_filtered <= 0] = np.nan

        valid_idx = ~(np.isnan(x_filtered) | np.isnan(y_filtered))
        x_filtered, y_filtered = x_filtered[valid_idx], y_filtered[valid_idx]
        metrics['N'].append(len(x_filtered))

        if metrics['N'][i] > 0:
            # Non-log version
            metrics['RMSD'].append(np.sqrt(np.mean((y_filtered - x_filtered) ** 2)))
            metrics['MAE'].append(np.mean(np.abs(y_filtered - x_filtered)))  # Mean Absolute Error
            metrics['MR'].append(np.median(y_filtered / x_filtered))  # Median Ratio
            metrics['MAPD'].append(
                np.median(np.abs((y_filtered - x_filtered) / x_filtered)) * 100)  # Median Absolute Percentage Deviation
            metrics['MRAD'].append(
                np.mean(np.abs(x_filtered - y_filtered) / x_filtered) * 100)  # Mean Relative Absolute Difference

            # Use geometric mean regression for slope calculation, but linregress for R2
            slope, intercept, _, _, _ = lsqfitgm(x_filtered, y_filtered)
            _, _, r_value, _, _ = linregress(x_filtered, y_filtered)
            metrics['R2'].append(r_value ** 2)
            metrics['Slope'].append(slope)
            metrics['Intercept'].append(intercept)

            # Log version
            metrics['RMSD_log'].append(np.sqrt(np.mean((np.log10(y_filtered) - np.log10(x_filtered)) ** 2)))
            metrics['MAE_log'].append(
                np.mean(np.abs(np.log10(y_filtered) - np.log10(x_filtered))))  # Log Mean Absolute Error

            # Calculate MR_log correctly as median of log10(y/x) - fixed calculation
            log_ratios = np.log10(y_filtered / x_filtered)
            metrics['MR_log'].append(np.median(log_ratios))

            metrics['MAPD_log'].append(np.median(np.abs((np.log10(y_filtered) - np.log10(x_filtered)) / np.log10(
                x_filtered))) * 100)  # Log Median Absolute Percentage Deviation
            metrics['MRAD_log'].append(np.mean(np.abs(np.log10(x_filtered) - np.log10(y_filtered)) / np.log10(
                x_filtered)) * 100)  # Log Mean Relative Absolute Difference

            # Use geometric mean regression for log slope calculation, but linregress for log R2
            log_slope, log_intercept, _, _, _ = lsqfitgm(np.log10(x_filtered), np.log10(y_filtered))
            _, _, log_r_value, _, _ = linregress(np.log10(x_filtered), np.log10(y_filtered))
            metrics['R2_log'].append(log_r_value ** 2)
            metrics['Slope_log'].append(log_slope)
            metrics['Intercept_log'].append(log_intercept)

            # Calculate SSPB correctly based on the fixed MR_log calculation
            mr_log_value = metrics['MR_log'][i]
            sspb_value = 100.0 * np.sign(mr_log_value) * (10.0 ** abs(mr_log_value) - 1.0)
            metrics['SSPB'].append(sspb_value)

    metrics_norm = {}
    for key in metrics:
        if not key == 'N':
            metrics_norm[key + '_n'] = []

    # Normalize metrics
    for key in metrics:
        if key == 'N':
            continue
        if 'R2' in key or 'Slope' in key:
            max_val = np.nanmax([abs(1 - val) for val in metrics[key]])
            for i in range(Y.shape[1]):
                metrics_norm[key + '_n'].append(abs(1 - metrics[key][i]) / max_val)
        else:
            max_val = np.nanmax(metrics[key])
            if max_val == 0:
                metrics_norm[key + '_n'].append(0)
            else:
                metrics_norm[key + '_n'].append(metrics[key][i] / max_val)

    # Area computation
    ErrorIndices_norm = {}
    ErrorIndices = {}
    for stat in stat_area_norm:
        ErrorIndices_norm[stat + '_n'] = metrics_norm[stat + '_n']
        ErrorIndices[stat] = metrics[stat]
    area_stat_n_tmp = np.zeros((Y.shape[1], len(stat_area_norm)))
    area_stat_tmp = np.zeros((Y.shape[1], len(stat_area_norm)))
    for i, stat in enumerate(stat_area_norm):
        for ii in range(Y.shape[1]):
            if ii == Y.shape[1] - 1:
                area_stat_n_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * \
                                         ErrorIndices_norm[stat + '_n'][ii] * ErrorIndices_norm[stat + '_n'][0]
                area_stat_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * ErrorIndices[stat][ii] * \
                                       ErrorIndices[stat][0]
            else:
                area_stat_n_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * \
                                         ErrorIndices_norm[stat + '_n'][ii] * ErrorIndices_norm[stat + '_n'][ii + 1]
                area_stat_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * ErrorIndices[stat][ii] * \
                                       ErrorIndices[stat][ii + 1]
    area_stat_n_tmp = np.round(area_stat_n_tmp, decimals=5)
    area_stat_tmp = np.round(area_stat_tmp, decimals=5)
    area_stat_n = np.sum(area_stat_n_tmp, 1)
    area_stat = np.sum(area_stat_tmp, 1)

    # Convert metrics to DataFrame
    T = pd.DataFrame(metrics)
    for key in metrics:
        if not key == 'N':
            T[key + '_n'] = metrics_norm[key + '_n']

    T['area_stat'] = area_stat
    T['area_stat_n'] = area_stat_n
    return T

def stat_gen(stats,x,y):
    stat_string =[]
    stat_info = ErrorNorm(x, y)
    for stat in stats:
        value=stat_info[stat].values[0]
        if 'MRAD' in stat or 'MAPD' in stat or 'SSPB' in stat:  # Added SSPB to percentage formatting
            stat_string.append(f'{stat} = {value:.2f}% \n')
        elif stat == 'N':
            stat_string.append(f'{stat} = {value} \n')
        else:
            stat_string.append(f'{stat} = {value:.2f} \n')
    stat_string = ''.join(stat_string)
    stat_string = stat_string.replace('R2', 'R$^{\\text{2}}$')
    stat_string = stat_string.replace('_log', '$_{\\text{log}}$')

    return stat_string




def generate_rainbow_colors(num_colors, brightness=0.7, reverse=False):
    """
    Generate a list of colors with hues ranging from violet to red in rainbow order.

    Parameters:
    - num_colors: Number of colors to generate
    - brightness: Value (brightness) in HSV color space (0-1)
    - reverse: Whether to reverse the color order

    Returns:
    - List of RGB color tuples
    """
    colors = []

    for i in range(num_colors):
        # Map index to hue value from violet (0.8) to red (0)
        # Using a linear mapping
        hue = 0.8 - (0.8 * i / max(1, num_colors - 1))

        saturation = 1.0  # Maximum saturation
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        colors.append((r, g, b))

    if reverse:
        colors.reverse()

    return colors


def pscatter_update(x, y, g=None, upoint=None, **opts):
    """
    Create and update an interactive scatter plot with extensive customization options.
    Features lightweight interactive annotations using blitting for fast updates
    during zoom and pan. The fit line's slope is computed using lsqfitgm (which uses lsqfity).

    Parameters:
      x, y       : Data vectors (NumPy arrays) for plotting.
      g          : Grouping vector for data points.
      upoint     : Indicator for unselected data points.
      opts       : Keyword options for customization.

    Returns:
      A dictionary containing handles to scatter plots, the axis, and the figure.
    """
    # ----- Default Options -----
    opts.setdefault('upointshow', True)
    opts.setdefault('classname', None)
    opts.setdefault('classcolor', None)
    opts.setdefault('upointcolor', 'r')
    opts.setdefault('upointedgecolor', 'r')
    opts.setdefault('upointname', 'Unselected')
    opts.setdefault('eqstring', '')
    opts.setdefault('eqfontsize', 16)
    opts.setdefault('stat', ['R2_log', 'Slope_log', 'SSPB', 'MAPD', 'N'])
    opts.setdefault('axis', None)
    opts.setdefault('xscale', 'log')
    opts.setdefault('yscale', 'log')
    opts.setdefault('xlabel', 'x')
    opts.setdefault('ylabel', 'y')
    opts.setdefault('xlim', None)
    opts.setdefault('ylim', None)
    opts.setdefault('title', '')
    opts.setdefault('titlefont', 'Gulliver Bold')
    opts.setdefault('titlefontsize', 18)
    opts.setdefault('titlelocation', 'in')
    opts.setdefault('anfontsize', 15)
    opts.setdefault('anlocation', 'top left')
    opts.setdefault('legend', 'on')
    opts.setdefault('legendfontsize', 15)
    opts.setdefault('legendlocation', 'lower right')
    opts.setdefault('legendborder', True)
    opts.setdefault('legendtitle', None)
    opts.setdefault('legendcolumn', 1)
    opts.setdefault('marker', None)
    opts.setdefault('markersize', 25)
    opts.setdefault('fit', True)
    opts.setdefault('transparency', 0.5)
    opts.setdefault('interactive', True)

    # ----- Validate Input Data -----
    if g is None:
        g = np.zeros_like(x, dtype=np.int8)
    if upoint is None:
        upoint = np.zeros_like(x, dtype=bool)
        opts['upointshow'] = False

    if np.issubdtype(g.dtype, np.number):
        valid_ind = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(g)) & (~np.isnan(upoint)) & (x != 0) & (y != 0)
    elif list_contains_string(g):
        valid_ind = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(upoint)) & (x != 0) & (y != 0) & (
            ~np.array([xi == '' for xi in g]))
    else:
        valid_ind = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)

    x, y, g, upoint = x[valid_ind], y[valid_ind], g[valid_ind], upoint[valid_ind]
    orig_indices = np.where(valid_ind)[0]

    # ----- Setup the Plot Axis -----
    if not opts['axis']:
        fig, ax = plt.subplots(constrained_layout=True)
        opts['axis'] = ax
    else:
        ax = opts['axis']
        fig = plt.gcf()

    ax.set_xscale(opts['xscale'])
    ax.set_yscale(opts['yscale'])
    ax.set_xlim(opts['xlim'])
    ax.set_ylim(opts['ylim'])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.3)

    # Determine reference limits
    all_data = np.concatenate([x[~upoint], y[~upoint]])
    ref1, ref2 = np.median(all_data) / 100, np.median(all_data) * 100
    str_ref1 = "{:e}".format(ref1)
    str_ref2 = "{:e}".format(round(ref2, -len(str(round(ref2)))))
    if opts['xscale'] == 'log' and opts['yscale'] == 'log':
        lim1 = float('1e' + str_ref1.split('e')[1]) / 10
        lim2 = float('1e' + str_ref2.split('e')[1]) * 10
    else:
        lim1 = np.min([np.min(x), np.min(y)])
        lim2 = np.max([np.max(x), np.max(y)])
    if (opts['xlim'] is None) and (opts['ylim'] is None):
        opts['xlim'] = [lim1, lim2]
        opts['ylim'] = [lim1, lim2]

    # Plot reference lines
    ax.plot([lim1 * 1e-3, lim2 * 1e3], [lim1 * 1e-3, lim2 * 1e3], color='black', zorder=1, linewidth=1)
    ax.plot([2 * lim1 * 1e-3, 2 * lim2 * 1e3], [lim1 * 1e-3, lim2 * 1e3], linestyle='--', color='black', zorder=1,
            linewidth=1)
    ax.plot([lim1 * 1e-3, lim2 * 1e3], [2 * lim1 * 1e-3, 2 * lim2 * 1e3], linestyle='--', color='black', zorder=1,
            linewidth=1)
    ax.set_xscale(opts['xscale'])
    ax.set_yscale(opts['yscale'])
    ax.set_xlim(opts['xlim'])
    ax.set_ylim(opts['ylim'])

    # ----- Define Colors, Markers, and Labels for Groups -----
    unique_groups = np.unique(g[~np.isnan(g)])
    if opts['classcolor'] is None:
        opts['classcolor'] = generate_rainbow_colors(len(unique_groups))
    if not opts['marker']:
        opts['marker'] = [None] * len(unique_groups)
    if not opts['classname']:
        opts['classname'] = [f'Class {i}' for i in unique_groups] if len(unique_groups) > 1 else ['Data']

    scatter_plots = []
    all_x_pts, all_y_pts, all_g_pts = [], [], []
    for i, group in enumerate(unique_groups):
        idx = (g == group) & (~upoint)
        point_color = opts['classcolor'][i]
        # Ensure that the color tuple is in [0,1] range:
        point_color = tuple(np.clip(point_color, 0, 1))
        all_x_pts.extend(x[idx])
        all_y_pts.extend(y[idx])
        all_g_pts.extend([i] * np.sum(idx))
        scatter_plots.append(ax.scatter(x[idx], y[idx],
                                        label=opts['classname'][i],
                                        color=to_rgba(point_color, opts['transparency']),
                                        edgecolors=point_color,
                                        s=opts['markersize'],
                                        marker=opts['marker'][i],
                                        zorder=2,
                                        picker=5))
    if opts['upointshow'] and np.any(upoint):
        all_x_pts.extend(x[upoint])
        all_y_pts.extend(y[upoint])
        all_g_pts.extend([len(unique_groups)] * np.sum(upoint))
        scatter_plots.append(ax.scatter(x[upoint], y[upoint],
                                        label=opts['upointname'],
                                        color=to_rgba(opts['upointcolor'], opts['transparency']),
                                        edgecolors=opts['upointcolor'],
                                        s=opts['markersize'],
                                        marker=opts['marker'][0],
                                        zorder=2,
                                        picker=5))
    all_x_pts = np.array(all_x_pts)
    all_y_pts = np.array(all_y_pts)
    all_g_pts = np.array(all_g_pts)

    # ----- Add Statistics Text -----
    stats_text = stat_gen(opts['stat'], x[~upoint], y[~upoint])
    place_text(ax, stats_text, opts['anlocation'], fontsize=opts['anfontsize'])

    # ----- Fit Line (if requested) using lsqfitgm -----
    if opts['fit'] and np.sum(~upoint) > 1:
        if opts['xscale'] == 'log' and opts['yscale'] == 'log':
            log_x = np.log10(x[~upoint])
            log_y = np.log10(y[~upoint])
            slope, intercept, r_value, sm, sb = lsqfitgm(log_x, log_y)
            fit_x = np.linspace(np.min(log_x), np.max(log_x), 100)
            fit_y = intercept + slope * fit_x
            ax.plot(10 ** fit_x, 10 ** fit_y, 'r-', label='Fit Line', zorder=3)
        else:
            slope, intercept, r_value, sm, sb = lsqfitgm(x[~upoint], y[~upoint])
            fit_x = np.linspace(np.min(x[~upoint]), np.max(x[~upoint]), 100)
            fit_y = intercept + slope * fit_x
            ax.plot(fit_x, fit_y, 'r-', label='Fit Line', zorder=3)

    ax.set_xlabel(opts['xlabel'], fontsize=opts['anfontsize'])
    ax.set_ylabel(opts['ylabel'], fontsize=opts['anfontsize'])
    if opts['title']:
        if opts['titlelocation'] == 'in':
            ax.text(0.5, 0.95, opts['title'], fontfamily=opts['titlefont'],
                    fontsize=opts['titlefontsize'], ha='center', transform=ax.transAxes)
        else:
            ax.set_title(opts['title'], fontsize=opts['titlefontsize'], fontfamily=opts['titlefont'])
    if opts['legend'] == 'on':
        bbox_loc = (1, 0.5) if opts['legendlocation'] == 'east outside' else None
        leg_loc = 'center left' if opts['legendlocation'] == 'east outside' else opts['legendlocation']
        ax.legend(loc=leg_loc, bbox_to_anchor=bbox_loc, fontsize=opts['legendfontsize'],
                  title=opts['legendtitle'], ncol=opts['legendcolumn'], frameon=opts['legendborder'])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Segoe UI')
        label.set_fontsize(10)

    fig = plt.gcf()

    # ----- Interactive Annotation using Blitting (with background update on zoom/pan) -----
    if opts['interactive']:
        annotation = ax.annotate('', xy=(0, 0), xytext=(10, 10),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                                 arrowprops=dict(arrowstyle="->"))
        annotation.set_visible(False)
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)

        def update_background(event):
            nonlocal background
            background = fig.canvas.copy_from_bbox(ax.bbox)

        fig.canvas.mpl_connect('draw_event', update_background)

        if cKDTree is not None:
            if opts['xscale'] == 'log' and opts['yscale'] == 'log':
                tree_data = np.column_stack((np.log10(all_x_pts), np.log10(all_y_pts)))
            else:
                tree_data = np.column_stack((all_x_pts, all_y_pts))
            tree = cKDTree(tree_data)
        else:
            tree = None

        last_ind = None

        def hover(event):
            nonlocal last_ind, background
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    fig.canvas.restore_region(background)
                    ax.draw_artist(annotation)
                    fig.canvas.blit(ax.bbox)
                return

            if opts['xscale'] == 'log' and opts['yscale'] == 'log':
                query_point = np.array([np.log10(event.xdata), np.log10(event.ydata)])
            else:
                query_point = np.array([event.xdata, event.ydata])

            if tree is not None:
                dist, ind = tree.query(query_point, k=1, distance_upper_bound=0.05)
            else:
                min_dist = float('inf')
                ind = -1
                for i, (xi, yi) in enumerate(zip(all_x_pts, all_y_pts)):
                    if opts['xscale'] == 'log' and opts['yscale'] == 'log':
                        d = np.sqrt((np.log10(xi) - np.log10(event.xdata)) ** 2 +
                                    (np.log10(yi) - np.log10(event.ydata)) ** 2)
                    else:
                        d = np.hypot(xi - event.xdata, yi - event.ydata)
                    if d < min_dist:
                        min_dist = d
                        ind = i
                dist = min_dist

            if ind == last_ind:
                return
            last_ind = ind

            # --- Simplified annotation: Only show x, y, and group ---
            if ind < len(all_x_pts) and dist < 0.1:
                x_val = all_x_pts[ind]
                y_val = all_y_pts[ind]
                g_val = all_g_pts[ind]
                group_name = opts['classname'][int(g_val)] if g_val < len(unique_groups) else opts['upointname']
                annotation.xy = (x_val, y_val)
                annotation.set_text(
                    f"Group: {group_name}\n{opts['xlabel']}: {x_val:.4g}\n{opts['ylabel']}: {y_val:.4g}")
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)

            fig.canvas.restore_region(background)
            ax.draw_artist(annotation)
            fig.canvas.blit(ax.bbox)

        fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show(block=True)
    return {'scatter_plots': scatter_plots, 'axis': ax, 'figure': fig}

