import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.colors import to_rgba
import pandas as pd
from matplotlib import font_manager
import matplotlib as mpl
import colorsys
import warnings

# Suppress specific warning by category
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# font configuration
# =============================================================================
font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
font_manager.findfont("gulliver")
font_manager.findfont("gulliver bold")
# Set the custom font for non-math text
mpl.rcParams['font.family'] = 'gulliver'
mpl.rcParams["legend.labelspacing"] = 0.3
mpl.rcParams["legend.columnspacing"] = 0.3

# Enable LaTeX rendering
# mpl.rcParams['text.usetex'] = True

# Set the LaTeX preamble to use the Gulliver font
# Replace 'gulliver' with the actual name of the font package for LaTeX
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{gulliver}'

# mpl.rcParams['mathtext.default'] = 'rm'
# ['rm', 'cal', 'it', 'tt', 'sf', 'bf', 'default', 'bb', 'frak', 'scr', 'regular']


# =============================================================================
# functions
# =============================================================================


def generate_hue_colors(num_colors,brightness=0.7,reverse=False):
    colors = []
    for i in range(num_colors):
        hue = (1 - i/ num_colors) * 0.7  # 0.7 represents the range from violet to red in HSV
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

def place_text(ax, text, preference, fontsize=14, fontfamily='gulliver'):
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
def ErrorNorm(X, Y, stat_area_norm=['R2_log', 'Slope_log', 'RMSD_log', 'NV', 'MAE_log', 'MAPD_log'], up=None):
    if up is None:
        up = np.ones_like(X, dtype=bool)
    if Y.ndim==1:
        Y=Y.reshape(-1, 1)
    
    X = X[up]
    Y = Y[up, :]
    
        
    
    # Handle invalid data points
    filter_invalid = np.any(np.isnan(Y), axis=1) & ~np.all(np.isnan(Y), axis=1)
    Y_invalid = Y[filter_invalid,:];
    Y_invalid[np.isnan(Y_invalid)] = -1
    Y[filter_invalid,:]=Y_invalid;
    
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
        'Slope_log': []
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
            metrics['MAPD'].append(np.median(np.abs((y_filtered - x_filtered) / x_filtered)) * 100)  # Median Absolute Percentage Deviation
            metrics['MRAD'].append(np.mean(np.abs(x_filtered - y_filtered) / x_filtered) * 100)  # Mean Relative Absolute Difference
            slope, intercept, r_value, _, _ = linregress(x_filtered, y_filtered)
            metrics['R2'].append(r_value ** 2)
            metrics['Slope'].append(slope)
            metrics['Intercept'].append(intercept)
            
            # Log version
            metrics['RMSD_log'].append(np.sqrt(np.mean((np.log10(y_filtered) - np.log10(x_filtered)) ** 2)))
            metrics['MAE_log'].append(np.mean(np.abs(np.log10(y_filtered) - np.log10(x_filtered))))  # Log Mean Absolute Error
            metrics['MR_log'].append(np.median(np.log10(y_filtered) / np.log10(x_filtered)))  # Log Median Ratio
            metrics['MAPD_log'].append(np.median(np.abs((np.log10(y_filtered) - np.log10(x_filtered)) / np.log10(x_filtered))) * 100)  # Log Median Absolute Percentage Deviation
            metrics['MRAD_log'].append(np.mean(np.abs(np.log10(x_filtered) - np.log10(y_filtered)) / np.log10(x_filtered)) * 100)  # Log Mean Relative Absolute Difference
            slope_log, intercept_log, r_value_log, _, _ = linregress(np.log10(x_filtered), np.log10(y_filtered))
            metrics['R2_log'].append(r_value_log ** 2)
            metrics['Slope_log'].append(slope_log)
            metrics['Intercept_log'].append(intercept_log)


    metrics_norm={}
    for key in metrics:
        if not key == 'N':
            metrics_norm[key+'_n']=[]
    
    # Normalize metrics
    for key in metrics:
        if key == 'N':
            continue
        if 'R2' in key or 'Slope' in key:
            max_val = np.nanmax([abs(1 - val) for val in metrics[key]])
            for i in range(Y.shape[1]):
                metrics_norm[key+'_n'].append(abs(1-metrics[key][i]) / max_val)
        else:
            max_val = np.nanmax(metrics[key])
            if max_val == 0:
                metrics_norm[key+'_n'].append(0)
            else:
                metrics_norm[key+'_n'].append(metrics[key][i] / max_val)
                
    # Area computation
    ErrorIndices_norm={}
    ErrorIndices={}
    for stat in stat_area_norm:
        ErrorIndices_norm[stat+'_n'] = metrics_norm[stat+'_n']
        ErrorIndices[stat] = metrics[stat]
    area_stat_n_tmp = np.zeros((Y.shape[1],len(stat_area_norm)))
    area_stat_tmp = np.zeros((Y.shape[1],len(stat_area_norm)))
    for i,stat in enumerate(stat_area_norm):
        for ii in range(Y.shape[1]):
            if ii == Y.shape[1]-1:
                area_stat_n_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * ErrorIndices_norm[stat+'_n'][ii] * ErrorIndices_norm[stat+'_n'][0]
                area_stat_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * ErrorIndices[stat][ii] * ErrorIndices[stat][0]
            else:
                area_stat_n_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * ErrorIndices_norm[stat+'_n'][ii] * ErrorIndices_norm[stat+'_n'][ii+1]
                area_stat_tmp[ii][i] = 0.5 * np.sin(np.radians(360 / len(stat_area_norm))) * ErrorIndices[stat][ii] * ErrorIndices[stat][ii+1]
    area_stat_n_tmp=np.round(area_stat_n_tmp,decimals=5)
    area_stat_tmp=np.round(area_stat_tmp,decimals=5)
    area_stat_n=np.sum(area_stat_n_tmp,1)
    area_stat=np.sum(area_stat_tmp,1)
    
    # Convert metrics to DataFrame
    T = pd.DataFrame(metrics)
    for key in metrics:
        if not key == 'N':
            T[key+'_n']=metrics_norm[key+'_n']
            
    T['area_stat']=area_stat
    T['area_stat_n']=area_stat_n
    return T

def stat_gen(stats,x,y):
    stat_string =[]
    stat_info = ErrorNorm(x, y)
    for stat in stats:
        value=stat_info[stat].values[0]
        if 'MRAD' in stat or 'MAPD' in stat:
            stat_string.append(f'{stat} = {value:.2f}% \n')
        elif stat == 'N':
            stat_string.append(f'{stat} = {value} \n')
        else:
            stat_string.append(f'{stat} = {value:.2f} \n')
    stat_string = ''.join(stat_string)
    stat_string = stat_string.replace('R2','R$^{2}$')
    stat_string = stat_string.replace('_log','$_{log}$')
    
    return stat_string



def pscatter_update(x, y, g=None, upoint=None, **opts):
    """
    Create and update a scatter plot with extensive customization options.
    
    Parameters:
    - x, y: Data vectors for plotting.
    - g: Grouping vector for data points.
    - upoint: Unselected data points indicator.
    - opts: Various customization options.
    
    Returns:
    - A dictionary containing handles to plot elements.
    """
    # Default options handling
    opts.setdefault('upointshow', True)
    opts.setdefault('classname', None)
    opts.setdefault('classcolor', None)  # Default class color
    opts.setdefault('upointcolor', 'r')
    opts.setdefault('upointedgecolor', 'b')
    opts.setdefault('upointname', 'Unselected')
    opts.setdefault('eqstring', '')
    opts.setdefault('eqfontsize', 16)
    opts.setdefault('stat', ['R2_log', 'Slope_log', 'RMSD_log', 'MRAD', 'MAPD_log', 'N'])
    opts.setdefault('axis', None)
    opts.setdefault('xscale', 'log')
    opts.setdefault('yscale', 'log')
    opts.setdefault('xlabel', 'x')
    opts.setdefault('ylabel', 'y')
    opts.setdefault('xlim', None)
    opts.setdefault('ylim', None)
    opts.setdefault('title', '')
    opts.setdefault('titlefont', 'gulliver bold')
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
    opts.setdefault('markeredgecolor', opts['classcolor'])
    
    if g is None:
        g = np.zeros_like(x,dtype=np.int8)
    if upoint is None:
        upoint = np.zeros_like(x, dtype=bool)
        opts['upointshow']=False
    
    
    if np.issubdtype(g.dtype, np.number):
        valid_ind=(~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(g)) & (~np.isnan(upoint)) & ~(x==0) & ~(y==0)
    elif list_contains_string(g):
        valid_ind=~np.isnan(x) & ~np.isnan(y) & (~np.isnan(upoint)) & ~(x==0) & ~(y==0) & ~np.array([x=='' for x in g])
        
    x=x[valid_ind]
    y=y[valid_ind]
    g=g[valid_ind]
    upoint=upoint[valid_ind]
    
        
    
    if not opts['axis']:
        fig, ax = plt.subplots(constrained_layout=True)
        opts['axis'] = ax
        # fig.tight_layout()
    opts['axis'].set_xscale(opts['xscale'])
    opts['axis'].set_yscale(opts['yscale'])
    opts['axis'].set_xlim(opts['xlim'])
    opts['axis'].set_ylim(opts['ylim'])
    opts['axis'].set_aspect('equal', adjustable='box')
    opts['axis'].grid(True, which='both', linestyle='--', color='gray', alpha=0.3)
    
    # Handle reflines and axis limit
    # Ensuring x and y axes have equal limits
    all_data = np.concatenate([x[~upoint], y[~upoint]])
    
    ref1, ref2 = np.median(all_data)/100, np.median(all_data)*100
    str_ref1="{:e}".format(ref1)
    str_ref2="{:e}".format(round(ref2,-len(str(round(ref2)))))
    if opts['xscale'] == 'log' and opts['yscale'] == 'log':
        lim1 = float('1e' + str_ref1[str_ref1.index('e')+1:]) / 10
        lim2 = float('1e' + str_ref2[str_ref2.index('e')+1:]) * 10
    elif opts['xscale'] == 'linear' and opts['yscale'] == 'linear':
        lim1 = min([x, y])
        lim2 = max([x, y])

    if not opts['xlim'] and not opts['ylim']:
        opts['xlim'] = [lim1, lim2]
        opts['ylim'] = [lim1, lim2]

    # Plot ref lines
    opts['axis'].plot([lim1*1e-3, lim2*1e3], [lim1*1e-3, lim2*1e3], color='black',zorder=1,linewidth=1)
    opts['axis'].plot([2*lim1*1e-3, 2*lim2*1e3], [lim1*1e-3, lim2*1e3], linestyle='--', color='black',zorder=1,linewidth=1)
    opts['axis'].plot([lim1*1e-3, lim2*1e3], [2*lim1*1e-3, 2*lim2*1e3], linestyle='--', color='black',zorder=1,linewidth=1)
    
    
    opts['axis'].set_xscale(opts['xscale'])
    opts['axis'].set_yscale(opts['yscale'])
    opts['axis'].set_xlim(opts['xlim'])
    opts['axis'].set_ylim(opts['ylim'])

    # Handling class colors and markers
    unique_groups = np.unique(g[~np.isnan(g)])
    if not opts['classcolor']:
        opts['classcolor'] = generate_hue_colors(len(unique_groups))
        
    if not opts['marker']:
        opts['marker'] = [opts['marker']] * len(unique_groups)
        
    if not opts['classname']:
        if len(unique_groups)>1:
            opts['classname'] = [f'Class {x}' for x in unique_groups]
        else:
            opts['classname'] = ['Data']
        
    # Plotting
    scatter_plots = []
    for i, group in enumerate(unique_groups):
        idx = (g == group) & ~upoint
        scatter_plots.append(opts['axis'].scatter(x[idx], y[idx], label=opts['classname'][i],
                                           color=to_rgba(opts['classcolor'][i], opts['transparency']),
                                           edgecolors=opts['markeredgecolor'],
                                           s=opts['markersize'], marker=opts['marker'][i], zorder=2))
    
    # Plot unselected points if needed
    if opts['upointshow']:
        scatter_plots.append(opts['axis'].scatter(x[upoint], y[upoint], label=opts['upointname'],
                                           color=to_rgba(opts['upointcolor'], opts['transparency']),
                                           edgecolors=opts['upointedgecolor'],
                                           s=opts['markersize'], marker=opts['marker'][0], zorder=2))
    
    stats_text = stat_gen(opts['stat'],x[~upoint], y[~upoint])
    place_text(opts['axis'],stats_text,opts['anlocation'],fontsize=opts['anfontsize'])
    # Fit line
    if opts['fit']:
        if opts['xscale'] == 'log' and opts['yscale'] == 'log':
            log_x = np.log10(x[~upoint])
            log_y = np.log10(y[~upoint])
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
            fit_x = np.linspace(min(log_x), max(log_x), 100)
            fit_y = intercept + slope * fit_x
            opts['axis'].plot(10**fit_x, 10**fit_y, 'r-', label='Fit Line', zorder=3)
        else:
            slope, intercept, r_value, p_value, std_err = linregress(x[~upoint], y[~upoint])
            fit_x = np.linspace(min(x[~upoint]), max(x[~upoint]), 100)
            fit_y = intercept + slope * fit_x
            opts['axis'].plot(fit_x, fit_y, 'r-', label='Fit Line', zorder=3)
    
    # Setting labels, title, and legend
    opts['axis'].set_xlabel(opts['xlabel'], fontsize=opts['anfontsize'])
    opts['axis'].set_ylabel(opts['ylabel'], fontsize=opts['anfontsize'])
    if opts['title']:
        if opts['titlelocation'] == 'in':
            opts['axis'].text(0.5, 0.95, opts['title'],
                              fontfamily=opts['titlefont'],
                              fontsize=opts['titlefontsize'], 
                              ha='center', 
                              transform=opts['axis'].transAxes)
        else:
            opts['axis'].set_title(opts['title'], fontsize=opts['titlefontsize'],fontfamily=opts['titlefont'])
    
    if opts['legend'] == 'on':
        if opts['legendlocation'] == 'east outside':
            opts['legendlocation']='center left'
            bbox_loc=(1, 0.5)
        else:
            bbox_loc=None
        opts['axis'].legend(loc=opts['legendlocation'], bbox_to_anchor=bbox_loc,
                            fontsize=opts['legendfontsize'],
                            title=opts['legendtitle'],
                            ncol=opts['legendcolumn'],
                            frameon=opts['legendborder'])

    
    # Return plot elements for further customization
    for label in opts['axis'].get_xticklabels() + opts['axis'].get_yticklabels():
        label.set_fontfamily('Segoe UI')
        label.set_fontsize(10)
    
    # return {
    #     'scatter_plots': scatter_plots,
    #     'axis': opts['axis']
    # }


