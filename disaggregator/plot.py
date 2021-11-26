# -*- coding: utf-8 -*-
# Written by Fabian P. Gotzens, 2019.

# This file is part of disaggregator.

# disaggregator is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.

# disaggregator is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.

# You should have received a copy of the GNU General Public License
# along with disaggregator.  If not, see <https://www.gnu.org/licenses/>.
"""
Provides functions for plotting
"""
# %% Imports

import math
import locale
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
from .config import get_config, data_in
from .data import database_shapes, transpose_spatiotemporal
logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'de')
plt.rcParams['axes.formatter.use_locale'] = True
ScaMap = plt.cm.ScalarMappable

# %% Main functions


def choropleth_map(df, cmap=None, interval=None, annotate=None,
                   annotate_zeros=False, relative=True,
                   colorbar_each_subplot=False, hide_colorbar=False,
                   add_percentages=False, add_sums=False, license_tag=True,
                   background=True, show_cities=False, **kwargs):
    """
    Plot a choropleth map* for each column of data in passed df.

    * Choropleth map = A map with countries colored according to a value.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Table holding the values (required index: NUTS-3 codes)
    cmap : str or list or Colormap instance, optional
        matplotlib colormap code(s)
    interval : <tuple> or <str> or <list of tuples> or <list of str>, optional
        Defines the interval(s) set on the colorbar.
        If `tuple`: min/max-range e.g. (0, 100) for all
        If `str`: find min/max autom.
        If `list of tuples`: min/max-range for each subplot.
    annotate: None, str or list
        If `str` or `list` used to write annotation on map, valid values are:
            'nuts3': annonate the nuts3-code
            `name` or `names`: annonate the name of the region
            `value` or `values`: annotate the passed values
            `percentage` or `percentages`: annonate the percentage of values
    annotate_zeros : bool, optional
        Flag to annotate regions whose values is zero (default False)
    relative : bool, optional
        Flag if to use relative values in <unit>/(km²) (default True) or
        to use absolutes values if False.
    colorbar_each_subplot : bool, optional
        Flag if to show a colorbar for each subplot (default False)
    hide_colorbar : bool, default False
        Flag to hide (True) or show (False) the colorbar below the plot.
    add_percentages : bool, optional
        Flag if to add the percentage share into the axtitle (default False)
    add_sums : bool, optional
        Flag if to add the sum of all values into the axtitle (default False)
    license_tag : bool, optional
        Flag if to write a license_tag into the figure (default True)
    background : bool, optional
        Flag if to plot a grey'ish background layer with all regions, so that
        regions that are NaN do still appear on the chart (default True)
    show_cities : bool, optional
        Flag if to plot dots with annotated cities to the map (default False)

    Returns
    -------
    fig, ax
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Mend and/or transpose the data
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df.index, pd.DatetimeIndex):
        df = transpose_spatiotemporal(df)
    if annotate is None or annotate == '':
        annotate = []
    if isinstance(annotate, str):
        annotate = [annotate]

    cfg = kwargs.get('cfg', get_config())
    ncols = kwargs.get('ncols', 0)
    nrows = kwargs.get('nrows', 0)
    suptitle = kwargs.get('suptitle', None)
    axtitle = kwargs.get('axtitle', '')
    unit = kwargs.get('unit', '-')
    fontsize = kwargs.get('fontsize', 12)
    fontsize_an = kwargs.get('fontsize', 6)
    sep = kwargs.get('sep', '\n')
    color = kwargs.get('color', 'black')
    edgecolor = kwargs.get('edgecolor', 'darkgrey')
    bgcolor = kwargs.get('bgcolor', 'grey')
    bgedgecolor = kwargs.get('bgedgecolor', 'white')
    rem = nrows * ncols - len(df.columns)
    shape_source_api = kwargs.get('shape_source_api', True)
    reg_filter = kwargs.get('reg_filter', None)
    default_cities = kwargs.get('default_cities', cfg.get(
        'default_cities',  # If not provided, use those from ARD Wetterkarte
        ['Kiel', 'Rostock', 'Hamburg', 'Hannover', 'Dresden', 'Berlin', 'Köln',
         'Frankfurt', 'Stuttgart', 'München']))
    highlight_cities = kwargs.get('highlight_cities',
                                  cfg.get('highlight_cities', []))
    extend = kwargs.get('extend', 'neither')
    mode = kwargs.get('mode', 'a4screen')
    plt.rcParams.update({'font.size': fontsize})

    if shape_source_api:
        # Perform request through RestfulAPI
        DE = database_shapes()
    else:
        # Load from local shape files
        base_year = cfg['base_year']
        if base_year >= 2016:
            nuts = 'NUTS_RG_01M_2016_4326_LEVL_3_DE'
        else:
            nuts = 'NUTS_RG_01M_2013'
        DE = (gpd.read_file(gpd.datasets.get_path(nuts))
                 .set_index('NUTS_ID'))

    if show_cities or any(highlight_cities):
        CI = get_cities()
        if show_cities:
            CI_default = CI.loc[lambda x: x['name'].isin(default_cities)]
        if any(highlight_cities):
            CI_high = CI.loc[lambda x: x['name'].isin(highlight_cities)]

    assert(isinstance(DE, gpd.GeoDataFrame))
    DF = pd.concat([DE, df], axis=1, join='inner')
    # Derive lat/lon tuple as representative point for each shape
    DF['coords'] = DF.geometry.apply(
        lambda x: x.representative_point().coords[:][0])
    DF['coords_WGS84'] = DF.to_crs('EPSG:4326').geometry.apply(
        lambda x: x.representative_point().coords[:][0])

    if reg_filter is not None:
        DE = DE.reindex(reg_filter, axis=0)
        DF = DF.reindex(reg_filter, axis=0)

    cols = df.columns

    # Handle units
    if isinstance(unit, str) or unit is None:
        units = [unit for c in cols]
    elif isinstance(unit, list):
        if len(unit) == len(cols):
            units = unit
        else:
            raise ValueError('The number of `unit`s does not match columns!')
    else:
        raise ValueError('Wrong datatype for `unit` given!')
    mod_units = []
    for unit in units:
        unit = '\\%' if unit == '%' else unit
        if relative:  # convert to unit per km²
            DF.loc[:, cols] = DF.loc[:, cols].apply(lambda x: x/DF.fl_km2)
            unit = '1' if unit == '-' else unit
            if unit[-2:] == '/a':
                unit = '[${} / (km² × a)$]'.format(unit[:-2])
            else:
                unit = '[${} / km²$]'.format(unit)
        else:
            unit = '[${}$]'.format(unit)
        mod_units.append(unit)

    # Try to get cmap from config file, by default use viridis
    if cmap is None:
        try:
            cmap = cfg.get('userdef_colormaps')['default']
        except(TypeError, KeyError):
            cmap = 'viridis'
    cmap = cmap_handler(cmap)
    if isinstance(cmap, str) or isinstance(cmap, LinearSegmentedColormap):
        cmap = [cmap for c in cols]

    # if len(cols) == 1:
    #     colorbar_each_subplot = False

    # Organize intervals if a colorbar is drawn for each subplot
    if colorbar_each_subplot:
        if isinstance(interval, tuple):
            intervals = [interval for c in cols]
        elif isinstance(interval, list):
            intervals = interval
        else:
            intervals = []
    else:
        if isinstance(interval, str) or (interval is None):
            intervals = [(DF[cols].min().min(), DF[cols].max().max())
                         for c in cols]
        else:
            intervals = [interval for c in cols]

    figsize, orientation = handle_plot_mode(mode, **kwargs)
    if nrows == 0 or ncols == 0:
        if nrows != ncols or ncols < 0 or nrows < 0:
            logger.warn('When passing `nrows` and `ncols` both (!) must be '
                        'passed as int and be greater zero. Gathering these '
                        'values now based on the given pd.DataFrame.')
        nrows, ncols, rem = gather_nrows_ncols(len(cols),
                                               orientation=orientation)

    if mode == 'quick':
        # In this special case override figsize gathered above:
        figsize = (4 * ncols * 1.05, 6 * nrows * 1.2)

    i, j = [0, 0]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                           figsize=figsize)
    for a, col in enumerate(cols):
        if j == ncols:
            i += 1
            j = 0
        if colorbar_each_subplot:
            intervals.append((DF[col].min(), DF[col].max()))
        # First layer with backgroundcolor'ish countries/regions:
        if background:
            DE.plot(ax=ax[i, j], color=bgcolor, edgecolor=bgedgecolor)
        # Second layer: make subplot
        (DF.dropna(subset=[col], axis=0)
           .plot(ax=ax[i, j], column=col, cmap=cmap[a], edgecolor=edgecolor,
                 vmin=intervals[a][0], vmax=intervals[a][1]))
        # Third layer: cities
        if show_cities:
            fs = {1: 14, 2: 12}.get(len(cols), 11)
            CI_default.plot(ax=ax[i, j], color=color)
            for idx, row in CI_default.iterrows():
                txt = ax[i, j].annotate(text=row['name'], xy=row['coords'],
                                        fontsize=fs, color=color,
                                        horizontalalignment='center',
                                        verticalalignment='bottom')
        if any(highlight_cities):
            fs = {1: 14, 2: 12}.get(len(cols), 11)
            CI_high.plot(ax=ax[i, j], color='red', marker='s')
            for idx, row in CI_high.iterrows():
                txt = ax[i, j].annotate(text=row['name'], xy=row['coords'],
                                        fontsize=fs, color='red',
                                        horizontalalignment='center',
                                        verticalalignment='top')
        if not shape_source_api:
            ax[i, j].set_xlim(5.5, 15.3)
            ax[i, j].set_ylim(47.0, 55.3)

        # Deal with axes titles
        if len(cols) == 1:
            axt = axtitle
        else:
            axt = '{} {}'.format(axtitle, col)
        if add_percentages:
            ratio = '{:.1f}'.format(df[col].sum()/df.sum().sum()*100.)
            axt = '{} {}'.format(axt, ratio)
        if add_sums:
            sums = '{:.0f} {}'.format(df[col].sum(), units[a])
            axt = '{} {}'.format(axt, sums)
        ax[i, j].set_title(axt)

        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

        # Add annotations
        for idx, row in DF.iterrows():
            s = ''
            for a, ann in enumerate(annotate):
                if a >= 1:
                    s += sep
                if ann == 'nuts3':
                    s += idx
                if ann in ['name', 'names']:
                    s += row.gen
                if ann in ['value', 'values']:
                    if annotate_zeros:
                        s += ('' if np.isnan(row[col])
                              else '{:.0f}'.format(row[col]))
                    else:
                        s += ('' if (np.isnan(row[col])
                                     or row[col] == 0.0)
                              else '{:.0f}'.format(row[col]))
                    if relative:
                        s += '/km²'
                if ann in ['percentage', 'percentages']:
                    if relative:
                        if annotate_zeros:
                            s += ('' if np.isnan(df.loc[idx, col]) else
                                  '{:.2%}'.format(df.loc[idx, col]
                                                  / float(df[col].sum())))
                        else:
                            s += ('' if (np.isnan(df.loc[idx, col])
                                         or df.loc[idx, col] == 0.0)
                                  else '{:.2%}'.format(df.loc[idx, col]
                                                       / float(df[col].sum())))
                    else:
                        if annotate_zeros:
                            s += ('' if np.isnan(row[col]) else
                                  '{:.2%}'.format(row[col]/DF[col].sum()))
                        else:
                            s += ('' if (np.isnan(row[col]) or row[col] == 0.0)
                                  else '{:.2%}'.format(row[col]/DF[col].sum()))
            txt = ax[i, j].annotate(text=s, xy=row['coords'],
                                    fontsize=fontsize_an,
                                    horizontalalignment='center', color=color,
                                    verticalalignment='center')
            txt.set_path_effects([PathEffects.withStroke(
                linewidth=1, foreground='silver')])
        j += 1

    # Deactivate possibly remaining axes
    for j in range(rem):
        ax[i, -(j+1)].axis('off')

    if isinstance(suptitle, str):
        fig.suptitle(suptitle, fontsize=20, weight='heavy')
        fig.subplots_adjust(top=0.97)

    if len(cols) == 1:
        fig.tight_layout()
    if not hide_colorbar:
        if colorbar_each_subplot:
            for a, axes in enumerate(ax.ravel().tolist()):
                if a >= len(cols):
                    break  # To deal with possibly deactivated remaining axes

                sm = ScaMap(cmap=cmap[a],
                            norm=plt.Normalize(vmin=intervals[a][0],
                                               vmax=intervals[a][1]))
                sm._A = []
                divider = make_axes_locatable(axes)
                cax = divider.append_axes("bottom", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    sm, cax=cax, shrink=1.0, pad=0.01, fraction=0.046,
                    orientation='horizontal', anchor=(0.5, 1.0),
                    format=mticker.StrMethodFormatter('{x:n}'),
                    extend=extend)
                cbar.set_label(mod_units[a])
            if len(cols) > 1:
                fig.tight_layout()
        else:
            sm = ScaMap(cmap=cmap[0],
                        norm=plt.Normalize(vmin=intervals[0][0],
                                           vmax=intervals[0][1]))
            sm._A = []
            shr = 1.0 if ncols <= 2 else 0.5
            cbar = fig.colorbar(
                sm, ax=ax.ravel().tolist(), shrink=shr, pad=0.01,
                orientation='horizontal', anchor=(0.5, 1.0),
                format=mticker.StrMethodFormatter('{x:n}'),
                extend=extend)
            cbar.set_label(mod_units[0])
        # if len(cols) > 1:
        #     fig.tight_layout()

    if license_tag:
        add_license_to_figure(fig, geotag=True, into_ax=True)

    plt.rcParams.update({'font.size': 10})  # reset rcParams to default
    return fig, ax


def heatmap_timeseries(df, **kwargs):
    """
    Plot a heatmap for a given time series df.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Please pass a DateTimeIndex'ed pd.DataFrame or "
                         "pd.Series.")

    nrows = len(df.columns)
    clabel = kwargs.get('clabel', '')
    figsize = kwargs.get('figsize', (12, 3*nrows))
    cmap = kwargs.get('cmap', 'viridis')
    vmin = kwargs.get('vmin', df.min().min())
    vmax = kwargs.get('vmax', df.max().max())
    hours = 24
    days = int(len(df) / hours)
    i, j = [0, 0]

    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=1, sharex=True,
                           squeeze=False)
    for col, ser in df.iteritems():
        dfs = pd.DataFrame(np.array(ser).reshape(days, hours)).T
        cax = ax[i, j].imshow(dfs, interpolation='nearest', cmap=cmap,
                              vmin=vmin, vmax=vmax)
        ax[i, j].set_aspect('auto')
        ax[i, j].set_title(col)
        ax[i, j].set_ylabel('Stunde')
        if i == nrows - 1:
            ax[i, j].set_xlabel('Tag des Jahres')
        i += 1

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    #  Dimensions [left, bottom, width, height] of the new axes
    ax_cbar = fig.add_axes([0.85, 0.05, 0.03, 0.90])
    cbar = plt.colorbar(cax, cax=ax_cbar)
    cbar.set_label(clabel)
    add_license_to_figure(fig, into_ax=False)
    return fig, ax


def multireg_generic(df, **kwargs):
    """
    Plot a generic pd.DataFrame with regions in columns and e.g. DateTimeIndex
    as index. Each column/region will get its own subplot.
    """
    from matplotlib.offsetbox import AnchoredText
    # Mend and/or transpose the data
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # Handle keyword-arguments
    multiindexed = (df.columns.nlevels > 1)
    sharex = kwargs.get('sharex', True)
    sharey = kwargs.get('sharey', False)
    ncols = kwargs.get('ncols', 0)
    nrows = kwargs.get('nrows', 0)
    kind = kwargs.get('kind', 'line')
    suptitle = kwargs.get('suptitle', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    ylim = kwargs.get('ylim', None)  # valid: NoneType, tuple, list w/ 2 elem.
    color = kwargs.get('color', None)
    mode = kwargs.get('mode', 'screen')  # valid: ['screen', 'print']
    dpi = kwargs.get('dpi', 50)
    stats = kwargs.get('stats', False)
    legend = kwargs.get('legend', multiindexed)
    legend_cols = kwargs.get('legendcols', 6)
    legend_loc = kwargs.get('legend_loc', 8)
    show_means = kwargs.get('show_means', False)
    means_loc = kwargs.get('means_loc', 1)
    unit = kwargs.get('unit', '')
    if kwargs.get('fontsize'):
        orig_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': kwargs.get('fontsize')})

    if mode == 'screen':
        # Figsize close to FullHD Resolution
        figsize = kwargs.get('figsize', (27, 15))
        orientation = 'landscape'
    elif mode == 'print':
        # Figsize for Din (A2, A3, A4, etc.) paper print
        figsize = kwargs.get('figsize', (16.54, 23.38))
        orientation = 'portrait'
    else:
        raise ValueError('Wrong print mode given!')

    i, j = [0, 0]
    if ncols == 0 and nrows == 0:
        if multiindexed:
            # in this case take the number of outer (!) columns
            nrows, ncols, rem = gather_nrows_ncols(len(df.columns.levels[0]),
                                                   orientation)
        else:
            nrows, ncols, rem = gather_nrows_ncols(len(df.columns),
                                                   orientation)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                           sharex=sharex, sharey=sharey, figsize=figsize,
                           dpi=dpi)
    if multiindexed:
        for reg in df.columns.levels[0]:
            if j == ncols:
                i += 1
                j = 0
            if stats:
                df_stats = pd.DataFrame().assign(min=df[reg].min(axis=1),
                                                 mean=df[reg].mean(axis=1),
                                                 max=df[reg].max(axis=1))
                df_stats.plot(ax=ax[i, j], kind=kind, legend=False)
            else:
                df[reg].plot(ax=ax[i, j], kind=kind, legend=False, lw=0.5,
                             color=color)
            set_ax_format(ax=ax[i, j], ylim=ylim, axtitle=reg, ylabel=ylabel,
                          xlabel=xlabel)
            if show_means:
                s = [u'∅ {}: {:+.2f} {}'.format(c, col.mean(), unit)
                     for c, col in df[reg].iteritems()]
                txt = AnchoredText('\n'.join(s), loc=means_loc,
                                   prop={'size': 12})
                txt.patch.set(boxstyle='round', alpha=0.5)
                ax[i, j].add_artist(txt)
            j += 1
    else:
        for reg, col in df.iteritems():
            if j == ncols:
                i += 1
                j = 0
            col.plot(ax=ax[i, j], kind=kind, legend=False)
            set_ax_format(ax=ax[i, j], ylim=ylim, axtitle=reg, ylabel=ylabel,
                          xlabel=xlabel)
            if show_means:
                txt = AnchoredText(u'mean: {:+.2f}'.format(col.mean()),
                                   loc=1, prop={'size': 12})
                txt.patch.set(boxstyle='round', alpha=0.5)
                ax[i, j].add_artist(txt)
            j += 1
    fig.tight_layout()
    if isinstance(suptitle, str):
        fig.suptitle(suptitle, fontsize=20, weight='heavy')
        fig.subplots_adjust(top=0.95)
    if legend:
        handles, labels = ax[0, 0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc=legend_loc, ncol=legend_cols)
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        fig.subplots_adjust(bottom=0.07)
    # reset rcParams:
    if kwargs.get('fontsize'):
        plt.rcParams.update({'font.size': orig_size})
    add_license_to_figure(fig)
    return fig, ax


# %% Utility functions


def get_cities(types=['city', 'town']):
    CI = (gpd.read_file(data_in('regional', 'shapes', 'germany_places',
                                'places.shp'), encoding='utf-8')
          .loc[lambda x: x['type'].isin(types)]
          .dropna(subset=['name'])
          .sort_values(by='name')
          .to_crs('EPSG:25832'))
    CI['coords'] = CI.geometry.apply(
        lambda x: x.representative_point().coords[:][0])
    return CI


def gather_nrows_ncols(x, orientation='landscape'):
    """
    Derive the number of rows and columns and a possible remainder based on a
    given number of plots, in such a way that a subplot looks nicely.

    Parameters
    ----------
    x : int
        Number of subplots between [0, 42]
    orientation : str
        must be one of ['landscape', 'portrait']
    """
    def calc(n, m):
        if n <= 0:
            n = 1
        if m <= 0:
            m = 1
        while (n*m < x):
            m += 1
        return n, m

    if not isinstance(x, int):
        raise ValueError('An integer needs to be passed to this function.')
    elif x <= 0:
        raise ValueError('The given number of subplots is less or equal zero.')
    elif x > 42:
        raise ValueError("Are you sure that you want to put more than 42 "
                         "subplots in one diagram?\n You better don't, it "
                         "looks squeezed. Otherwise adapt the code.")
    k = math.sqrt(x)
    if k.is_integer():
        return [int(k), int(k), 0]  # Square format!
    else:
        k = int(math.floor(k))
        # Solution 1
        n, m = calc(k, k+1)
        sol1 = {'n': n, 'm': m, 'dif': (m*n) - x}
        # Solution 2:
        n, m = calc(k-1, k+1)
        sol2 = {'n': n, 'm': m, 'dif': (m*n) - x}
        if (((sol1['dif'] <= sol2['dif']) & (sol1['n'] >= 2))
                | (x in [7, 13, 14])):
            n, m = [sol1['n'], sol1['m']]
        else:
            n, m = [sol2['n'], sol2['m']]
        remainder = m*n - x
        if orientation == 'landscape':
            return n, m, remainder
        elif orientation == 'portrait':
            return m, n, remainder
        else:
            raise ValueError('Wrong `orientation` given!')


def handle_plot_mode(s, **kwargs):
    s = s.lower()
    if s == 'screen' or s == 'fullhd':  # Close to FullHD Resolution
        figsize = (27, 15)
        orientation = 'landscape'
    elif s == 'a4print':  # For Din (A2, A3, A4, etc.) paper print
        figsize = (16.54, 23.38)
        orientation = 'portrait'
    elif s == 'a4screen':  # for Din A4 slide format
        figsize = (23.38, 16.54)
        orientation = 'landscape'
    elif s == 'manual':
        figsize = kwargs.get('figsize')
        orientation = kwargs.get('orientation')
    elif s == 'quick':
        figsize = None  # Will be overridden anyway!
        orientation = 'landscape'
    else:
        raise ValueError("Wrong plot mode given! Must be any of ['screen', "
                         "'a4print', 'a4screen', 'manual']")
    return figsize, orientation


def set_ax_format(ax, axtitle=None, axtitlesize=None, axtitlebold=False,
                  xlabel=None, ylabel=None, hline=True, hline_y=0.0, xlim=None,
                  ylim=None, xlabel_visible=True, ylabel_visible=True,
                  y_axis_numeric=True, minorticks='auto', xticks=None,
                  xticklabels=None, yticks=None, yticklabels=None):
    # Backgroud and grid
    ax.set_facecolor('lightgray')  # gray background  #d9d9d9
    ax.set_axisbelow(True)       # puts the grid behind the bars
    ax.grid(color='white', linestyle='dotted')
    # axtitle
    if axtitle is not None:
        weight = 'bold' if axtitlebold else 'normal'
        if axtitlesize is None:
            ax.set_title(axtitle, size=plt.rcParams['font.size']+2,
                         weight=weight)
        else:
            ax.set_title(axtitle, size=plt.rcParams['font.size']+axtitlesize,
                         weight=weight)
    # xlabel
    if xlabel is not None and xlabel_visible:
        ax.set_xlabel(xlabel)
    else:
        ax.xaxis.label.set_visible(False)
    # ylabel
    if ylabel is not None and ylabel_visible:
        ax.set_ylabel(ylabel)
    else:
        ax.yaxis.label.set_visible(False)
    # xlim
    if xlim is not None:
        ax.set_xlim(xlim)
    # ylim
    if ylim is not None:
        ax.set_ylim(ylim)
    # hline
    if hline:
        ax.axhline(y=hline_y, color='#050505')
    if y_axis_numeric:
        (ax.get_yaxis()
           .set_major_formatter(mticker.StrMethodFormatter('{x:,g}')))
    if minorticks == 'off':
        ax.minorticks_off()
    elif minorticks == 'on':
        ax.minorticks_on()
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)


def add_license_to_figure(fig, license='CC BY 4.0', geotag=True,
                          into_ax=True, **kwargs):
    """
    Add a license (and possibly geotag) text to a passed figure object.
    """
    # Create the string to put into the figure
    s = 'License: {}'.format(license)
    if geotag:
        s += ('\nAdministrative boundaries: © GeoBasis-DE\n'
              '/ BKG 2017; Generalization: FfE e.V.')
    # Set position
    if into_ax:
        # Get the very first AxesSubplot
        ax = fig.get_axes()[0]
        ax.text(x=0, y=0, s=s, transform=ax.transAxes, fontsize=6,
                alpha=0.5, color='gray',
                horizontalalignment='left', verticalalignment='bottom')
    else:
        logger.warning("This is currently buggy and may distort your figure!")
        x, y = (kwargs.get('x', 0.5), kwargs.get('y', 0.01))
        fig.text(x, y, s, fontsize=6, color='gray',
                 ha='left', va='bottom', alpha=0.5)


def cmap_handler(cmap, **kwargs):
    """
    Handle existing and user-defined colormaps.
    """
    from matplotlib.colors import (Colormap, ListedColormap,
                                   LinearSegmentedColormap)

    # if passed `cmap` is a Colormap instance: return it directly
    if (isinstance(cmap, Colormap) or isinstance(cmap, ListedColormap)
            or isinstance(cmap, LinearSegmentedColormap)):
        return cmap

    elif isinstance(cmap, str):
        # if passed `cmap` is an existing colormap string: return it directly
        if cmap in plt.colormaps():
            return cmap

        # handle user-defined `cmap` string
        cfg = kwargs.get('cfg', get_config())
        if 'userdef_colors' not in cfg:
            raise ValueError("config.yaml doesn't contain `userdef_colormaps`")

        f_cmaps = cfg['userdef_colors']['file']
        sheet_name = cfg['userdef_colors']['sheet_cbars']
        ser_cmap = (pd.read_excel(f_cmaps, sheet_name=sheet_name,
                                  index_col='order')
                       .dropna(axis=0, how='all')
                       .loc[lambda x: x['cmap'] == cmap])['hex']
        if len(ser_cmap) == 0:
            raise ValueError(f"'{cmap}' not contained in file \n{f_cmaps}")

        nodes = np.linspace(0, 1, len(ser_cmap))
        return LinearSegmentedColormap.from_list(
            cmap, list(zip(nodes, ser_cmap)))

    elif isinstance(cmap, list):
        logger.warning('list-handling is not yet supported by `cmap_handler`, '
                       'returning as list!')
        return cmap
    else:
        raise ValueError("`cmap` must be str or a Colormap instance!")


def color_name2hex(**kwargs):
    # handle user-defined `cmap` string
    cfg = kwargs.get('cfg', get_config())
    f = cfg['userdef_colors']['file']
    sheet_name = cfg['userdef_colors']['sheet_colors']
    return pd.read_excel(f, sheet_name=sheet_name,
                         index_col='Python Name')['Hex'].to_dict()
