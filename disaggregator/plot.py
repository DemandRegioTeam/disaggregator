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

import math
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as PathEffects
from .config import get_config
from .data import database_shapes, transpose_spatiotemporal
logger = logging.getLogger(__name__)
ScaMap = plt.cm.ScalarMappable
cfg = get_config()


def choropleth_map(df, cmap='viridis', interval=None, annotate=None,
                   relative=True, colorbar_each_subplot=False,
                   add_percentages=True, **kwargs):
    """
    Plot a choropleth map (=a map with countries colored according to a value)
    for each column of data in given pd.DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Holding the values (required index: NUTS-3 codes)
    cmap : str or list, optional
        matplotlib colormap code(s)
    interval : tuple or str, optional
        if tuple: min/max-range e.g. (0, 1) | if str: find min/max autom.
    annotate: None, str or list
        If `str` or `list` used to write annotation on map, valid values are:
            'nuts3': annonate the nuts3-code
            `name` or `names`: annonate the name of the region
            `value` or `values`: annotate the passed values
            `percentage` or `percentages`: annonate the percentage of values
    relative : bool, optional
        Flag if to use relative values in <unit>/(km²) (default True) or
        to use absolutes values if False.
    colorbar_each_subplot : bool, optional
        Flag if to show a colorbar for each subplot (default False)
    add_percentages : bool, optional
        Flag if to add the percentage share into the axtitle (default True)
    """
    # Mend and/or transpose the data
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df.index, pd.DatetimeIndex):
        df = transpose_spatiotemporal(df)
    if annotate is None or annotate == '':
        annotate = []
    if isinstance(annotate, str):
        annotate = [annotate]

    ncols = kwargs.get('ncols', 0)
    nrows = kwargs.get('nrows', 0)
    suptitle = kwargs.get('suptitle', None)
    axtitle = kwargs.get('axtitle', '')
    unit = kwargs.get('unit', '-')
    fontsize = kwargs.get('fontsize', 6)
    sep = kwargs.get('sep', '\n')
    color = kwargs.get('color', 'black')
    rem = nrows * ncols - len(df.columns)
    shape_source_api = kwargs.get('shape_source_api', True)

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

    assert(isinstance(DE, gpd.GeoDataFrame))
    DF = pd.concat([DE, df], axis=1, join='inner')
    # Derive lat/lon tuple as representative point for each shape
    DF['coords'] = DF.geometry.apply(
        lambda x: x.representative_point().coords[:][0])
    DF['coords_WGS84'] = DF.to_crs({'init': 'epsg:4326'}).geometry.apply(
        lambda x: x.representative_point().coords[:][0])

    cols = df.columns
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

    if isinstance(cmap, str):
        cmap = [cmap for c in cols]
    if len(cols) == 1:
        colorbar_each_subplot = False
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

    if nrows == 0 or ncols == 0:
        if nrows != ncols or ncols < 0 or nrows < 0:
            logger.warn('When passing `nrows` and `ncols` both (!) must be '
                        'passed as int and be greater zero. Gathering these '
                        'values now based on the given pd.DataFrame.')
        nrows, ncols, rem = gather_nrows_ncols(len(cols))
    figsize = kwargs.get('figsize', (4*ncols*1.05, 6*nrows*1.1))

    i, j = [0, 0]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                           figsize=figsize)
    for a, col in enumerate(cols):
        if j == ncols:
            i += 1
            j = 0
        if colorbar_each_subplot:
            intervals.append((DF[col].min(), DF[col].max()))
        # First layer with grey'ish countries/regions:
        DE.plot(ax=ax[i, j], color='grey')
        # Second layer: make subplot
        (DF.dropna(subset=[col], axis=0)
           .plot(ax=ax[i, j], column=col, cmap=cmap[a],
                 vmin=intervals[a][0], vmax=intervals[a][1]))
        if not shape_source_api:
            ax[i, j].set_xlim(5.5, 15.3)
            ax[i, j].set_ylim(47.0, 55.3)
        # Deal with axes titles
        if isinstance(axtitle, str):
            if len(cols) == 1:
                ax[i, j].set_title('{}'.format(axtitle))
            else:
                if add_percentages:
                    ratio = '{:.1f}'.format(df[col].sum()/df.sum().sum()*100.)
                    ax[i, j].set_title('{} {} ({}%)'.format(axtitle, col,
                                                            ratio))
                else:
                    ax[i, j].set_title('{} {}'.format(axtitle, col))
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
                    s += ('' if np.isnan(row[col])
                          else '{:.0f}'.format(row[col]))
                    if relative:
                        s += '/km²'
                if ann in ['percentage', 'percentages']:
                    if relative:
                        s += ('' if np.isnan(df.loc[idx, col]) else
                              '{:.2%}'.format(df.loc[idx, col]
                                              / float(df[col].sum())))
                    else:
                        s += ('' if np.isnan(row[col]) else
                              '{:.2%}'.format(row[col]/DF[col].sum()))
            txt = ax[i, j].annotate(s=s, xy=row['coords'], fontsize=fontsize,
                                    horizontalalignment='center', color=color,
                                    verticalalignment='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=1,
                                                         foreground='silver')])
        j += 1

    # Deactivate possibly remaining axes
    for j in range(rem):
        ax[i, -(j+1)].axis('off')

    if isinstance(suptitle, str):
        fig.suptitle(suptitle, fontsize=20, weight='heavy')
        fig.subplots_adjust(top=0.82)

    fig.tight_layout()
    if colorbar_each_subplot:
        for a, axes in enumerate(ax.ravel().tolist()):
            sm = ScaMap(cmap=cmap[a], norm=plt.Normalize(vmin=intervals[a][0],
                                                         vmax=intervals[a][1]))
            sm._A = []
            cbar = fig.colorbar(sm, ax=axes, shrink=1.0, pad=0.01,
                                fraction=0.046,
                                orientation='horizontal', anchor=(0.5, 1.0),
                                format=mticker.StrMethodFormatter('{x:,g}'))
            cbar.set_label(unit)
    else:
        sm = ScaMap(cmap=cmap[0], norm=plt.Normalize(vmin=intervals[0][0],
                                                     vmax=intervals[0][1]))
        sm._A = []
        shr = 1.0 if ncols <= 2 else 0.5
        cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), shrink=shr, pad=0.01,
                            orientation='horizontal', anchor=(0.5, 1.0),
                            format=mticker.StrMethodFormatter('{x:,g}'))
        cbar.set_label(unit)

    add_license_to_figure(fig, geotag=True)
    return fig, ax


def heatmap_timeseries(df, **kwargs):
    """
    ToDo DocString
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


# --- HELPER FUNCTIONS --------------------------------------------------------


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


def set_ax_format(ax, axtitle=None, axtitlesize=None, axtitlebold=False,
                  xlabel=None, ylabel=None, hline=True, hline_y=0.0, xlim=None,
                  ylim=None, xlabel_visible=True, ylabel_visible=True,
                  y_axis_numeric=True, minorticks='auto', xticks=None,
                  xticklabels=None, yticks=None, yticklabels=None):
    # Backgroud and grid
    ax.set_facecolor('#d9d9d9')  # gray background
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


def add_license_to_figure(fig, license='CC BY 4.0', geotag=False,
                          into_ax=True, **kwargs):
    """
    Add a license (and possibly geotag) text to a passed figure object.
    """
    # Create the string to put into the figure
    s = 'License: {}'.format(license)
    if geotag:
        s += ('\nAdministrative boundaries: © GeoBasis-DE / BKG 2017; '
              'Generalization: FfE e.V.')
    # Set position
    if into_ax:
        # Get the position of the very first AxesSubplot
        pos = fig.get_axes()[0].get_position()
        x, y = (pos.x0 * 1.01, pos.y0 * 1.01)
    else:
        x, y = (kwargs.get('x', 0.05), kwargs.get('y', 0.05))

    fig.text(x, y, s, fontsize=6, color='gray',
             ha='left', va='bottom', alpha=0.5)
