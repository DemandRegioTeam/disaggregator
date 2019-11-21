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
This file is an executable script for multi-threaded batch plotting. Therefore
it is NOT a library and should not be imported into other modules.
"""

from disaggregator.config import get_config, _data_out
from disaggregator.data import (ambient_T, solar_irradiation,
                                elc_consumption_HH_spatiotemporal,
                                transpose_spatiotemporal)
from disaggregator.plot import choropleth_map
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
logger = logging.getLogger(__name__)


def create_plot(input):
    """
    The function for creating one single plot that gets pickled internally.
    """
    import numpy as np
    import datetime as dt
    t, ser, choro_kwargs = input
    fn = _data_out('batch/{:05d}_batch.png'.format(t))
    if os.path.isfile(fn):
        print('Skipping: {:04d} | '.format(t), end='')
    else:
        print('Plotting: {:04d} | '.format(t), end='')
        s = '{:04d}/{:02d}/{:02d} | {:02d}:{:02d}'
        y = choro_kwargs.get('year', get_config()['base_year'])
        tspd = choro_kwargs.get('tspd', 24)  # tspd: time steps per day
        doy = int(np.floor(t/tspd)) + 1      # doy:  day of year
        h = int(np.floor(24.0/tspd * (t % tspd)))
        m = int(((t % tspd) % (tspd/24.0)) * 15)
        dto = dt.datetime(y, 1, 1) + dt.timedelta(doy - 1)
        plt.rcParams['font.size'] = 16
        fig, ax = choropleth_map(ser,
                                 axtitle=s.format(y, dto.month, dto.day, h, m),
                                 **choro_kwargs)
        fig.savefig(fn, bbox_inches='tight')
        plt.rcParams['font.size'] = 10
        plt.close(fig)


def invoke_batch_plotting(df, **choro_kwargs):
def create_animation(name, dir_in=None, dir_out=None, extension='mp4', fps=24):
    """
    Create an animation from png files in a given directory and save it to
    another directory.

    Parameters
    ----------
    name : str
        A name for the created animation.
    dir_in : str
        A path to the directory. If None, it will look in _data_out/batch/
    dir_out : str
        A path to the directory. If None, it will save to _data_out/
    extension : str
        Either 'mp4' or 'gif'
    fps : int
        The number of frames per second. Default: 24
    """
    import datetime
    import imageio as im

    if dir_in is None:
        dir_in = _data_out('batch')
    if dir_out is None:
        dir_out = _data_out()

    list_png = [os.path.join(dir_in, f)
                for f in os.listdir(dir_in) if f.endswith('.png')]
    logger.info('Found {} pictures.'.format(len(list_png)))

    # Create new folder to put the animation into
    now = datetime.datetime.now()
    new_dir = ('{:04d}-{:02d}-{:02d}_{}'
               .format(now.year, now.month, now.day, name))
    new_path = os.path.join(dir_out, new_dir)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    logger.info('Loading all pictures. This may take a while.')
    images = []
    for i, file in enumerate(list_png):
        # Loading bar
        tmp_per = int(round(i/len(list_png) * 25))
        if i == 1:
            print("Working:")
            print("[#" + "-"*24 + "]", end="\r")
        elif i == len(list_png):
            print(" ", end="\r")
            print("[" + "#"*25 + "]")
        else:
            print(" ", end="\r")
            print("[" + "#"*tmp_per + "-"*(25-tmp_per) + "]", end="\r")
        # Appending
        images.append(im.imread(file))

    logger.info('Creating animation...')
    if extension == 'mp4':
        file_name = os.path.join(new_path, 'video_{}-fps.mp4'.format(fps))
        writer = im.get_writer(file_name, fps=fps)
        for image in images:
            writer.append_data(image)
        writer.close()
    elif extension == 'gif':
        file_name = os.path.join(new_path, 'animation.gif')
        im.mimsave(file_name, images)
    else:
        raise ValueError("The extension must be one of ['mp4', 'gif'] but "
                         "given was: {}".format(extension))

    logger.info('Done! The animation can be found here: {}'.format(file_name))
    return
    """
    import multiprocessing

    # make sure that output directory exists
    if not os.path.isdir(_data_out('batch')):
        os.mkdir(_data_out('batch'))

    # figure out the maximum number of CPU threads to be used for the job
    max_os_cpus = multiprocessing.cpu_count()
    use_cpus = get_config()['max_cpu_count']
    if not (isinstance(use_cpus, int) or isinstance(use_cpus, None)):
        raise ValueError('Wrong type for `max_cpu_count` in config given! Must'
                         ' be int or None, but is: {}'.format(type(use_cpus)))
    if (use_cpus is None or use_cpus == 0 or use_cpus >= max_os_cpus or
            -use_cpus >= max_os_cpus):
        threads = max_os_cpus
    elif 0 < use_cpus < max_os_cpus:
        threads = use_cpus
    elif -max_os_cpus < use_cpus < 0:
        threads = max_os_cpus + use_cpus
    else:
        raise ValueError('This cannot be! `use_cpus` is: {}'.format(threads))

    print('Creating plots with {} CPUs.'.format(threads))
    pool = multiprocessing.Pool(threads)
    input = zip([int(t) for t in list(df.columns)],
                [ser for t, ser in df.iteritems()],
                [choro_kwargs for i in df.columns])
    pool.map(create_plot, input)
    logger.info('...done!')
    return


"""
Please uncomment the sections you don't need and run the entire script in an
external (!) console.
"""
if __name__ == '__main__':
    invoke_batch_plotting(ambient_T().reset_index(drop=True),
                          relative=False, interval=(-12, 40), cmap='jet',
                          unit='°C')

    invoke_batch_plotting(solar_irradiation().reset_index(drop=True),
                          relative=False, interval=(0, 245), cmap='jet',
                          unit='Wh/m²', tspd=96)

    invoke_batch_plotting(elc_consumption_HH_spatiotemporal()
                          .reset_index(drop=True),
                          relative=True, interval=(0, 1.54), cmap='jet',
                          unit='MW')

    df = (pd.read_csv(_data_out('gas_disagg.csv'), index_col=0, engine='c')
            .pipe(transpose_spatiotemporal)
            .reset_index(drop=True))
    invoke_batch_plotting(df.loc[4344:4511],
                          relative=True, interval=(0, 3.25), cmap='jet',
                          unit='MWh/h')
