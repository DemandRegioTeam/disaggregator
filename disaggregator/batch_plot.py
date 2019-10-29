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
                                elc_consumption_HH_spatiotemporal)
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
        fig, ax = choropleth_map(ser,
                                 axtitle=s.format(y, dto.month, dto.day, h, m),
                                 **choro_kwargs)
        fig.savefig(fn, bbox_inches='tight')
        plt.close(fig)


def invoke_batch_plotting(df, **choro_kwargs):
    """
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
    invoke_batch_plotting(ambient_T(),
                          relative=False, interval=(-12, 40), cmap='jet',
                          unit='°C')

    invoke_batch_plotting(solar_irradiation(),
                          relative=False, interval=(0, 245), cmap='jet',
                          unit='Wh/m²', tspd=96)

    invoke_batch_plotting(elc_consumption_HH_spatiotemporal(),
                          relative=True, interval=(0, 1.54), cmap='jet',
                          unit='MW')

    df = pd.read_csv(_data_out('custom_disagg.csv'), index_col=0, engine='c')
    df.columns = [int(r) for r in range(0, 8760)]
    invoke_batch_plotting(df,
                          relative=True, interval=(0, 3.25), cmap='jet',
                          unit='MWh/h')
