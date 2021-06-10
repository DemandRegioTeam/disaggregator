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
disaggregator

A set of tools for processing of spatial and temporal disaggregations.
"""


from __future__ import absolute_import
import sys
import logging
from .config import data_out


# Logging: General Settings
class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level < LEVEL"""
    # http://stackoverflow.com/a/24956305/408556

    def __init__(self, level):
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should
        # be exclusive
        return record.levelno < self.level


MIN_LEVEL = logging.DEBUG
stdout_hdlr = logging.StreamHandler(sys.stdout)
stderr_hdlr = logging.StreamHandler(sys.stderr)
log_filter = LogFilter(logging.WARNING)
stdout_hdlr.addFilter(log_filter)
stdout_hdlr.setLevel(MIN_LEVEL)
stderr_hdlr.setLevel(max(MIN_LEVEL, logging.WARNING))
DisplayFormat = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', "%Y-%m-%d %H:%M:%S")
stdout_hdlr.setFormatter(DisplayFormat)
stderr_hdlr.setFormatter(DisplayFormat)
# messages lower than WARNING go to stdout
# messages >= WARNING (and >= STDOUT_LOG_LEVEL) go to stderr

rootLogger = logging.getLogger()
rootLogger.addHandler(stdout_hdlr)
rootLogger.addHandler(stderr_hdlr)
logger = logging.getLogger(__name__)
logging.basicConfig(level=10)
logger.setLevel('INFO')

# Logging to File
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] "
                                 "[%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler(data_out('disaggregator.log'))
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
# logger.info('Initialization complete.')
