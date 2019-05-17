# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:00:27 2018

@author: shine
"""

#! /usr/bin/env python
import logging
import logging.handlers
import os
import yaml
LOG_FILENAME = 'logging.out'

# Set up a specific logger with our desired output level
logger = logging.getLogger('MyLogger')
logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=1*1024*1024, backupCount=2)
formatter = logging.Formatter("[%(asctime)s] [%(filename)s][line:%(lineno)d][func: %(funcName)s] - [%(levelname)s] : %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

logger.warning('logger is work')

