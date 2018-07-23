import logging

from Dates import date_finder
import sys, logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


dt=date_finder.DateFinder()
date_string="June 25th was a good date. Early 2012 wasn't so bad either."
for actual_date_string, indexes, captures in dt.extract_date_strings(date_string):
    logger.debug("Str: {}, idx: {}".format(actual_date_string, indexes))