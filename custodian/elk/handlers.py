# coding: utf-8

from __future__ import unicode_literals, division

from custodian.custodian import ErrorHandler

"""
This module implements specific error handlers for ELK runs. These handlers
tries to detect common errors in ELK runs and attempt to fix them on the fly
by modifying the input files.
"""

__author__ = "James K. Glasbrenner"
__version__ = "0.1"
__maintainer__ = "James K. Glasbrenner"
__email__ = "jglasbr2@gmu.edu"
__status__ = "alpha"
__date__ = "4/11/18"

ELK_BACKUP_FILES = {
    "elk.in", "INFO.OUT", "ELK_ERR.txt", "ELK_OUT.txt"
}


class UnconvergedErrorHandler(ErrorHandler):
    """
    Check if a run is converged.
    """
    is_monitor = False

    def __init__(self, output_filename="INFO.OUT"):
        """
        Initializes the handler with the output file to check.

        Args:
            output_filename (str): Filename for the INFO.OUT file. Change
                this only if it is different from the default (unlikely).
        """
        self.output_filename = output_filename

    def check(self):
        pass

    def correct(self):
        pass
