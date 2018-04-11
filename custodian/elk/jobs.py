# coding: utf-8

from __future__ import unicode_literals, division

import logging
import os
import shutil
import subprocess

from monty.shutil import decompress_dir

from custodian.custodian import Job

"""
This module implements basic kinds of jobs for ELK runs.
"""

logger = logging.getLogger(__name__)

__author__ = "James K. Glasbrenner"
__version__ = "0.1"
__maintainer__ = "James K. Glasbrenner"
__email__ = "jglasbr2@gmu.edu"
__status__ = "alpha"
__date__ = "2/6/18"

ELK_INPUT_FILES = {
    "elk.in",
}

ELK_OUTPUT_FILES = [
    'INFO.OUT', 'KPOINTS.OUT', 'EIGVAL.OUT', 'DTOTENERGY.OUT', 'EFERMI.OUT',
    'EQATOMS.OUT', 'EVALCORE.OUT', 'FERMIDOS.OUT', 'GAP.OUT', 'GEOMETRY.OUT',
    'IADIST.OUT', 'LATTICE.OUT', 'LINENGY.OUT', 'MOMENT.OUT', 'MOMENTM.OUT',
    'OCCSV.OUT', 'RMSDVS.OUT', 'STATE.OUT', 'SYMCRYS.OUT', 'SYMLAT.OUT',
    'SYMSITE.OUT', 'TOTENERGY.OUT'
]


class ElkJob(Job):
    """
    A basic ELK job. Just runs whatever is in the directory. But conceivably
    can be a complex processing of inputs etc. with initialization.
    """

    def __init__(self, elk_cmd, output_file="ELK_OUT.txt",
                 stderr_file="ELK_ERR.txt", suffix="", final=True, backup=True,
                 rm_temp_binaries=False):
        """
        This constructor is necessarily complex due to the need for
        flexibility. For standard kinds of runs, it's often better to use one
        of the static constructors. The defaults are usually fine too.

        Args:
            elk_cmd (str): Command to run elk as a list of args. For example,
                if you are using mpirun, it can be something like
                ["mpirun", "elk"]
            output_file (str): Name of file to direct standard out to.
                Defaults to "ELK_OUT.txt".
            stderr_file (str): Name of file to direct standard error to.
                Defaults to "ELK_ERR.txt".
            suffix (str): A suffix to be appended to the final output. E.g.,
                to rename all ELK output from say ELK_OUT.txt to
                ELK_OUT.txt.relax1, provide ".relax1" as the suffix.
            final (bool): Indicating whether this is the final elk job in a
                series. Defaults to True.
            backup (bool): Whether to backup the initial input files. If True,
                the elk.in will be copied with a ".orig" appended. Defaults to
                True.
        """
        self.elk_cmd = elk_cmd
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.final = final
        self.backup = backup
        self.suffix = suffix
        self.rm_temp_binaries = rm_temp_binaries

    def setup(self):
        """
        Performs initial setup for ElkJob, including overriding any settings
        and backing up.
        """
        decompress_dir('.')

        if self.backup:
            for f in ELK_INPUT_FILES:
                shutil.copy(f, "{}.orig".format(f))

    def run(self):
        """
        Perform the actual ELK run.

        Returns:
            (subprocess.Popen) Used for monitoring.
        """
        cmd = list(self.elk_cmd)
        logger.info("Running {}".format(" ".join(cmd)))
        with open(self.output_file, 'w') as f_std, \
                open(self.stderr_file, "w", buffering=1) as f_err:
            # use line buffering for stderr
            p = subprocess.Popen(cmd, stdout=f_std, stderr=f_err)
        return p

    def postprocess(self):
        """
        Postprocessing includes renaming and gzipping where necessary.
        """
        for f in ELK_OUTPUT_FILES + [self.output_file, self.stderr_file]:
            if os.path.exists(f):
                if self.final and self.suffix != "":
                    shutil.move(f, "{}{}".format(f, self.suffix))
                elif self.suffix != "":
                    shutil.copy(f, "{}{}".format(f, self.suffix))

        # Remove temporary ELK binaries after run is completed.
        if self.rm_temp_binaries:
            if os.path.exists("EVALFV.OUT"):
                os.remove("EVALFV.OUT")

            if os.path.exists("EVALSV.OUT"):
                os.remove("EVALSV.OUT")

            if os.path.exists("EVECFV.OUT"):
                os.remove("EVECFV.OUT")

            if os.path.exists("EVECSV.OUT"):
                os.remove("EVECSV.OUT")

        # Remove continuation so if a subsequent job is run in
        # the same directory, will not restart this job.
        if os.path.exists("continue.json"):
            os.remove("continue.json")
