# coding: utf-8

from __future__ import unicode_literals, division

import logging
import os
import shutil
import subprocess

from monty.serialization import dumpfn, loadfn
from monty.shutil import decompress_dir

from custodian.custodian import Job
from custodian.utils import backup
from custodian.vasp.handlers import VASP_BACKUP_FILES
from custodian.vasp.interpreter import VaspModder
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
    'RMSDVS.OUT', 'SYMCRYS.OUT', 'SYMLAT.OUT', 'SYMSITE.OUT', 'TOTENERGY.OUT'
]


class ElkJob(Job):
    """
    A basic vasp job. Just runs whatever is in the directory. But conceivably
    can be a complex processing of inputs etc. with initialization.
    """

    def __init__(self, vasp_cmd, output_file="vasp.out",
                 stderr_file="std_err.txt", suffix="", final=True, backup=True,
                 settings_override=None, auto_continue=False):
        """
        This constructor is necessarily complex due to the need for
        flexibility. For standard kinds of runs, it's often better to use one
        of the static constructors. The defaults are usually fine too.

        Args:
            vasp_cmd (str): Command to run vasp as a list of args. For example,
                if you are using mpirun, it can be something like
                ["mpirun", "pvasp.5.2.11"]
            output_file (str): Name of file to direct standard out to.
                Defaults to "vasp.out".
            stderr_file (str): Name of file to direct standard error to.
                Defaults to "std_err.txt".
            suffix (str): A suffix to be appended to the final output. E.g.,
                to rename all VASP output from say vasp.out to
                vasp.out.relax1, provide ".relax1" as the suffix.
            final (bool): Indicating whether this is the final vasp job in a
                series. Defaults to True.
            backup (bool): Whether to backup the initial input files. If True,
                the INCAR, KPOINTS, POSCAR and POTCAR will be copied with a
                ".orig" appended. Defaults to True.
            settings_override ([dict]): An ansible style list of dict to
                override changes. For example, to set ISTART=1 for subsequent
                runs and to copy the CONTCAR to the POSCAR, you will provide::

                    [{"dict": "INCAR", "action": {"_set": {"ISTART": 1}}},
                     {"file": "CONTCAR",
                      "action": {"_file_copy": {"dest": "POSCAR"}}}]
            auto_continue (bool): Whether to automatically continue a run
                if a STOPCAR is present. This is very usefull if using the
                wall-time handler which will write a read-only STOPCAR to
                prevent VASP from deleting it once it finishes
        """
        self.vasp_cmd = vasp_cmd
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.final = final
        self.backup = backup
        self.suffix = suffix
        self.settings_override = settings_override
        self.auto_continue = auto_continue

    def setup(self):
        """
        Performs initial setup for VaspJob, including overriding any settings
        and backing up.
        """
        decompress_dir('.')

        if self.backup:
            for f in ELK_INPUT_FILES:
                shutil.copy(f, "{}.orig".format(f))

        if self.auto_continue:
            if os.path.exists("continue.json"):
                actions = loadfn("continue.json").get("actions")
                logger.info(
                    "Continuing previous VaspJob. Actions: {}".format(actions))
                backup(VASP_BACKUP_FILES, prefix="prev_run")
                VaspModder().apply_actions(actions)

            else:
                # Default functionality is to copy CONTCAR to POSCAR and set
                # ISTART to 1 in the INCAR, but other actions can be specified
                if self.auto_continue is True:
                    actions = [{
                        "file": "CONTCAR",
                        "action": {
                            "_file_copy": {
                                "dest": "POSCAR"
                            }
                        }
                    }, {
                        "dict": "INCAR",
                        "action": {
                            "_set": {
                                "ISTART": 1
                            }
                        }
                    }]
                else:
                    actions = self.auto_continue
                dumpfn({"actions": actions}, "continue.json")

        if self.settings_override is not None:
            VaspModder().apply_actions(self.settings_override)

    def run(self):
        """
        Perform the actual ELK run.

        Returns:
            (subprocess.Popen) Used for monitoring.
        """
        cmd = list(self.vasp_cmd)
        logger.info("Running {}".format(" ".join(cmd)))
        with open(self.output_file, 'w') as f_std, \
                open(self.stderr_file, "w", buffering=1) as f_err:
            # use line buffering for stderr
            p = subprocess.Popen(cmd, stdout=f_std, stderr=f_err)
        return p

    def postprocess(self):
        """
        Postprocessing includes renaming and gzipping where necessary.
        Also copies the magmom to the incar if necessary
        """
        for f in ELK_OUTPUT_FILES + [self.output_file]:
            if os.path.exists(f):
                if self.final and self.suffix != "":
                    shutil.move(f, "{}{}".format(f, self.suffix))
                elif self.suffix != "":
                    shutil.copy(f, "{}{}".format(f, self.suffix))

        # Remove continuation so if a subsequent job is run in
        # the same directory, will not restart this job.
        if os.path.exists("continue.json"):
            os.remove("continue.json")
