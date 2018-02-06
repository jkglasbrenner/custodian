# coding: utf-8

from __future__ import unicode_literals, division

import logging
import math
import os
import shutil
import subprocess

from monty.os.path import which
from monty.serialization import dumpfn, loadfn
from monty.shutil import decompress_dir
from pymatgen import Structure
from pymatgen.io.vasp import VaspInput, Incar, Outcar, Kpoints

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
                 auto_npar=True, auto_gamma=True, settings_override=None,
                 gamma_vasp_cmd=None, copy_magmom=False, auto_continue=False):
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
            auto_npar (bool): Whether to automatically tune NPAR to be sqrt(
                number of cores) as recommended by VASP for DFT calculations.
                Generally, this results in significant speedups. Defaults to
                True. Set to False for HF, GW and RPA calculations.
            auto_gamma (bool): Whether to automatically check if run is a
                Gamma 1x1x1 run, and whether a Gamma optimized version of
                VASP exists with ".gamma" appended to the name of the VASP
                executable (typical setup in many systems). If so, run the
                gamma optimized version of VASP instead of regular VASP. You
                can also specify the gamma vasp command using the
                gamma_vasp_cmd argument if the command is named differently.
            settings_override ([dict]): An ansible style list of dict to
                override changes. For example, to set ISTART=1 for subsequent
                runs and to copy the CONTCAR to the POSCAR, you will provide::

                    [{"dict": "INCAR", "action": {"_set": {"ISTART": 1}}},
                     {"file": "CONTCAR",
                      "action": {"_file_copy": {"dest": "POSCAR"}}}]
            gamma_vasp_cmd (str): Command for gamma vasp version when
                auto_gamma is True. Should follow the list style of
                subprocess. Defaults to None, which means ".gamma" is added
                to the last argument of the standard vasp_cmd.
            copy_magmom (bool): Whether to copy the final magmom from the
                OUTCAR to the next INCAR. Useful for multi-relaxation runs
                where the CHGCAR and WAVECAR are sometimes deleted (due to
                changes in fft grid, etc.). Only applies to non-final runs.
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
        self.auto_npar = auto_npar
        self.auto_gamma = auto_gamma
        self.gamma_vasp_cmd = gamma_vasp_cmd
        self.copy_magmom = copy_magmom
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

        if self.auto_npar:
            try:
                incar = Incar.from_file("INCAR")
                # Only optimized NPAR for non-HF and non-RPA calculations.
                if not (incar.get("LHFCALC") or incar.get("LRPA")
                        or incar.get("LEPSILON")):
                    if incar.get("IBRION") in [5, 6, 7, 8]:
                        # NPAR should not be set for Hessian matrix
                        # calculations, whether in DFPT or otherwise.
                        del incar["NPAR"]
                    else:
                        import multiprocessing
                        # try sge environment variable first
                        # (since multiprocessing counts cores on the current
                        # machine only)
                        ncores = os.environ.get('NSLOTS') or \
                            multiprocessing.cpu_count()
                        ncores = int(ncores)
                        for npar in range(int(math.sqrt(ncores)), ncores):
                            if ncores % npar == 0:
                                incar["NPAR"] = npar
                                break
                    incar.write_file("INCAR")
            except:
                pass

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
        Perform the actual VASP run.

        Returns:
            (subprocess.Popen) Used for monitoring.
        """
        cmd = list(self.vasp_cmd)
        if self.auto_gamma:
            vi = VaspInput.from_directory(".")
            kpts = vi["KPOINTS"]
            if kpts.style == Kpoints.supported_modes.Gamma \
                    and tuple(kpts.kpts[0]) == (1, 1, 1):
                if self.gamma_vasp_cmd is not None and which(
                        self.gamma_vasp_cmd[-1]):
                    cmd = self.gamma_vasp_cmd
                elif which(cmd[-1] + ".gamma"):
                    cmd[-1] += ".gamma"
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

        if self.copy_magmom and not self.final:
            try:
                outcar = Outcar("OUTCAR")
                magmom = [m['tot'] for m in outcar.magnetization]
                incar = Incar.from_file("INCAR")
                incar['MAGMOM'] = magmom
                incar.write_file("INCAR")
            except:
                logger.error('MAGMOM copy from OUTCAR to INCAR failed')

        # Remove continuation so if a subsequent job is run in
        # the same directory, will not restart this job.
        if os.path.exists("continue.json"):
            os.remove("continue.json")


class GenerateVaspInputJob(Job):

    def __init__(self, input_set, contcar_only=True, **kwargs):
        """
        Generates a VASP input based on an existing directory. This is typically
        used to modify the VASP input files before the next VaspJob.

        Args:
            input_set (str): Full path to the input set. E.g.,
                "pymatgen.io.vasp.sets.MPNonSCFSet".
            contcar_only (bool): If True (default), only CONTCAR structures
                are used as input to the input set.
        """
        self.input_set = input_set
        self.contcar_only = contcar_only
        self.kwargs = kwargs

    def setup(self):
        pass

    def run(self):
        if os.path.exists("CONTCAR"):
            structure = Structure.from_file("CONTCAR")
        elif (not self.contcar_only) and os.path.exists("POSCAR"):
            structure = Structure.from_file("POSCAR")
        else:
            raise RuntimeError("No CONTCAR/POSCAR detected to generate input!")
        modname, classname = self.input_set.rsplit(".", 1)
        mod = __import__(modname, globals(), locals(), [classname], 0)
        vis = getattr(mod, classname)(structure, **self.kwargs)
        vis.write_input(".")

    def postprocess(self):
        pass
