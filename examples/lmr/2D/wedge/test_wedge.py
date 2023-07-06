# Author: Rowan J. Gollan
# Date: 2023-07-05
#
# Integration test for the 15 deg wedge case.

import pytest
import subprocess
import re
import os

# This is used to change to local directory so that subprocess runs nicely.
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

expected_number_steps = 29
expected_final_cfl = 2.577e+04
expected_number_steps_on_restart = 29
expected_final_cfl_on_restart = 2.600e+04

def expected_output(proc, expected_n_steps, expected_final_cfl):
    steps = 0
    cfl = 0.0
    lines = proc.stdout.split("\n")
    for line in lines:
        if line.find("FINAL-STEP") != -1:
            steps = int(line.split()[1])
        if line.find("FINAL-CFL") != -1:
            cfl = float(line.split()[1])
    assert steps == expected_n_steps, "Failed to take correct number of steps."
    assert abs(cfl - expected_final_cfl)/expected_final_cfl < 0.005, \
        "Failed to arrive at expected CFL value on final step."


def test_prep():
    make_targets = ['gas', 'grid', 'init']
    for tgt in make_targets:
        cmd = "make " + tgt
        proc = subprocess.run(cmd.split())
        assert proc.returncode == 0, "Failed during: " + cmd


def test_run():
    cmd = "lmr run-steady"
    proc = subprocess.run(cmd.split(), capture_output=True, text=True)
    assert proc.returncode == 0, "Failed during: " + cmd
    expected_output(proc, expected_number_steps, expected_final_cfl)

def test_snapshot():
    cmd = "lmr snapshot2vtk --all"
    proc = subprocess.run(cmd.split())
    assert proc.returncode == 0, "Failed during: " + cmd
    assert os.path.exists('lmrsim/vtk')


def test_restart():
    cmd = "lmr run-steady -s 1"
    proc = subprocess.run(cmd.split(), capture_output=True, text=True)
    assert proc.returncode == 0, "Failed during: " + cmd
    expected_output(proc, expected_number_steps_on_restart, expected_final_cfl_on_restart)

def test_cleanup():
    cmd = "make clean"
    proc = subprocess.run(cmd.split())
    assert proc.returncode == 0, "Failed during: " + cmd

