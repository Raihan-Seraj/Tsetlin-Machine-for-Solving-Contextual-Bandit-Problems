#!/bin/bash

VMS_DIR=${PWD}/python-vms
PROJECT=cb_tm
mkdir -p ${VMS_DIR}
virtualenv --no-download -p python3 ${VMS_DIR}/${PROJECT}
source ${VMS_DIR}/${PROJECT}/bin/activate
pip install --upgrade pip

# Install libraries
pip install numpy tqdm pyTsetlinMachine sklearn contextualbandits sympy matplotlib keras tensorflow tmu



