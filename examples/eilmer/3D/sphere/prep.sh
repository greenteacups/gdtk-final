#! /bin/bash
# prep.sh
prep-gas ideal-air.inp ideal-air-gas-model.lua
e4shared --prep --job=sphere
e4shared --post --job=sphere --vtk-xml
