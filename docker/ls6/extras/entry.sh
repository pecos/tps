#!/bin/bash

# This script can either be a wrapper around arbitrary command lines,
# or it will simply exec bash if no arguments were given
if [[ $# -eq 0 ]]; then
    exec "/bin/bash"
else
    exec "$@"
fi