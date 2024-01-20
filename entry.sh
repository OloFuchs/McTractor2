#!/bin/bash -ex
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# sudo apt-get update
# sudo apt-get install libturbojpeg
# pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git

$DIR/bootstrap.sh $DIR $DIR/venv

DISPLAY=:0 $DIR/venv/bin/python $DIR/src/main.py $@

exit 0
