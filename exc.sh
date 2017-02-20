#!/bin/sh

CMDNAME=`basename $0`
if [ $# -ne 2 ]; then
  echo "Usage: $CMDNAME file1 file2" 1>&2
  exit 1
fi

awk '{ t = $1; $1 = $2; $2 = t; print; }' "$1" > .temp
cat .temp >"$2"
rm .temp
