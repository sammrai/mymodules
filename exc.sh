#!/bin/sh

# cat $1>$2
awk '{ t = $1; $1 = $2; $2 = t; print; }' "$1" > "$2"
