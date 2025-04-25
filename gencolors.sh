#!/bin/bash
"$(dirname "$(realpath "$0")")"/colormap.py "$1" | xsel -bi
