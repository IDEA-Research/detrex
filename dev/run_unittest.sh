#!/bin/bash -e

# cd to detrex project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pytest --disable-warnings ./tests