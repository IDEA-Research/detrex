#!/bin/bash -e

# cd to ideadet project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pytest --disable-warnings ./tests