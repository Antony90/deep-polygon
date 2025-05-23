#!/bin/sh

if [ -n "$PROFILE" ]; then
    sleep 1
    echo Profiling is enabled
    # python -m cProfile -o profiling/profile.prof main.py "$@"
    py-spy record -o ./profiling/profile.svg -- python main.py "$@"
    # scalene --profile-all --outfile profile.html main.py "$@"
else
    echo Profiling is disabled
    exec python main.py "$@"
fi