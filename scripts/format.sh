#!/usr/bin/env bash

type=$1
if [ "$1" == "cf" ] ; then
    find src tests bindings -not -iname '*toml.c' -a \( -iname '*.h' -o -iname '*.c' \) | xargs clang-format -i
    if [ -n "$(git diff)" ] ; then
        echo "clang-format check failed"
        echo "$(git diff)"
        exit -1
    fi
fi

exit 0
