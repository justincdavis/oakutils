#!/usr/bin/env bash
find . -name '*.py' -exec pyupgrade --py38-plus {} +
