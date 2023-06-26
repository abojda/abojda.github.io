#!/usr/bin/env bash
docker run -v $(pwd):/site -p 4000:4000 -it ruby_jekyll bash -c 'cd site && jekyll serve --host 0.0.0.0'
