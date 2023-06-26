#!/usr/bin/env bash

if [ -z "$1" ]
then
  # Use localhost by default
  IP=127.0.0.1
  echo -e "Serving Jekyll on localhost\n"
else
  IP=$1
  echo -e "Serving Jekyll on $IP\n"
fi

docker run -v $(pwd):/site -p $IP:4000:4000 -it ruby_jekyll bash -c 'cd site && jekyll serve --host 0.0.0.0'
