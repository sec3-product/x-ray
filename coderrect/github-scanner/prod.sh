#/bin/bash

location=$pwd
if [ "$1" != "" ]; then
    location=$1
else
    echo "Example: ./run.sh /path/to/www"
    exit 1
fi

docker run -p 80:8080 \
    -d --restart always \
    -v $location:/www \
    -v $(pwd)/.data/logs:/opt/app/logs  \
    -v $(pwd)/.data/data:/data  \
    --name coderrect_github \
    coderrect/github-app:dev
