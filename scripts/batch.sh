#!/bin/bash
datapath=data/${1:mvtec}
mode=${2:-train}
savepath=${3:-results}
for subdataset in $(ls $datapath)
do
    if [ -d data/mvtec/$subdataset ]
    then
        scripts/run_dpbdf.sh $1/$subdataset $mode $savepath
    fi
done
