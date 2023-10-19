#!/bin/sh

if [ $# -lt 1 ]
then
    echo "run wait_service.sh with a address: wait_service.sh www.baidu.com !"
    exit 1
fi

times=100
period=5
addr=$1

# wati service to start
for cnt in $(seq $times)
do
    echo "Try to connect $addr time $cnt ..."
    ping  $addr -c 1 -W 1
    # echo "return code $?"
    if [ $? -eq 0 ]
    then
        echo "Connect $addr successfully!"
        exit 0
    fi
    sleep $period
done

echo "Wait deploy time out(500s) !"
exit -1
