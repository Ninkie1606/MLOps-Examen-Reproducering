#!/bin/bash

prefect work-pool create --type process batch-pool --overwrite
prefect worker start -p batch-pool &

python /batch.py