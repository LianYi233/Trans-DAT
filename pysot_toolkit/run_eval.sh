#!/bin/bash

export PYTHONPATH=/home/songshu/Code/02-project-ssd/01-project/23-Transt/TransT-github

for epoch in '0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' '0009' '0010'
#for epoch in '0020'
#for epoch in '0001' '0010' '0020' '0030' '0040' '0050' '0060' '0070' '0080' '0090' '0100' \
#                    '0110' '0120' '0130' '0140' '0150' '0160' '0170' '0180' '0190' '0200' \
#                    '0210' '0220' '0230' '0240'
do
  python eval.py --tracker_prefix nirrednir-smlr-3transt-${epoch}-NIR
done
