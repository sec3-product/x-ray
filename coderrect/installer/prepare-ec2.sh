#!/bin/bash


sudo mkfs -t ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt
sudo chmod 777 /mnt
cd /mnt && mkdir build && chmod 777 build
cd /mnt/build
git clone ssh://git@github.com/coderrect-inc/installer
git clone ssh://git@github.com/coderrect-inc/LLVM-Coderrect

