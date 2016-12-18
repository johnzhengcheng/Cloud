#!/bin/sh
ifconfig bond0:1 192.168.40.30 netmask 255.255.255.0 up
umount /mnt
mount -t ext3 /dev/sda1 /mnt
