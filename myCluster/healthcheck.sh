#!/bin/sh
export JAVA_HOME=/java/jdk1.6.0_10
export PATH=$JAVA_HOME/bin:$PATH
checkFTP=`ps -ef|grep proftp|grep -v grep|wc -l`
if [ ${checkFTP} -lt 1 ]; then
echo ${checkFTP}
/usr/local/sbin/in.proftpd
fi

checkPing=`ping -c 1 192.168.40.1 |grep Unreachable|wc -l`
if [ ${checkPing} -gt 0 ]; then
pid=`ps -ef|grep java|grep myCluster|gawk '{print $2}'`
kill -9 $pid
ifconfig bond0:1 down
umount /mnt
exit
fi

checkCluster=`ps -ef|grep java|grep myCluster|grep -v grep|wc -l`
echo ${checkCluster}
if [ ${checkCluster} -lt 1 ]; then
echo ${checkCluster}
cd /root/myCluster
nohup java myCluster &
fi
cd /root/myCluster
#cp /dev/null nohup.out
