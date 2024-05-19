#!/bin/bash

tar zxvf cherokee-0.9.2.tar.gz
cd cherokee-0.9.2/
./configure --prefix=$PWD
coderrect make



#  1) cget/.libs/cget
#  2) cherokee/cherokee
#  3) cherokee/.libs/cherokee-admin
#  4) cherokee/.libs/cherokee-worker
#  5) cherokee/.libs/cherokee-tweak
#  6) cherokee/.libs/libplugin_round_robin.a
#  7) cherokee/.libs/libplugin_round_robin.so
#  8) cherokee/.libs/libplugin_htpasswd.a
#  9) cherokee/.libs/libplugin_htpasswd.so
# 10) cherokee/.libs/libplugin_htdigest.a
# 11) cherokee/.libs/libplugin_htdigest.so
# 12) cherokee/.libs/libplugin_plain.a
# 13) cherokee/.libs/libplugin_plain.so
# 14) cherokee/.libs/libplugin_w3c.a
# 15) cherokee/.libs/libplugin_w3c.so
# 16) cherokee/.libs/libplugin_combined.a
# 17) cherokee/.libs/libplugin_combined.so
# 18) cherokee/.libs/libplugin_ncsa.a
# 19) cherokee/.libs/libplugin_ncsa.so
# 20) cherokee/.libs/libplugin_deflate.a
# 21) cherokee/.libs/libplugin_deflate.so