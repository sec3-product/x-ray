#!/usr/bin/perl
#

use warnings;
use strict;
use Getopt::Std;


# you must run this script under $installer


# main
#
my $INSTALLER_ROOT=`pwd`;
chomp $INSTALLER_ROOT;

my %opt;
getopts('k', \%opt);
my ($keep) = @opt{ qw(k) };

if ( ! -d "build/llvm10" ) {
	`mkdir -p build/llvm10`;
}

print "Prepare classic flang llvm project ...\n";
if (-d "build/llvm10/classic-flang-llvm-project" && !$keep) {
	`rm -fr build/llvm10/classic-flang-llvm-project`;
}
if (! -d "build/llvm10/classic-flang-llvm-project") {
	`cd build/llvm10 && git clone git\@github.com:coderrect-inc/classic-flang-llvm-project.git`;
}
`cd build/llvm10/classic-flang-llvm-project && git checkout develop && git pull`;

print "Prepare flang ...\n";
if (-d "build/llvm10/flang" && !$keep) {
	`rm -fr build/llvm10/flang`;
}
if (!-d "build/llvm10/flang") {
	`cd build/llvm10 && git clone git\@github.com:coderrect-inc/flang.git`;
}
`cd build/llvm10/flang && git checkout develop && git pull`;


print "Starting docker container to build the project ...\n";
`docker run --rm --user=\$(id -u):\$(id -g) -v $INSTALLER_ROOT/build/llvm10:/build -v $INSTALLER_ROOT/dockerstuff/scripts:/scripts coderrect/flang:1.0 /scripts/build-custom-clang10.sh`;


print "Copying tarball ...\n";
`cd $INSTALLER_ROOT && rm -f custom_clang_10.tar.gz && cp build/llvm10/*.tar.gz .`;


__END__
