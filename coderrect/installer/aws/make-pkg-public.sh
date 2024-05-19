#!/bin/bash

public_bucket=public-installer-pkg
internal_bucket=internal-installer-pkg

function usage() {
        echo "Usage:"
	echo "./make-pkg-public.sh [-l <public|internal>] [-p <file>] [-d <public|internal> <file>]"
	echo "-l     Show files in buckets, you need specify which bucket you want to list:"
	echo "             public   - public-installer-pkg"
	echo "             internal - internal-installer-pkg"
	echo "-p     Publish package, you need give the file name in bucket internal-installer-pkg, it"
	echo "       will be copied into bucket public-installer-pkg."
	echo "-d     Delete file in bucket, you need specify the bucket and file name."
	echo "-h     Print help message."
}	

function getBucketName() {
        if [ $1 = "public" ];then
                bucket=$public_bucket
        elif [ $1 = "internal" ];then
                bucket=$internal_bucket
        else
                echo "Unknow bucket param, please check your input."
                usage
		exit 1
        fi
}

function listFiles() {
	getBucketName $1
	aws s3 ls s3://$bucket
}

function publish() {
	aws s3 cp s3://$internal_bucket/$1 s3://$public_bucket/
}

function deleteFile() {
	getBucketName $1
	aws s3 rm s3://$bucket/$2
}

#main
while getopts "l:p:d:h" arg
do
        case $arg in
		l)
			listFiles $OPTARG;;
		p)
			publish $OPTARG;;
		d)
			shift
			deleteFile $@;;
		h)
			usage;;	
		?)
                	echo "Unkown argument"
			usage
			exit 1
         esac
done

