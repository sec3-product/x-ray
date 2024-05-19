# Build Instruction

### First Time Build
To build the Coderrect package from a fresh environment for the first time, you need to first build the installer docker image for later use.

- Run script `./build-docker-image.sh`

### Set up AWS Credentials and AWS CLI
Create a file under `~/.aws/credentials`, and put the following content into it (remember to keep a space around `=`):

```
[default]
aws_access_key_id = <AWS Key ID>
aws_secret_access_key = <AWS Key Secret>
region = us-east-2
```

To install AWS CLI, use command `sudo apt install awscli`

### Build Dev Package
You can use the follow command to build the newest develop package

```
./buildpkg -d
```

### Build Release Package

You can use the follow command to build the newest develop package

```
./buildpkg -r 1.0.0
```

### Upload package

You can use the follow command to upload a tarball to our aws server

```
./uploadpkg build/coderrect-linux-develop.tar.gz
```

