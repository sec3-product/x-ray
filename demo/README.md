
### Run x-ray demo

```sh
mkdir -p workspace
git clone --branch demo-v1 https://github.com/sec3-product/x-ray.git workspace/x-ray
```

Start a scan with one-line command, using docker:

```sh
docker run --rm --volume "$(pwd)/workspace:/workspace" \
    ghcr.io/sec3-product/x-ray:latest \
    /workspace/x-ray/demo
```


![](https://miro.medium.com/v2/resize:fit:875/1*n5LYU1L9ESy7QPiaB5kaUQ.png)

*▲ running X-Ray with docker*

![](https://miro.medium.com/v2/resize:fit:875/1*8eyK0FgfMZGvZgRrBZiMaQ.png)

*▲ X-Ray reported vulnerabilities*


