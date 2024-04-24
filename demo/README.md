## Demo

In order to scan a target repository, simply clone it to local environment, and run `coderrect` it in the project's root directory (no need to look for the source code location)

As a demo, we choose [jet-v1](https://github.com/jet-lab/jet-v1), a smart contract on Solana, as our target repo:

```
git clone https://github.com/jet-lab/jet-v1
cd jet-v1
coderrect -t .
```
The result of command-line tool:

<img src="../docs/images/jet-v1-result-backend.jpg" width="700px">

The result on X-ray online (same input repo):

<img src="../docs/images/jet-v1-result-web.jpg" width="300px">

An example of vulnerbility labeled on command-line tool:

<img src="../docs/images/jet-v1-label-backend.png" width="900px">

The same vulnerbility labeled on X-ray online:

<img src="../docs/images/jet-v1-label-web.png" width="750px">
