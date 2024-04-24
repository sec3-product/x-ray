<div align="center">
  <img height="170" src="https://github.com/jet-lab/jet-v1/raw/master/app/public/img/jet/jet_logomark_gradient.png" />

  <h1>Jet Protocol</h1>

  <p>
    <strong>Jet V1</strong>
  </p>

  <p>
    <img alt="Build Status" src="https://github.com/jet-lab/jet-v1/actions/workflows/build.yml/badge.svg" />
    <a href="https://discord.com/channels/880316176612343891"><img alt="Discord Chat" src="https://img.shields.io/discord/833805114602291200?color=blueviolet" /></a>
    <a href="https://opensource.org/licenses/AGPL-3.0"><img alt="License" src="https://img.shields.io/github/license/jet-lab/jet-v1?color=blue" /></a>
  </p>

  <h4>
    <a href="https://jetprotocol.io/">Website</a>
  </h4>
</div>

## Note

* The software in this repository is deprecated, and only considered useful for educational purpose
* You can find the successor protocol under active development in [jet-v2](https://github.com/jet-lab/jet-v2)

## Contributing

### Install Anchor

Directions can be found [here](https://project-serum.github.io/anchor/getting-started/installation.html).

### Build, deploy, and test programs

You may also need to install the NPM dependencies for tests

```
npm install
```

You can use anchor to build and test the program.

```
anchor test
```

## Run Frontend

### Initial localnet setup - startup a local validator and deploy

```
npm install
cd scripts
./localnet-start.sh
```

Set up frontend app environment variable to use localnet
```
vi ../app/.env # Set IDL = localnet
```

Run `ctrl-c.` to kill the solana-test-validator process to stop the localnet

### Startup a local validator and deploy

Install any other NPM dependencies in the app directory

```
cd ../app
npm install
```

`npm run dev` to run live-reloading dev environment

`npm run build` to compile app to the /build directory

`npm run check` typescript type-checker

`localhost:3000` in browser
