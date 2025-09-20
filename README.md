# LoopNav

This project provides a framework for collecting navigation trajectories in the Minecraft environment using [Mineflayer](https://github.com/PrismarineJS/mineflayer). It supports controllable agents, path planning via A*[Mineflayer-pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder), and image rendering [Prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer).

Certain code is modified from above projects to support features. Modified version is in
[LoopNav-mineflayer](https://github.com/Kevin-lkw/mineflayer),
[LoopNav-pathfinder](https://github.com/Kevin-lkw/mineflayer-pathfinder),
[LoopNav-prismarine-viewer](https://github.com/Kevin-lkw/prismarine-viewer).

See [CHANGES.md](Changes.md) for details.

## ✨ Features

- Loop-based navigation (`A → B → A`) or (`A → B → C → A`)
- Recording observation, action, position, Goal.
- only one action is allowed on one timestep
- avoid stuck by edge of blocks


## 📦 Installation

```bash
git clone https://github.com/Kevin-lkw/LoopNav
cd LoopNav
npm install
```

## Usage

### Step 1. start a minecraft server on PORT

for example, in liunx server
```bash
java -Xmx2G -Xms2G -jar server.jar nogui
```

### Step 2. run the script
```bash
python run.py --name Bot --target village --output_path ./output
```
--name: the name of the bot
--target: the target to collect data, can be 'village', 'biome' or 'structure'
--output_path: the path to save the collected data

### Step 3. process the collected data
there might be some illegal data(e.g. the agent is died in lava or loss connection with the server)
we provide a script to process the data
this script will delete the illegal data and count the number of trajectories for each target
```bash
python process.py --output_path ./output --target village
```

## FAQ

- tp command failed: give the Bot OP permission to use tp command


