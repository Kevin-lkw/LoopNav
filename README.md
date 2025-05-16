# LoopNav

This project provides a framework for collecting navigation trajectories in the Minecraft environment using [Mineflayer](https://github.com/PrismarineJS/mineflayer). It supports controllable agents, path planning via A*[Mineflayer-pathfinder](https://github.com/Kevin-lkw/mineflayer-pathfinder), and image rendering [Prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer).

Certain code is modified from above projects to support features. Modified version is in
[LoopNav-mineflayer](https://github.com/Kevin-lkw/prismarine-viewer),
[LoopNav-pathfinder](https://github.com/Kevin-lkw/mineflayer-pathfinder),
[LoopNav-prismarine-viewer](https://github.com/Kevin-lkw/prismarine-viewer).

See [CHANGES] for details.

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

### Step 1. start a minecraft server
Assume the port is PORT.

### Step 2. run the script
```bash
node 
```


## Changes
