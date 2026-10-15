# Changes
Here are the changes we made to the original mineflayer, mineflayer-pathfinder and prismarine-viewer.
We made these changes to enable better navigation in loopnav.
## LoopNav-mineflayer
### lib/plugins/physics.js
add logic to send action and position.

## LoopNav-mineflayer-pathfinder
### index.js
Increased the cost of diagonal movements to encourage the agent to follow straight paths, preventing zigzagging along edges and avoiding sharp changes in viewpoint.

### movement.js
Increased the cost of diagonal movements to encourage the agent to follow straight paths, preventing zigzagging along edges and avoiding sharp changes in viewpoint.


## LoopNav-prismarine-viewer
We changed from asynchronous to synchronous, resulting in smoother behavior.

### viewer.js
modify setFirstPersonCamera to enable smooth camera rotation.

### worldview.js
modify init, updatePosition, loadChunk, _loadChunks from asynchronous to synchronous.

### headless.js
Modified the transmission logic and added the sending of an end signal.