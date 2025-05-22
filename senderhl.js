const mineflayer = require('mineflayer')
const minimist = require('minimist')
const mineflayerViewerhl = require('prismarine-viewer').headless
const mineflayerViewer = require('prismarine-viewer').mineflayer
const { pathfinder, Movements, goals: { GoalNear, GoalNearXZ, GoalBlock } } = require('mineflayer-pathfinder')
const {get_target} = require('./targets.js')
const Vec3 = require('vec3').Vec3
const fs = require('fs')  // 添加 fs 模块

const args = minimist(process.argv.slice(2))
let tp_target = get_target(args.location)
let nv_type = args.nvtype
if (tp_target === null) {
  console.log("Invalid location")
  process.exit(1)
}
console.log("collecting in",args.location,tp_target," using ",nv_type,"type navigation")

const bot = mineflayer.createBot({
  username: args.name,
  host: 'localhost',
  port: 8105
})

bot.loadPlugin(pathfinder)

/*
  find the first non air block on given x,z
*/
function land_pos (x, z) {
  const pos = new Vec3(x, 64, z)
  let block = bot.blockAt(pos)

  if (block === null) {
    // unloaded chunk
    return null
  }
  let dy = 0
  while(block.type !== bot.registry.blocksByName.air.id) {
    dy++
    block = bot.blockAt(pos.offset(0, dy, 0))
    if (block.type === bot.registry.blocksByName.air.id) {
      return pos.offset(0, dy-1, 0)
    }
  }
  while (block.type === bot.registry.blocksByName.air.id) {
    dy--
    block = bot.blockAt(pos.offset(0, dy, 0))
    if (block.type !== bot.registry.blocksByName.air.id) {
      return pos.offset(0, dy, 0)
    }
  }
}
/*
  [-range,range] 区域内的一个(x,z)，该位置的xz坐标上最高的非空气方块
  *要求并且必须是泥土或者石头（放置goal在屋顶上的奇怪情况）
  *要求距离tp点不超过80格, 并且距离起始点至少20格
*/

function random_pos (range){
  const start_pos = bot.entity.position.clone()
  while (true) {
    const x = Math.floor(Math.random() * range * 2) - range
    const z = Math.floor(Math.random() * range * 2) - range
    let limit = range*4/5
    if (x*x + z*z < limit*limit) {
      // ensure the distance is not to short
      continue
    }
    // ensure the distance is not to far away from village center
    dx = start_pos.x+x - tp_target.x
    dz = start_pos.z+z - tp_target.z
    if (dx*dx + dz*dz > 80*80) {
      continue
    }
    if (args.location === 'stronghold' || args.location === 'nether_bastion' || args.location === 'nether_fortress') {
      return new Vec3(start_pos.x+x, start_pos.y, start_pos.z+z)
    }
    const pos = land_pos(start_pos.x+x, start_pos.z+z)
    if (Math.abs(pos.y - start_pos.y) > 10) {
      continue
    }
    landable = new Set([
      bot.registry.blocksByName.dirt.id, 
      bot.registry.blocksByName.stone.id, 
      bot.registry.blocksByName.grass_path.id, 
      bot.registry.blocksByName.sand.id, 
      bot.registry.blocksByName.grass_block.id, 
      bot.registry.blocksByName.snow.id,
      bot.registry.blocksByName.gravel.id,
      bot.registry.blocksByName.sandstone.id,
      bot.registry.blocksByName.red_sand.id,
      bot.registry.blocksByName.terracotta.id,
      bot.registry.blocksByName.mycelium.id,
      bot.registry.blocksByName.end_stone.id,
      bot.registry.blocksByName.nether_bricks.id,
      bot.registry.blocksByName.blackstone.id,
      bot.registry.blocksByName.polished_blackstone_bricks.id,
      bot.registry.blocksByName.cracked_polished_blackstone_bricks.id,
      bot.registry.blocksByName.netherrack.id,
    ])
    if (pos !== null){
      const block = bot.blockAt(pos)
      blockunder = bot.blockAt(pos.offset(0, -1, 0))
      if (landable.has(block.type) && landable.has(blockunder.type)){
        pos.y = pos.y + 1
        return pos
      }
      else {
        console.log('rej block type', block.type, blockunder.type)
      }
    }
  }
}
/*
  move to a random position in a range*range cube around the bot
*/
function move (range) {
  const pos = random_pos(range)
  console.log('moving to', pos)
  console.log('distance', bot.entity.position.distanceTo(pos))
  const defaultMove = new Movements(bot)
  defaultMove.allowSprinting = false
  bot.pathfinder.setMovements(defaultMove)
  if (args.location === 'stronghold' || args.location === 'nether_bastion' || args.location === 'nether_fortress') {
    bot.pathfinder.setGoal(new GoalNearXZ(pos.x, pos.z, 1), false)
  }
  else {
    bot.pathfinder.setGoal(new GoalBlock(pos.x, pos.y, pos.z), false)
  }
}

function* moveAround(range){
  const start_pos = bot.entity.position.clone()
  move(range)
  yield 
  if (nv_type === 'ABCA') {
    move(range)
    yield
  }
  console.log('moving back to', start_pos)
  const defaultMove = new Movements(bot)
  defaultMove.allowSprinting = false
  bot.pathfinder.setMovements(defaultMove)
  bot.pathfinder.setGoal(new GoalNearXZ(start_pos.x, start_pos.z, 1), false)
  yield
  setTimeout(() => {
    bot.emit('endtask')
  }, 3000)
}
function* lookaround () {
  let pitch = 0
  let yaw = 0
  bot.look(yaw, pitch, false)
  yield
  yaw = Math.PI/2
  bot.look(yaw, pitch, false)
  yield
  yaw = Math.PI
  bot.look(yaw, pitch, false)
  yield
  yaw = Math.PI*3/2
  bot.look(yaw, pitch, false)
  yield
  yaw = 0
  bot.look(yaw, pitch, false)
  yield
  let controller = moveAround(args.nvrange)
  controller.next()
  // Force end of stream
  bot.on('goal_reached', () => {
    console.log('goal reached')
    controller.next()
    // bot.quit()
  })
}
bot.once('spawn', () => {
  if (args.location === 'nether_bastion' || args.location === 'nether_fortress') {
    bot.chat(`/execute in minecraft:the_nether run tp @p ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  }
  else if (args.location === 'end_city' || args.location === 'end_mainland') {
    bot.chat(`/execute in minecraft:the_end run tp @p ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  }
  else {
    bot.chat(`/execute in minecraft:overworld run tp @p ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  }
  // wait for 1 seconds to let the bot tp
  setTimeout(() => {
    // First move to a random position with inital range
    move(20)
    
    // Track if we've started the viewer
    let viewerStarted = false;
    
    // Handle goal_reached events
    bot.on('goal_reached', () => {
      if (!viewerStarted) {
        // This is the first goal_reached (initial move)
        console.log('Reached initial position, starting viewer and lookaround')
        // Start the viewer after reaching the position
        mineflayerViewerhl(bot, { output: `127.0.0.1:${args.port}`, frames: -1, width: 640, height: 360 })
        viewerStarted = true;
        
        // Start lookaround after viewer is ready
        let lk = lookaround()
        bot.on('lookingdone', () => {
          lk.next()
        })
        lk.next()
      }
      // Subsequent goal_reached events will be handled by the lookaround function
    })

    bot.on('path_update', (r) => {
      const nodesPerTick = (r.visitedNodes * 50 / r.time).toFixed(2)
      console.log(`I can get there in ${r.path.length} moves. Computation took ${r.time.toFixed(2)} ms (${nodesPerTick} nodes/tick). ${r.status}`)
      if (r.status === 'timeout') {
        console.log('timeout, quit')
        if (bot.viewer) {
          bot.viewer.close()
        }
        bot.emit('abnormal_exit')
        process.exit(1)
      }
    })
    
    stuck_count = 0
    bot.on('path_stuck', () => {
      stuck_count += 1
      if (stuck_count > 5) {
        console.log('stuck for 5 times, quit')
        // Close the viewer connection before exiting
        if (bot.viewer) {
          bot.viewer.close()
        }
        bot.emit('abnormal_exit')
        process.exit(1)
      }
    })
  }, 1000)
})
