const mineflayer = require('mineflayer')
const minimist = require('minimist')
const mineflayerViewerhl = require('prismarine-viewer').headless
const mineflayerViewer = require('prismarine-viewer').mineflayer
const { pathfinder, Movements, goals: { GoalNearXZ, GoalBlock } } = require('mineflayer-pathfinder')
let getTargetModule
try {
  getTargetModule = require('./targets.js')
} catch (err) {
  getTargetModule = require('./target.js')
}
const { get_target } = getTargetModule
const Vec3 = require('vec3').Vec3
const fs = require('fs')  // 添加 fs 模块
const { installSmoothCamera } = require('./smooth_camera')

const args = minimist(process.argv.slice(2))
let tp_target = get_target(args.location)
let nv_type = args.nvtype
const symmetricReturn = Boolean(args.symmetric_return || args['symmetric-return'])
const mcPort = Number(args['mc-port'] || args.mc_port || 8105)
const endHoldMs = Number(args['end-hold-ms'] ?? args.end_hold_ms ?? 0)
const abacaMinAngleDeg = Number(args['abaca-min-angle'] ?? args.abaca_min_angle ?? 45)
const postTeleportWaitMs = Number(args['post-tp-wait-ms'] ?? args.post_tp_wait_ms ?? 5000)
const randomPosMaxAttempts = Number(args['random-pos-max-attempts'] ?? args.random_pos_max_attempts ?? 20000)
if (tp_target === null) {
  console.log("Invalid location")
  process.exit(1)
}
console.log("collecting in",args.location,tp_target," using ",nv_type,"type navigation")

const bot = mineflayer.createBot({
  username: args.name,
  host: 'localhost',
  port: mcPort
})

let smoothCamera = null
bot.loadPlugin(pathfinder)

let routePhase = 'setup'
let currentWaypointIndex = null
let returnWaypointTotal = 0
let outboundWaypoints = []
let recordingOutbound = false
let lastEmittedYaw = null
let lastEmittedPitch = null
let currentCycle = null
let currentTargetLabel = null
let currentTargetPosition = null

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

async function waitForReadyChunks () {
  await sleep(postTeleportWaitMs)
  if (typeof bot.waitForChunksToLoad !== 'function') return
  try {
    await Promise.race([
      bot.waitForChunksToLoad(),
      sleep(5000)
    ])
  } catch (err) {
    console.log('waitForChunksToLoad failed', err.message)
  }
}

function useSymmetricReturn () {
  return symmetricReturn && nv_type !== 'ABCA'
}

function finishTask () {
  currentWaypointIndex = null
  bot.clearControlStates()
  if (endHoldMs > 0) {
    routePhase = 'return_hold'
    setTimeout(() => {
      bot.emit('endtask')
    }, endHoldMs)
    return
  }
  bot.emit('endtask')
}

function waypointFromPosition (pos) {
  return new Vec3(Math.floor(pos.x), Math.floor(pos.y), Math.floor(pos.z))
}

function waypointKey (pos) {
  return `${pos.x},${pos.y},${pos.z}`
}

function angleDelta (target, current) {
  let delta = target - current
  while (delta <= -Math.PI) delta += 2 * Math.PI
  while (delta > Math.PI) delta -= 2 * Math.PI
  return delta
}

function clamp (value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value))
}

function routePhaseName (kind, cycle) {
  if (nv_type === 'ABACA_FIXED') {
    return kind === 'outbound' ? `exploration_${cycle}` : `evaluation_${cycle}`
  }
  return kind
}

function phaseIndex (phase) {
  const indices = {
    exploration_1: 1,
    evaluation_1: 2,
    exploration_2: 3,
    evaluation_2: 4,
    outbound: 1,
    middle: 2,
    return: 3
  }
  return indices[phase] || null
}

function vectorAngleDeg (a, b) {
  const normA = Math.sqrt(a.x * a.x + a.z * a.z)
  const normB = Math.sqrt(b.x * b.x + b.z * b.z)
  if (normA === 0 || normB === 0) return 0
  const cos = clamp((a.x * b.x + a.z * b.z) / (normA * normB), -1, 1)
  return Math.acos(cos) * 180 / Math.PI
}

function appendOutboundWaypoint () {
  if (!recordingOutbound) return
  const waypoint = waypointFromPosition(bot.entity.position)
  const last = outboundWaypoints[outboundWaypoints.length - 1]
  if (!last || waypointKey(last) !== waypointKey(waypoint)) {
    outboundWaypoints.push(waypoint)
  }
}

function movementConfig () {
  const defaultMove = new Movements(bot)
  defaultMove.allowSprinting = false
  return defaultMove
}

function goalForPosition (pos, ignoreY = false) {
  if (ignoreY || args.location === 'stronghold' || args.location === 'nether_bastion' || args.location === 'nether_fortress') {
    return new GoalNearXZ(pos.x, pos.z, 0.5)
  }
  return new GoalBlock(pos.x, pos.y, pos.z)
}

function setGoalForPosition (pos, ignoreY = false) {
  bot.pathfinder.setMovements(movementConfig())
  bot.pathfinder.setGoal(goalForPosition(pos, ignoreY), false)
}

function buildReverseWaypoints (startPos) {
  const startWaypoint = waypointFromPosition(startPos)
  const currentWaypoint = waypointFromPosition(bot.entity.position)
  const forward = outboundWaypoints.slice()

  if (forward.length === 0 || waypointKey(forward[0]) !== waypointKey(startWaypoint)) {
    forward.unshift(startWaypoint)
  }
  if (waypointKey(forward[forward.length - 1]) !== waypointKey(currentWaypoint)) {
    forward.push(currentWaypoint)
  }

  const reverse = forward.slice(0, -1).reverse()
  return reverse.filter((waypoint, index) => (
    index === 0 || waypointKey(waypoint) !== waypointKey(reverse[index - 1])
  ))
}

bot.on('physicsTick', appendOutboundWaypoint)

bot.checklook = (yaw, pitch) => {
  if (smoothCamera && smoothCamera.isLookingAt(yaw, pitch)) {
    return 1
  }
  bot.clearControlStates()
  if (!smoothCamera) return 2
  bot.look(yaw, pitch).catch(err => {
    console.error('[smooth-camera] failed to set pathfinder target', err)
  })
  return 2
}

bot.on('sentPosAction', (pos) => {
  pos.route_phase = routePhase
  pos.return_strategy = useSymmetricReturn() ? 'reverse_waypoints' : 'independent_astar'
  pos.waypoint_index = currentWaypointIndex
  pos.return_waypoint_total = returnWaypointTotal
  pos.outbound_waypoint_count = outboundWaypoints.length
  pos.phase_index = phaseIndex(routePhase)
  pos.memory_eval_phase = routePhase === 'evaluation_1' || routePhase === 'evaluation_2'
  pos.abaca_cycle = currentCycle
  pos.abaca_min_angle = nv_type === 'ABACA_FIXED' ? abacaMinAngleDeg : null
  pos.target_label = currentTargetLabel
  if (currentTargetPosition) {
    pos.target_x = currentTargetPosition.x
    pos.target_y = currentTargetPosition.y
    pos.target_z = currentTargetPosition.z
  }
})

bot.on('physicsTick', () => {
  if (!bot.entity) return
  const yaw = bot.entity.yaw
  const pitch = bot.entity.pitch
  const cameraYaw = lastEmittedYaw === null ? 0 : angleDelta(yaw, lastEmittedYaw)
  const cameraPitch = lastEmittedPitch === null ? 0 : pitch - lastEmittedPitch
  lastEmittedYaw = yaw
  lastEmittedPitch = pitch
  bot.emit('sentPosAction', {
    x: bot.entity.position.x,
    y: bot.entity.position.y,
    z: bot.entity.position.z,
    yaw,
    pitch,
    action: {
      forward: bot.getControlState('forward'),
      back: bot.getControlState('back'),
      left: bot.getControlState('left'),
      right: bot.getControlState('right'),
      jump: bot.getControlState('jump'),
      sprint: bot.getControlState('sprint'),
      sneak: bot.getControlState('sneak'),
      camera: [cameraYaw, cameraPitch]
    },
    goal: bot.pathfinder.goal
  })
})

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
    if (block === null) {
      return null
    }
    if (block.type === bot.registry.blocksByName.air.id) {
      return pos.offset(0, dy-1, 0)
    }
  }
  while (block.type === bot.registry.blocksByName.air.id) {
    dy--
    block = bot.blockAt(pos.offset(0, dy, 0))
    if (block === null) {
      return null
    }
    if (block.type !== bot.registry.blocksByName.air.id) {
      return pos.offset(0, dy, 0)
    }
  }
}

function blockId (name) {
  return bot.registry.blocksByName[name] ? bot.registry.blocksByName[name].id : null
}

function landableBlocks () {
  return new Set([
    'dirt',
    'stone',
    'grass_path',
    'dirt_path',
    'sand',
    'grass_block',
    'snow',
    'snow_block',
    'gravel',
    'sandstone',
    'red_sand',
    'terracotta',
    'mycelium',
    'end_stone',
    'nether_bricks',
    'blackstone',
    'polished_blackstone_bricks',
    'cracked_polished_blackstone_bricks',
    'netherrack'
  ].map(blockId).filter(id => id !== null))
}

/*
  [-range,range] 区域内的一个(x,z)，该位置的xz坐标上最高的非空气方块
  *要求并且必须是泥土或者石头（放置goal在屋顶上的奇怪情况）
  *要求距离tp点不超过80格, 并且距离起始点至少20格
*/

function random_pos (range, acceptPosition = null){
  const start_pos = bot.entity.position.clone()
  const referenceY = routePhase === 'initial_move' ? tp_target.y : start_pos.y
  const landable = landableBlocks()
  const rejected = {
    radial: 0,
    villageBoundary: 0,
    unloaded: 0,
    height: 0,
    surface: 0,
    surfaceTypes: {},
    custom: 0
  }
  for (let attempt = 0; attempt < randomPosMaxAttempts; attempt++) {
    const x = Math.floor(Math.random() * range * 2) - range
    const z = Math.floor(Math.random() * range * 2) - range
    let limit = range*4/5
    if (x*x + z*z < limit*limit) {
      // ensure the distance is not to short
      rejected.radial++
      continue
    }
    // ensure the distance is not to far away from village center
    const dx = start_pos.x + x - tp_target.x
    const dz = start_pos.z + z - tp_target.z
    if (dx*dx + dz*dz > 80*80) {
      rejected.villageBoundary++
      continue
    }
    if (args.location === 'stronghold' || args.location === 'nether_bastion' || args.location === 'nether_fortress') {
      const netherEndPos = new Vec3(start_pos.x+x, start_pos.y, start_pos.z+z)
      if (acceptPosition && !acceptPosition(netherEndPos, start_pos)) {
        rejected.custom++
        continue
      }
      return netherEndPos
    }
    const pos = land_pos(start_pos.x+x, start_pos.z+z)
    if (pos === null) {
      rejected.unloaded++
      continue
    }
    if (Math.abs(pos.y - referenceY) > 10) {
      rejected.height++
      continue
    }
    const block = bot.blockAt(pos)
    const blockunder = bot.blockAt(pos.offset(0, -1, 0))
    if (landable.has(block.type) && landable.has(blockunder.type)){
      pos.y = pos.y + 1
      if (acceptPosition && !acceptPosition(pos, start_pos)) {
        rejected.custom++
        continue
      }
      return pos
    }
    else {
      rejected.surface++
      const surfaceKey = `${block.name}/${blockunder.name}`
      rejected.surfaceTypes[surfaceKey] = (rejected.surfaceTypes[surfaceKey] || 0) + 1
    }
  }
  throw new Error(
    `failed to sample random_pos range=${range} attempts=${randomPosMaxAttempts} ` +
    `start=${start_pos.x.toFixed(1)},${start_pos.y.toFixed(1)},${start_pos.z.toFixed(1)} ` +
    `referenceY=${referenceY} rejected=${JSON.stringify(rejected)}`
  )
}
/*
  move to a random position in a range*range cube around the bot
*/
function move (range, acceptPosition = null) {
  const pos = random_pos(range, acceptPosition)
  console.log('moving to', pos)
  console.log('distance', bot.entity.position.distanceTo(pos))
  setGoalForPosition(pos)
  return pos
}

function* moveOutAndReverseBack(range, cycle, startPos, targetLabel, acceptPosition = null){
  routePhase = routePhaseName('outbound', cycle)
  currentCycle = cycle
  currentTargetLabel = targetLabel
  currentWaypointIndex = null
  currentTargetPosition = null
  returnWaypointTotal = 0
  outboundWaypoints = []
  recordingOutbound = true
  appendOutboundWaypoint()
  const target = move(range, acceptPosition)
  currentTargetPosition = target
  yield

  appendOutboundWaypoint()
  recordingOutbound = false
  const returnWaypoints = buildReverseWaypoints(startPos)
  returnWaypointTotal = returnWaypoints.length
  console.log('reverse waypoint count', returnWaypointTotal)
  if (returnWaypointTotal === 0) {
    console.log('empty reverse waypoint path, quit')
    bot.emit('abnormal_exit')
    process.exit(1)
  }

  routePhase = routePhaseName('return', cycle)
  for (let i = 0; i < returnWaypoints.length; i++) {
    currentWaypointIndex = i
    setGoalForPosition(returnWaypoints[i], true)
    yield
  }
  return target
}

function* moveAround(range){
  const start_pos = bot.entity.position.clone()
  if (nv_type === 'ABACA_FIXED') {
    console.log('using ABACA reverse waypoint return')
    const firstStart = start_pos.clone()
    const targetB = yield* moveOutAndReverseBack(range, 1, firstStart, 'B')
    const secondStart = bot.entity.position.clone()
    const firstVector = {
      x: targetB.x - firstStart.x,
      z: targetB.z - firstStart.z
    }
    const acceptDifferentDirection = (candidate, origin) => {
      const candidateVector = {
        x: candidate.x - origin.x,
        z: candidate.z - origin.z
      }
      const angle = vectorAngleDeg(firstVector, candidateVector)
      if (angle < abacaMinAngleDeg) {
        console.log('rej ABACA target angle', angle.toFixed(2))
        return false
      }
      return true
    }
    yield* moveOutAndReverseBack(range, 2, secondStart, 'C', acceptDifferentDirection)
    currentCycle = null
    currentTargetLabel = null
    currentTargetPosition = null
    finishTask()
    return
  }

  if (useSymmetricReturn()) {
    console.log('using reverse waypoint return')
    yield* moveOutAndReverseBack(range, null, start_pos, 'B')
    finishTask()
    return
  }

  routePhase = 'outbound'
  move(range)
  yield 
  if (nv_type === 'ABCA') {
    routePhase = 'middle'
    move(range)
    yield
  }
  console.log('moving back to', start_pos)
  routePhase = 'return'
  bot.pathfinder.setMovements(movementConfig())
  bot.pathfinder.setGoal(new GoalNearXZ(start_pos.x, start_pos.z, 1), false)
  yield
  finishTask()
}
async function turnTo (yaw, pitch) {
  while (bot.checklook(yaw, pitch) !== 1) {
    await sleep(50)
  }
}

async function lookaround () {
  routePhase = 'lookaround'
  let pitch = 0
  await turnTo(0, pitch)
  await turnTo(Math.PI / 2, pitch)
  await turnTo(Math.PI, pitch)
  await turnTo(Math.PI * 3 / 2, pitch)
  await turnTo(0, pitch)
  let controller = moveAround(args.nvrange)
  controller.next()
  bot.on('goal_reached', () => {
    console.log('goal reached')
    controller.next()
  })
}
bot.once('spawn', () => {
  smoothCamera = installSmoothCamera(bot, { maxStep: 0.1, tolerance: 0.02 })
  // Collection bots must not die and respawn at the world spawn while a
  // trajectory is being recorded; that creates a large position jump.
  bot.chat(`/gamemode creative ${args.name}`)
  if (args.location === 'nether_bastion' || args.location === 'nether_fortress') {
    bot.chat(`/execute in minecraft:the_nether run tp ${args.name} ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  }
  else if (args.location === 'end_city' || args.location === 'end_mainland') {
    bot.chat(`/execute in minecraft:the_end run tp ${args.name} ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  }
  else {
    bot.chat(`/execute in minecraft:overworld run tp ${args.name} ${tp_target.x} ${tp_target.y} ${tp_target.z}`)
  }
  // wait for 1 seconds to let the bot tp
  setTimeout(async () => {
    try {
      await waitForReadyChunks()
	    // First move to a random position with inital range
	    routePhase = 'initial_move'
	    move(20)
    } catch (err) {
      console.error(err)
      bot.emit('abnormal_exit')
      process.exit(1)
    }
    
    // Track if we've started the viewer
    let viewerStarted = false;
    
    // Handle goal_reached events
	    bot.on('goal_reached', () => {
	      stuck_count = 0
	      if (!viewerStarted) {
        // This is the first goal_reached (initial move)
        console.log('Reached initial position, starting viewer and lookaround')
        // Start the viewer after reaching the position
        mineflayerViewerhl(bot, { output: `127.0.0.1:${args.port}`, frames: -1, width: 640, height: 360 })
        viewerStarted = true;
        
        // Start lookaround after viewer is ready
	        lookaround().catch((err) => {
	          console.error(err)
	          bot.emit('abnormal_exit')
	          process.exit(1)
	        })
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
    
	    let stuck_count = 0
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
