const mineflayer = require('mineflayer')
const minimist = require('minimist')
const mineflayerViewerhl = require('prismarine-viewer').headless
const { get_target } = require('./target.js')
const Vec3 = require('vec3').Vec3

const args = minimist(process.argv.slice(2))
const tpTarget = get_target(args.location)
const turnOrder = args['turn-order'] || args.turn_order || 'right_left'
const turnAngleDeg = Number(args['turn-angle'] || args.turn_angle || 90)
const spawnRange = Number(args['spawn-range'] || args.spawn_range || 20)
const viewerPort = Number(args.port)
const mcPort = Number(args['mc-port'] || args.mc_port || 8105)
const username = args.name || 'Bot'

if (tpTarget === null) {
  console.log('Invalid location')
  process.exit(1)
}

if (!['right_left', 'left_right'].includes(turnOrder)) {
  console.log('Invalid turn order:', turnOrder)
  process.exit(1)
}

if (!(turnAngleDeg >= 45 && turnAngleDeg <= 180)) {
  console.log('Invalid turn angle:', turnAngleDeg)
  process.exit(1)
}

const bot = mineflayer.createBot({
  username,
  host: 'localhost',
  port: mcPort
})

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))
let trunInitialYaw = null

function yawForTp (yawRad) {
  return yawRad * 180 / Math.PI
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
    'mycelium'
  ].map(blockId).filter(id => id !== null))
}

function surfacePos (x, z) {
  const landable = landableBlocks()
  for (let y = 255; y >= 1; y--) {
    const below = bot.blockAt(new Vec3(x, y, z))
    const here = bot.blockAt(new Vec3(x, y + 1, z))
    const head = bot.blockAt(new Vec3(x, y + 2, z))
    if (below === null || here === null || head === null) return null
    if (
      landable.has(below.type) &&
      here.type === bot.registry.blocksByName.air.id &&
      head.type === bot.registry.blocksByName.air.id
    ) {
      return new Vec3(x, y + 1, z)
    }
  }
  return null
}

function randomSpawnPos (range) {
  const center = bot.entity.position.clone()
  const minDistance = range * 4 / 5

  for (let attempt = 0; attempt < 2000; attempt++) {
    const dx = Math.floor(Math.random() * range * 2) - range
    const dz = Math.floor(Math.random() * range * 2) - range
    if (dx * dx + dz * dz < minDistance * minDistance) continue

    const x = Math.floor(center.x + dx)
    const z = Math.floor(center.z + dz)
    const centerDx = x - tpTarget.x
    const centerDz = z - tpTarget.z
    if (centerDx * centerDx + centerDz * centerDz > 80 * 80) continue

    const pos = surfacePos(x, z)
    if (pos === null) continue
    if (Math.abs(pos.y - center.y) > 10) continue
    return pos
  }

  const fallback = surfacePos(Math.floor(tpTarget.x), Math.floor(tpTarget.z))
  if (fallback !== null) return fallback
  return new Vec3(tpTarget.x, tpTarget.y, tpTarget.z)
}

function emptyAction (cameraYaw = 0, cameraPitch = 0) {
  return {
    forward: false,
    back: false,
    left: false,
    right: false,
    jump: false,
    sprint: false,
    sneak: false,
    camera: [cameraYaw, cameraPitch]
  }
}

function emitFrame (position, yaw, pitch, cameraYaw = 0, cameraPitch = 0) {
  bot.entity.yaw = yaw
  bot.entity.pitch = pitch
  bot.emit('sentPosAction', {
    x: position.x,
    y: position.y,
    z: position.z,
    yaw,
    pitch,
    action: emptyAction(cameraYaw, cameraPitch),
    goal: null,
    initial_yaw: trunInitialYaw
  })
}

async function emitHoldFrames (position, yaw, pitch, count) {
  for (let i = 0; i < count; i++) {
    emitFrame(position, yaw, pitch, 0, 0)
    await sleep(50)
  }
}

async function emitTurnFrames (position, startYaw, endYaw, pitch) {
  const maxStep = 0.1
  const totalDelta = endYaw - startYaw
  const steps = Math.max(1, Math.ceil(Math.abs(totalDelta) / maxStep))
  let lastYaw = startYaw
  for (let i = 1; i <= steps; i++) {
    const yaw = startYaw + totalDelta * i / steps
    const cameraYaw = yaw - lastYaw
    emitFrame(position, yaw, pitch, cameraYaw, 0)
    lastYaw = yaw
    await sleep(50)
  }
}

async function collectTrun () {
  console.log('collecting TURN in', args.location, tpTarget, turnOrder, turnAngleDeg)
  bot.chat(`/execute in minecraft:overworld run tp ${bot.username} ${tpTarget.x} ${tpTarget.y} ${tpTarget.z}`)
  await sleep(1500)

  const spawnPos = randomSpawnPos(spawnRange)
  const initialYaw = Math.random() * Math.PI * 2 - Math.PI
  const pitch = 0
  bot.chat(`/execute in minecraft:overworld run tp ${bot.username} ${spawnPos.x} ${spawnPos.y} ${spawnPos.z} ${yawForTp(initialYaw)} 0`)
  await sleep(1500)

  mineflayerViewerhl(bot, { output: `127.0.0.1:${viewerPort}`, frames: -1, width: 640, height: 360 })
  await sleep(1200)

  const signedAngle = turnAngleDeg * Math.PI / 180
  const targetYaw = turnOrder === 'right_left'
    ? initialYaw + signedAngle
    : initialYaw - signedAngle
  trunInitialYaw = initialYaw

  const recordPos = bot.entity.position.clone()
  await emitHoldFrames(recordPos, initialYaw, pitch, 20)
  await emitTurnFrames(recordPos, initialYaw, targetYaw, pitch)
  await emitHoldFrames(recordPos, targetYaw, pitch, 1)
  await emitTurnFrames(recordPos, targetYaw, initialYaw, pitch)
  await emitHoldFrames(recordPos, initialYaw, pitch, 3)

  bot.emit('endtask')
  await sleep(500)
  process.exit(0)
}

bot.once('spawn', () => {
  collectTrun().catch((err) => {
    console.error(err)
    if (bot.viewer) bot.viewer.close()
    process.exit(1)
  })
})
