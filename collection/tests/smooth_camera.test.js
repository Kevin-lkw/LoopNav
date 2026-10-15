const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const test = require('node:test')

const { angleDelta, installSmoothCamera } = require('../smooth_camera')

class FakeBot extends EventEmitter {
  constructor () {
    super()
    this.entity = { yaw: 0, pitch: 0 }
    this.poses = [{ yaw: 0, pitch: 0 }]
    this.look = async (yaw, pitch) => {
      this.entity.yaw = yaw
      this.entity.pitch = pitch
      this.poses.push({ yaw, pitch })
    }
  }
}

test('smooth camera limits every rendered pose step', async () => {
  const bot = new FakeBot()
  installSmoothCamera(bot, { maxStep: 0.1 })

  const completed = bot.look(Math.PI, -1.2, true)
  for (let index = 0; index < 40; index++) {
    bot.emit('physicsTick')
  }
  await completed

  for (let index = 1; index < bot.poses.length; index++) {
    const previous = bot.poses[index - 1]
    const current = bot.poses[index]
    assert.ok(Math.abs(angleDelta(current.yaw, previous.yaw)) <= 0.100001)
    assert.ok(Math.abs(current.pitch - previous.pitch) <= 0.100001)
  }
  assert.ok(Math.abs(angleDelta(bot.entity.yaw, Math.PI)) < 1e-6)
  assert.ok(Math.abs(bot.entity.pitch + 1.2) < 1e-6)
})

test('two 20 FPS camera steps stay below the 30 degree audit threshold', () => {
  const bot = new FakeBot()
  installSmoothCamera(bot, { maxStep: 0.1 })

  bot.look(Math.PI, 0)
  bot.emit('physicsTick')
  const firstYaw = bot.entity.yaw
  bot.emit('physicsTick')
  const secondYaw = bot.entity.yaw

  const nativeDeltaDegrees = Math.abs(angleDelta(secondYaw, 0)) * 180 / Math.PI
  assert.ok(nativeDeltaDegrees < 30)
  assert.ok(Math.abs(angleDelta(secondYaw, firstYaw)) <= 0.100001)
})

test('a new target supersedes an in-flight target without a jump', async () => {
  const bot = new FakeBot()
  installSmoothCamera(bot, { maxStep: 0.1 })

  const first = bot.look(Math.PI, 0)
  bot.emit('physicsTick')
  const beforeRetarget = bot.entity.yaw
  const second = bot.look(-Math.PI / 2, 0)
  await first
  bot.emit('physicsTick')

  assert.ok(Math.abs(angleDelta(bot.entity.yaw, beforeRetarget)) <= 0.100001)
  for (let index = 0; index < 20; index++) {
    bot.emit('physicsTick')
  }
  await second
  assert.ok(Math.abs(angleDelta(bot.entity.yaw, -Math.PI / 2)) < 1e-6)
})
