const DEFAULT_MAX_STEP = 0.1
const DEFAULT_TOLERANCE = 1e-3

function angleDelta (target, current) {
  let delta = target - current
  while (delta <= -Math.PI) delta += 2 * Math.PI
  while (delta > Math.PI) delta -= 2 * Math.PI
  return delta
}

function clamp (value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value))
}

function installSmoothCamera (
  bot,
  {
    maxStep = DEFAULT_MAX_STEP,
    tolerance = DEFAULT_TOLERANCE
  } = {}
) {
  if (!Number.isFinite(maxStep) || maxStep <= 0) {
    throw new TypeError(`maxStep must be positive and finite, got ${maxStep}`)
  }
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new TypeError(`tolerance must be non-negative and finite, got ${tolerance}`)
  }

  const instantLook = bot.look.bind(bot)
  let activeRequest = null

  function finishActiveRequest () {
    if (activeRequest === null) return
    const { resolve } = activeRequest
    activeRequest = null
    resolve()
  }

  function sameTarget (yaw, pitch) {
    if (activeRequest === null) return false
    return (
      Math.abs(angleDelta(yaw, activeRequest.yaw)) <= tolerance &&
      Math.abs(pitch - activeRequest.pitch) <= tolerance
    )
  }

  bot.look = (yaw, pitch) => {
    if (!Number.isFinite(yaw) || !Number.isFinite(pitch)) {
      return Promise.reject(new TypeError(`Invalid camera target yaw=${yaw} pitch=${pitch}`))
    }
    if (sameTarget(yaw, pitch)) {
      return activeRequest.promise
    }

    // Mineflayer completes an older look request when a new request supersedes it.
    finishActiveRequest()

    if (
      bot.entity &&
      Math.abs(angleDelta(yaw, bot.entity.yaw)) <= tolerance &&
      Math.abs(pitch - bot.entity.pitch) <= tolerance
    ) {
      return Promise.resolve(instantLook(yaw, pitch, true))
    }

    let finish
    const promise = new Promise(resolve => {
      finish = resolve
    })
    activeRequest = { yaw, pitch, promise, resolve: finish }
    return promise
  }

  bot.on('physicsTick', () => {
    if (activeRequest === null || !bot.entity) return

    const yawDelta = angleDelta(activeRequest.yaw, bot.entity.yaw)
    const pitchDelta = activeRequest.pitch - bot.entity.pitch
    const isFinalStep = (
      Math.abs(yawDelta) <= maxStep &&
      Math.abs(pitchDelta) <= maxStep
    )
    const nextYaw = isFinalStep
      ? activeRequest.yaw
      : bot.entity.yaw + clamp(yawDelta, -maxStep, maxStep)
    const nextPitch = isFinalStep
      ? activeRequest.pitch
      : bot.entity.pitch + clamp(pitchDelta, -maxStep, maxStep)

    const result = instantLook(nextYaw, nextPitch, true)
    if (result && typeof result.catch === 'function') {
      result.catch(err => {
        console.error('[smooth-camera] look failed', err)
      })
    }
    if (isFinalStep) finishActiveRequest()
  })

  return {
    angleDelta,
    isLookingAt (yaw, pitch) {
      return (
        bot.entity &&
        Math.abs(angleDelta(yaw, bot.entity.yaw)) <= tolerance &&
        Math.abs(pitch - bot.entity.pitch) <= tolerance
      )
    }
  }
}

module.exports = {
  DEFAULT_MAX_STEP,
  angleDelta,
  installSmoothCamera
}
