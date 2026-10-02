'use strict'
// The harness for one Paper server: one bot per agent, driven by JSON lines on stdin, answered on stdout.
//
//   {"id": 1, "op": "connect", "host": "127.0.0.1", "port": 25565, "team": ["ada", "ben", "cy", "dee"],
//    "version": "1.21.11"}                           the game version the server runs
//   {"id": 2, "op": "observe", "bot": "ada"}         what ada sees, her messages, her last action's result; asked
//                                                     again before the next thaw, it answers the same
//   {"id": 3, "op": "act", "bot": "ada", "action": {"name": "mine", "x": 1, "y": -58, "z": 4}}
//   {"id": 4, "op": "busy"}                           which bots are still acting, and which were hurt since the thaw
//   {"id": 4, "op": "unloaded"}                       which bots do not yet hold the chunks around them
//   {"id": 5, "op": "freeze"}                         stop every action (results are kept) and pause physics
//   {"id": 6, "op": "thaw"}                           resume physics, before actions start; the messages and deaths
//                                                     that observations have told are dropped
//   {"id": 7, "op": "quit"}
//
// Messages reach an agent only from its teammates: system messages (someone joining, deaths, server notices) and
// players outside the team never do, whatever they say. This is by construction, not by asking the model to ignore
// them.

const readline = require('node:readline')
const { Vec3 } = require('vec3')
const mineflayer = require('mineflayer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const { observe } = require('./lib/observe')
const { ACTIONS, ActionError, Interrupted } = require('./lib/actions')
const { fixMaterials } = require('./lib/data')
const { LIMITS } = require('./lib/limits')
const { News } = require('./lib/news')

const MAX_MESSAGES = 20
const MAX_WALK_DIG_MS = LIMITS.walk_dig_seconds * 1000
const RESTFUL = new Set(['wait', 'idle', 'chat'])

const bots = new Map() // name → { bot, memory, air, news, action, result }
let team = new Set()

async function connect ({ host, port, team: names, version }) {
  if (!version) throw new Error('connect needs the game version the server runs')
  team = new Set(names)
  await Promise.all(names.map(name => join(host, port, name, version)))
  return { connected: names }
}

function join (host, port, name, version) {
  return new Promise((resolve, reject) => {
    const bot = mineflayer.createBot({ host, port, username: name, version, auth: 'offline', hideErrors: true })
    const state = { bot, memory: new Map(), air: new Set(), news: new News(MAX_MESSAGES), action: null, result: null, kicked: null, health: null, hurt: false }
    bots.set(name, state)
    bot.loadPlugin(pathfinder)
    bot.once('spawn', () => {
      // The server makes a player's box from a 32-bit half width (0.30000001...), a little wider than mineflayer's
      // 0.3: a bot resting against a wall would be inside it by the server's reckoning, and Paper refuses such
      // moves ("clipped into block"), putting the bot back every tick.
      bot.physics.playerHalfWidth = Math.fround(0.3)
      fixMaterials(bot.registry)
      const movements = new Movements(bot)
      movements.canDig = true
      movements.allowParkour = false
      movements.allowSprinting = true
      // What walking may put under the bot or ahead of it when there is no other way (the `move` tools say so).
      movements.scafoldingBlocks = LIMITS.scaffolding.map(item => bot.registry.itemsByName[item]?.id).filter(id => id !== undefined)
      // Walking digs only what `mine` would: not what takes the bot's tools too long (stone by hand is 7.5 seconds a
      // block), and not what would drop nothing (ore under too weak a pickaxe is destroyed).
      const slow = new Map() // block type → whether walking may not dig it, with the tools held now
      bot.inventory.on('updateSlot', () => slow.clear())
      movements.exclusionAreasBreak.push((block) => {
        if (!slow.has(block.type)) {
          const tool = bot.pathfinder.bestHarvestTool(block)?.type ?? null
          slow.set(block.type, !block.canHarvest(tool) || block.digTime(tool, false, false, false, [], []) > MAX_WALK_DIG_MS)
        }
        return slow.get(block.type) ? 100 : 0
      })
      bot.pathfinder.setMovements(movements)
      bot.pathfinder.thinkTimeout = 2000
      resolve()
    })
    bot.on('chat', (username, message) => {
      if (username === name || !team.has(username)) return // teammates only
      state.news.hear(username, message)
    })
    bot.on('kicked', reason => { state.kicked = String(typeof reason === 'string' ? reason : JSON.stringify(reason)) })
    bot.on('end', reason => { state.ended = String(reason ?? 'the connection closed') })
    bot.on('error', error => { if (!bot.entity) reject(error) })
    bot.on('death', () => { state.news.died = true })
    bot.on('health', () => { // (also fires when only food changes)
      if (state.health !== null && bot.health < state.health) state.hurt = true
      state.health = bot.health
    })
    bot.on('respawn', () => { state.memory.clear(); state.air.clear() }) // another dimension (or a new life): what was seen is elsewhere
  })
}

// A bot the server has dropped has nothing true to report and can do nothing: say so, so that the episode ends.
function connected (name, state) {
  const reason = state.kicked ?? state.ended
  if (reason) throw new Error(`${name} is no longer connected: ${reason}`)
}

function observation (name) {
  const state = bots.get(name)
  if (!state) throw new Error(`no bot ${name}`)
  connected(name, state)
  const result = observe(state.bot, team, state.memory, state.air)
  const { messages, died } = state.news.tell()
  result.messages = messages
  result.last_action = state.result
  result.died = died
  return result
}

function act (name, action) {
  const state = bots.get(name)
  if (!state) throw new Error(`no bot ${name}`)
  connected(name, state)
  if (state.action) throw new Error(`${name} is still acting`)
  const handler = ACTIONS[action?.name]
  if (!handler) {
    state.result = { action, ok: false, error: `unknown action ${action?.name}; choose one of ${Object.keys(ACTIONS).filter(known => known !== 'idle').join(', ')}` }
    return { started: false }
  }
  const controller = new AbortController()
  const { name: actionName, ...args } = action
  const context = { signal: controller.signal, memory: state.memory }
  state.result = null
  const running = { controller, action, done: null }
  const finish = result => { // an action abandoned at a freeze reports nothing when it finally ends
    if (state.action !== running) return
    state.result = result
    state.action = null
  }
  running.done = (async () => {
    const started = Date.now()
    try {
      if (state.bot.isSleeping && !RESTFUL.has(actionName)) await state.bot.wake() // any deed gets a sleeper up
      finish({ action, ok: true, ...(await handler(state.bot, args, context)), seconds: (Date.now() - started) / 1000 })
    } catch (error) {
      if (error instanceof Interrupted) {
        finish(interrupted(action, state.bot))
      } else if (error instanceof ActionError) {
        finish({ action, ok: false, error: error.message })
      } else {
        finish({ action, ok: false, error: `${actionName} failed: ${error.message}` })
      }
    }
  })()
  state.action = running
  return { started: true }
}

async function freeze () {
  const pending = []
  for (const state of bots.values()) {
    if (state.action) {
      state.action.controller.abort()
      pending.push(state.action.done)
    }
  }
  await Promise.race([Promise.all(pending), new Promise(resolve => setTimeout(resolve, 3000))])
  for (const state of bots.values()) {
    if (state.action) { // it did not stop (a path waiting to reach its next block, say): leave it behind
      state.result = interrupted(state.action.action, state.bot)
      state.action = null
    }
    state.bot.pathfinder.setGoal(null)
    state.bot.stopDigging?.()
    state.bot.clearControlStates()
    state.bot.physicsEnabled = false // nothing moves while the agents think
  }
  return { frozen: true }
}

// How an action cut off by the freeze is reported: where the bot got to, so that the agent can decide how to go on
// (a relative move repeated as it was would go the whole distance again).
function interrupted (action, bot) {
  const here = bot.entity?.position.floored()
  return { action, ok: false, interrupted: true, now_at: here ? { x: here.x, y: here.y, z: here.z } : null }
}

function thaw () {
  for (const state of bots.values()) {
    state.bot.physicsEnabled = true
    state.hurt = false
    state.news.forget()
  }
  return { thawed: true }
}

// The bots that do not yet hold the chunks around them (five by five: all that sight reaches). An observation taken
// before they arrive shows a world with holes in it.
function unloaded () {
  const waiting = []
  for (const [name, state] of bots) {
    connected(name, state)
    const position = state.bot.entity?.position
    let missing = !position
    for (let dx = -2; dx <= 2 && !missing; dx++) {
      for (let dz = -2; dz <= 2 && !missing; dz++) {
        missing = !state.bot.world.getColumnAt(new Vec3((Math.floor(position.x) >> 4 << 4) + 16 * dx, 0, (Math.floor(position.z) >> 4 << 4) + 16 * dz))
      }
    }
    if (missing) waiting.push(name)
  }
  return { unloaded: waiting }
}

// Who is still acting, and who has been hurt since the thaw.
function busy () {
  const names = filter => [...bots.entries()].filter(([, state]) => filter(state)).map(([name]) => name)
  return { acting: names(state => state.action), hurt: names(state => state.hurt) }
}

async function handle (request) {
  switch (request.op) {
    case 'connect': return connect(request)
    case 'observe': return observation(request.bot)
    case 'act': return act(request.bot, request.action)
    case 'busy': return busy()
    case 'unloaded': return unloaded()
    case 'freeze': return freeze()
    case 'thaw': return thaw()
    case 'quit':
      for (const state of bots.values()) state.bot.quit()
      setTimeout(() => process.exit(0), 300)
      return { quit: true }
    default: throw new Error(`unknown op ${request.op}`)
  }
}

const input = readline.createInterface({ input: process.stdin })
input.on('line', async line => {
  let request
  try { request = JSON.parse(line) } catch { return }
  let reply
  try {
    reply = { id: request.id, result: await handle(request) }
  } catch (error) {
    reply = { id: request.id, error: error.message }
  }
  process.stdout.write(JSON.stringify(reply) + '\n')
})
input.on('close', () => process.exit(0))
process.stdout.write(JSON.stringify({ ready: true }) + '\n')
