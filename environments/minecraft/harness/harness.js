'use strict'
// The harness for one Paper server: one bot per agent, driven by JSON lines on stdin, answered on stdout.
//
//   {"id": 1, "op": "connect", "host": "127.0.0.1", "port": 25565, "team": ["ada", "ben", "cy", "dee"]}
//   {"id": 2, "op": "observe", "bot": "ada"}         what ada sees, her messages, her last action's result
//   {"id": 3, "op": "act", "bot": "ada", "action": {"name": "mine", "x": 1, "y": -58, "z": 4}}
//   {"id": 4, "op": "busy"}                           which bots are still acting
//   {"id": 5, "op": "freeze"}                         stop every action (results are kept) and pause physics
//   {"id": 6, "op": "thaw"}                           resume physics, before actions start
//   {"id": 7, "op": "quit"}
//
// Messages reach an agent only from its teammates: system messages (someone joining, deaths, server notices) and
// players outside the team never do, whatever they say. This is by construction, not by asking the model to ignore
// them.

const readline = require('node:readline')
const mineflayer = require('mineflayer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const { observe } = require('./lib/observe')
const { ACTIONS, ActionError, Interrupted } = require('./lib/actions')

const VERSION = '1.21.11'
const MAX_MESSAGES = 20

const bots = new Map() // name → { bot, memory, inbox, action, result }
let team = new Set()

async function connect ({ host, port, team: names }) {
  team = new Set(names)
  await Promise.all(names.map(name => join(host, port, name)))
  return { connected: names }
}

function join (host, port, name) {
  return new Promise((resolve, reject) => {
    const bot = mineflayer.createBot({ host, port, username: name, version: VERSION, auth: 'offline', hideErrors: true })
    const state = { bot, memory: new Map(), inbox: [], action: null, result: null, kicked: null }
    bots.set(name, state)
    bot.loadPlugin(pathfinder)
    bot.once('spawn', () => {
      const movements = new Movements(bot)
      movements.canDig = true
      movements.allowParkour = false
      movements.allowSprinting = true
      bot.pathfinder.setMovements(movements)
      bot.pathfinder.thinkTimeout = 2000
      resolve()
    })
    bot.on('chat', (username, message) => {
      if (username === name || !team.has(username)) return // teammates only
      state.inbox.push({ from: username, message })
      if (state.inbox.length > MAX_MESSAGES) state.inbox.shift()
    })
    bot.on('kicked', reason => { state.kicked = String(typeof reason === 'string' ? reason : JSON.stringify(reason)) })
    bot.on('error', error => { if (!bot.entity) reject(error) })
    bot.on('death', () => { state.died = true })
  })
}

function observation (name) {
  const state = bots.get(name)
  if (!state) throw new Error(`no bot ${name}`)
  const result = observe(state.bot, team, state.memory)
  result.messages = state.inbox.splice(0)
  result.last_action = state.result
  result.died = Boolean(state.died)
  state.died = false
  return result
}

function act (name, action) {
  const state = bots.get(name)
  if (!state) throw new Error(`no bot ${name}`)
  if (state.action) throw new Error(`${name} is still acting`)
  const handler = ACTIONS[action?.name]
  if (!handler) {
    state.result = { action, ok: false, error: `unknown action ${action?.name}; choose one of ${Object.keys(ACTIONS).join(', ')}` }
    return { started: false }
  }
  const controller = new AbortController()
  const { name: actionName, ...args } = action
  const context = { signal: controller.signal, memory: state.memory, team }
  state.result = null
  state.action = {
    controller,
    done: (async () => {
      try {
        state.result = { action, ok: true, ...(await handler(state.bot, args, context)) }
      } catch (error) {
        if (error instanceof Interrupted || controller.signal.aborted) {
          state.result = { action, ok: false, interrupted: true, note: 'time ran out before it finished; repeat it to continue' }
        } else if (error instanceof ActionError) {
          state.result = { action, ok: false, error: error.message }
        } else {
          state.result = { action, ok: false, error: `${actionName} failed: ${error.message}` }
        }
      } finally {
        state.action = null
      }
    })()
  }
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
    state.bot.pathfinder.stop()
    state.bot.stopDigging?.()
    state.bot.clearControlStates()
    state.bot.physicsEnabled = false // nothing moves while the agents think
  }
  return { frozen: true }
}

function thaw () {
  for (const state of bots.values()) state.bot.physicsEnabled = true
  return { thawed: true }
}

function busy () {
  return { acting: [...bots.entries()].filter(([, state]) => state.action).map(([name]) => name) }
}

async function handle (request) {
  switch (request.op) {
    case 'connect': return connect(request)
    case 'observe': return observation(request.bot)
    case 'act': return act(request.bot, request.action)
    case 'busy': return busy()
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
