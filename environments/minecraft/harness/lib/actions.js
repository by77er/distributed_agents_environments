'use strict'
// The actions agents choose from: a small vocabulary of skills over mineflayer.
//
// Every target must be one the agent could know: a block to mine must be the first thing a ray from the eyes hits,
// within reach; a place to walk to must be near a block the bot has seen. Unfinished actions are stopped when the
// world freezes (`signal` aborts) and report how far they got; the agent re-issues them to continue.

const { Vec3 } = require('vec3')
const { goals } = require('mineflayer-pathfinder')
const { look, lineOfSight, firstHit, eyes, key, DIRECTIONS } = require('./observe')

const REACH = 4.5
const MAX_TUNNEL = 16

class ActionError extends Error {}

// Each action: (bot, args, context) → result object. context: { signal, memory, team, harness }.
const ACTIONS = {
  async move_to (bot, { x, y, z }, context) {
    const target = new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z'))
    if (!known(context.memory, target) && target.distanceTo(bot.entity.position) > 6) {
      throw new ActionError(`(${x}, ${y}, ${z}) is not near anything you have seen; move toward what you can see first`)
    }
    await travel(bot, new goals.GoalNear(target.x, target.y, target.z, 1), context)
    return { arrived_at: position(bot) }
  },

  async move (bot, { direction, blocks }, context) {
    const vector = direction_(direction)
    const count = Math.max(1, Math.min(int(blocks ?? 8, 'blocks'), 32))
    const start = bot.entity.position.floored()
    const target = start.plus(vector.scaled(count))
    const goal = vector.y === 0 ? new goals.GoalXZ(target.x, target.z) : new goals.GoalY(target.y)
    await travel(bot, goal, context)
    return { arrived_at: position(bot), moved: round(bot.entity.position.distanceTo(start)) }
  },

  async mine (bot, { x, y, z }, context) {
    const block = visibleInReach(bot, new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z')))
    await equipBestTool(bot, block)
    if (!block.canHarvest(bot.heldItem ? bot.heldItem.type : null)) {
      throw new ActionError(`${block.name} drops nothing without a better tool`)
    }
    const name = block.name
    const before = inventoryCounts(bot)
    await dig(bot, block, context)
    await collectNearby(bot, context, 4)
    return { mined: name, gained: gained(before, bot) }
  },

  async tunnel (bot, { direction, length }, context) {
    const vector = direction_(direction)
    if (vector.y !== 0) throw new ActionError('tunnel goes north, south, east or west; use stairs to go up or down')
    const count = Math.max(1, Math.min(int(length ?? 8, 'length'), MAX_TUNNEL))
    const before = inventoryCounts(bot)
    let dug = 0; const revealed = new Set()
    for (let i = 0; i < count; i++) {
      const feet = bot.entity.position.floored().plus(vector)
      for (const position of [feet.offset(0, 1, 0), feet]) {
        const block = bot.blockAt(position)
        if (block && block.boundingBox === 'block') {
          await equipBestTool(bot, block)
          await dig(bot, block, context)
          dug++
        }
      }
      for (const name of hazardsInSight(bot, context)) revealed.add(name)
      if (revealed.has('lava')) return { dug, stopped: 'lava came into sight', position: position(bot), gained: gained(before, bot) }
      await travel(bot, new goals.GoalBlock(feet.x, feet.y, feet.z), context)
    }
    await collectNearby(bot, context, 3)
    return { dug, position: position(bot), gained: gained(before, bot) }
  },

  async stairs (bot, { direction, steps, way }, context) {
    // A staircase: each step goes one forward and one down (or up), never straight down.
    const vector = direction_(direction)
    if (vector.y !== 0) throw new ActionError('stairs need north, south, east or west')
    const down = (way ?? 'down') === 'down'
    const count = Math.max(1, Math.min(int(steps ?? 8, 'steps'), MAX_TUNNEL))
    let taken = 0
    for (let i = 0; i < count; i++) {
      const here = bot.entity.position.floored()
      const next = here.plus(vector).offset(0, down ? -1 : 1, 0)
      const toClear = down ? [here.plus(vector).offset(0, 1, 0), here.plus(vector), next] : [here.offset(0, 2, 0), next.offset(0, 1, 0), next]
      for (const position of toClear) {
        const block = bot.blockAt(position)
        if (block && block.boundingBox === 'block') {
          await equipBestTool(bot, block)
          await dig(bot, block, context)
        }
      }
      if (hazardsInSight(bot, context).has('lava')) return { steps: taken, stopped: 'lava came into sight', position: position(bot) }
      const floor = bot.blockAt(next.offset(0, -1, 0))
      if (!floor || floor.boundingBox !== 'block') return { steps: taken, stopped: 'no floor to stand on', position: position(bot) }
      await travel(bot, new goals.GoalBlock(next.x, next.y, next.z), context)
      taken++
    }
    return { steps: taken, position: position(bot) }
  },

  async collect (bot, args, context) {
    const before = inventoryCounts(bot)
    await collectNearby(bot, context, 8)
    return { gained: gained(before, bot) }
  },

  async craft (bot, { item, count }, context) {
    const wanted = itemNamed(bot, item)
    const table = stationInReach(bot, 'crafting_table')
    const recipes = bot.recipesFor(wanted.id, null, 1, table)
    if (recipes.length === 0) {
      const needsTable = bot.recipesAll(wanted.id, null, true).length > 0 && table === null
      throw new ActionError(needsTable
        ? `${item} needs a crafting table within reach; place one first`
        : `you do not have the ingredients for ${item}`)
    }
    const recipe = recipes[0]
    const times = Math.max(1, Math.ceil(Math.max(1, int(count ?? 1, 'count')) / recipe.result.count))
    const before = countOf(bot, wanted.name)
    await bot.craft(recipe, times, table ?? undefined)
    return { crafted: wanted.name, made: countOf(bot, wanted.name) - before }
  },

  async place (bot, { item, x, y, z }, context) {
    const held = bot.inventory.items().find(stack => stack.name === item)
    if (!held) throw new ActionError(`you have no ${item}`)
    const target = x === undefined ? freeSpotNextTo(bot) : new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z'))
    if (target === null) throw new ActionError('there is no free spot with a floor next to you')
    if (target.offset(0.5, 0.5, 0.5).distanceTo(eyes(bot)) > REACH) throw new ActionError('that spot is out of reach')
    const floor = bot.blockAt(target.offset(0, -1, 0))
    if (!floor || floor.boundingBox !== 'block') throw new ActionError('there is nothing to place it on')
    await bot.equip(held, 'hand')
    await bot.placeBlock(floor, new Vec3(0, 1, 0))
    return { placed: item, x: target.x, y: target.y, z: target.z }
  },

  async smelt (bot, { item, fuel, count }, context) {
    const block = stationInReach(bot, 'furnace')
    if (block === null) throw new ActionError('there is no furnace within reach; craft and place one')
    const input = itemNamed(bot, item); const burn = itemNamed(bot, fuel ?? 'coal')
    const furnace = await bot.openFurnace(block)
    try {
      const amount = Math.max(1, Math.min(int(count ?? 1, 'count'), countOf(bot, input.name)))
      if (countOf(bot, input.name) === 0) throw new ActionError(`you have no ${input.name}`)
      if (countOf(bot, burn.name) === 0 && !furnace.fuelItem()) throw new ActionError(`you have no ${burn.name} for fuel`)
      if (countOf(bot, burn.name) > 0) await furnace.putFuel(burn.id, null, Math.min(countOf(bot, burn.name), Math.ceil(amount / 8)))
      await furnace.putInput(input.id, null, amount)
      return { smelting: input.name, count: amount, note: 'each item takes 10 seconds; come back and use take_smelted' }
    } finally { furnace.close() }
  },

  async take_smelted (bot, args, context) {
    const block = stationInReach(bot, 'furnace')
    if (block === null) throw new ActionError('there is no furnace within reach')
    const furnace = await bot.openFurnace(block)
    try {
      const output = furnace.outputItem()
      if (!output) return { taken: null, still_smelting: furnace.inputItem()?.name ?? null }
      await furnace.takeOutput()
      return { taken: output.name, count: output.count, still_smelting: furnace.inputItem()?.name ?? null }
    } finally { furnace.close() }
  },

  async open_chest (bot, { x, y, z }, context) {
    const block = visibleInReach(bot, new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z')))
    if (!/chest|barrel/.test(block.name)) throw new ActionError(`${block.name} is not a container`)
    const container = await bot.openContainer(block)
    try {
      return { contents: summarize(container.containerItems()) }
    } finally { container.close() }
  },

  async take (bot, { x, y, z, item, count }, context) {
    const block = visibleInReach(bot, new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z')))
    const container = await bot.openContainer(block)
    try {
      const wanted = itemNamed(bot, item)
      const available = container.containerItems().filter(stack => stack.type === wanted.id).reduce((total, stack) => total + stack.count, 0)
      if (available === 0) throw new ActionError(`there is no ${item} in it`)
      const amount = Math.min(available, int(count ?? available, 'count'))
      await container.withdraw(wanted.id, null, amount)
      return { took: item, count: amount, left: summarize(container.containerItems()) }
    } finally { container.close() }
  },

  async store (bot, { x, y, z, item, count }, context) {
    const block = visibleInReach(bot, new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z')))
    const container = await bot.openContainer(block)
    try {
      const wanted = itemNamed(bot, item)
      const amount = Math.min(countOf(bot, wanted.name), int(count ?? countOf(bot, wanted.name), 'count'))
      if (amount === 0) throw new ActionError(`you have no ${item}`)
      await container.deposit(wanted.id, null, amount)
      return { stored: item, count: amount }
    } finally { container.close() }
  },

  async give (bot, { to, item, count }, context) {
    if (!context.team.has(to) || to === bot.username) throw new ActionError(`${to} is not a teammate`)
    const mate = bot.players[to]?.entity
    if (!mate || !lineOfSight(bot, eyes(bot), mate.position.offset(0, 1, 0))) throw new ActionError(`you cannot see ${to}`)
    const wanted = itemNamed(bot, item)
    const amount = Math.min(countOf(bot, wanted.name), int(count ?? 1, 'count'))
    if (amount === 0) throw new ActionError(`you have no ${item}`)
    await travel(bot, new goals.GoalNear(mate.position.x, mate.position.y, mate.position.z, 2), context)
    await bot.lookAt(mate.position.offset(0, 1.2, 0), true)
    await bot.toss(wanted.id, null, amount)
    return { tossed: item, count: amount, to }
  },

  async equip (bot, { item }, context) {
    const held = bot.inventory.items().find(stack => stack.name === item)
    if (!held) throw new ActionError(`you have no ${item}`)
    await bot.equip(held, 'hand')
    return { holding: item }
  },

  async eat (bot, { item }, context) {
    const food = bot.inventory.items().find(stack => stack.name === item)
    if (!food) throw new ActionError(`you have no ${item}`)
    await bot.equip(food, 'hand')
    await bot.consume()
    return { ate: item, food: bot.food }
  },

  async chat (bot, { message }, context) {
    const text = String(message ?? '').replace(/\s+/g, ' ').trim().slice(0, 240)
    if (!text) throw new ActionError('say something')
    bot.chat(text)
    return { said: text }
  },

  async wait (bot, args, context) {
    await new Promise((resolve) => {
      if (context.signal.aborted) return resolve()
      context.signal.addEventListener('abort', resolve, { once: true })
    })
    return { waited: true }
  }
}

// Helpers

async function travel (bot, goal, context) {
  const abort = () => bot.pathfinder.stop()
  context.signal.addEventListener('abort', abort, { once: true })
  try {
    await bot.pathfinder.goto(goal)
  } catch (error) {
    if (context.signal.aborted) throw new Interrupted()
    throw new ActionError(`could not get there: ${error.message}`)
  } finally {
    context.signal.removeEventListener('abort', abort)
  }
}

async function dig (bot, block, context) {
  if (context.signal.aborted) throw new Interrupted()
  const abort = () => bot.stopDigging()
  context.signal.addEventListener('abort', abort, { once: true })
  try {
    await bot.dig(block, true)
  } catch (error) {
    if (context.signal.aborted) throw new Interrupted()
    throw new ActionError(`digging ${block.name} failed: ${error.message}`)
  } finally {
    context.signal.removeEventListener('abort', abort)
  }
}

// Walk over visible drops nearby (most are picked up without walking, as they land). Each drop gets a short
// time: one already picked up, or out of reach, is skipped.
async function collectNearby (bot, context, radius) {
  await sleep(250) // drops spawn a moment after the block breaks
  const origin = eyes(bot)
  const drops = Object.values(bot.entities).filter(entity => entity.name === 'item' &&
    entity.position.distanceTo(bot.entity.position) <= radius && lineOfSight(bot, origin, entity.position.offset(0, 0.25, 0)))
  for (const drop of drops) {
    if (!bot.entities[drop.id]) continue // already picked up
    const goal = new goals.GoalNear(drop.position.x, drop.position.y, drop.position.z, 0.5)
    const gone = new Promise(resolve => {
      const check = setInterval(() => { if (!bot.entities[drop.id]) { clearInterval(check); resolve() } }, 100)
      setTimeout(() => { clearInterval(check); resolve() }, 2500)
    })
    try {
      await Promise.race([travel(bot, goal, context), gone])
    } catch (error) {
      if (error instanceof Interrupted) throw error
    } finally {
      bot.pathfinder.stop()
    }
  }
  await sleep(200)
}

function inventoryCounts (bot) {
  const totals = {}
  for (const stack of bot.inventory.items()) totals[stack.name] = (totals[stack.name] ?? 0) + stack.count
  return totals
}

// What the inventory gained since `before` (what was mined and picked up), as {item: count}.
function gained (before, bot) {
  const after = inventoryCounts(bot)
  const result = {}
  for (const [name, count] of Object.entries(after)) {
    if (count > (before[name] ?? 0)) result[name] = count - (before[name] ?? 0)
  }
  return result
}

async function equipBestTool (bot, block) {
  let best = null; let fastest = block.digTime(null, false, false, false, [], [])
  for (const item of bot.inventory.items()) {
    const time = block.digTime(item.type, false, false, false, [], [])
    if (time < fastest && block.canHarvest(item.type)) { best = item; fastest = time }
  }
  if (best && (!bot.heldItem || bot.heldItem.type !== best.type)) await bot.equip(best, 'hand')
}

function visibleInReach (bot, position) {
  const block = bot.blockAt(position)
  if (!block || block.boundingBox !== 'block') throw new ActionError(`there is no solid block at (${position.x}, ${position.y}, ${position.z})`)
  const center = position.offset(0.5, 0.5, 0.5)
  const origin = eyes(bot)
  if (center.distanceTo(origin) > REACH) throw new ActionError(`(${position.x}, ${position.y}, ${position.z}) is out of reach (${round(center.distanceTo(origin))} blocks; reach is ${REACH})`)
  const direction = center.minus(origin)
  const hit = firstHit(bot, origin, direction.scaled(1 / direction.norm()), REACH + 1)
  if (!hit || !hit.position.equals(position)) throw new ActionError(`you cannot see (${position.x}, ${position.y}, ${position.z}) from here`)
  return block
}

function stationInReach (bot, name) {
  const id = bot.registry.blocksByName[name]?.id
  const origin = eyes(bot)
  for (const position of bot.findBlocks({ matching: id, maxDistance: 5, count: 8 })) {
    const center = position.offset(0.5, 0.5, 0.5)
    if (center.distanceTo(origin) <= REACH && lineOfSight(bot, origin, center)) return bot.blockAt(position)
  }
  return null
}

function hazardsInSight (bot, context) {
  const names = new Set()
  for (const block of look(bot, context.memory).values()) {
    if (block.name === 'lava' && block.position.distanceTo(bot.entity.position) < 4) names.add('lava')
  }
  return names
}

function freeSpotNextTo (bot) {
  const here = bot.entity.position.floored()
  const occupied = new Set(Object.values(bot.entities).filter(e => e.type === 'player').map(e => key(e.position.floored())))
  for (const offset of [DIRECTIONS.north, DIRECTIONS.south, DIRECTIONS.east, DIRECTIONS.west]) {
    const spot = here.plus(offset)
    const block = bot.blockAt(spot); const floor = bot.blockAt(spot.offset(0, -1, 0))
    if (block && block.boundingBox === 'empty' && floor && floor.boundingBox === 'block' && !occupied.has(key(spot))) return spot
  }
  return null
}

function known (memory, target) {
  for (const [dx, dy, dz] of [[0, 0, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -2, 0]]) {
    if (memory.has(`${target.x + dx},${target.y + dy},${target.z + dz}`)) return true
  }
  return false
}

function direction_ (name) {
  const vector = DIRECTIONS[name]
  if (!vector) throw new ActionError(`direction must be one of ${Object.keys(DIRECTIONS).join(', ')}`)
  return vector
}

function itemNamed (bot, name) {
  const item = bot.registry.itemsByName[name]
  if (!item) throw new ActionError(`there is no item called ${name}`)
  return item
}

function countOf (bot, name) {
  return bot.inventory.items().filter(stack => stack.name === name).reduce((total, stack) => total + stack.count, 0)
}

function summarize (stacks) {
  const totals = {}
  for (const stack of stacks) totals[stack.name] = (totals[stack.name] ?? 0) + stack.count
  return totals
}

function int (value, name) {
  const number = Number(value)
  if (!Number.isFinite(number)) throw new ActionError(`${name} must be a number`)
  return Math.round(number)
}

function position (bot) {
  const here = bot.entity.position.floored()
  return { x: here.x, y: here.y, z: here.z }
}

function round (value) { return Math.round(value * 10) / 10 }
function sleep (ms) { return new Promise(resolve => setTimeout(resolve, ms)) }

class Interrupted extends Error {}

module.exports = { ACTIONS, ActionError, Interrupted }
