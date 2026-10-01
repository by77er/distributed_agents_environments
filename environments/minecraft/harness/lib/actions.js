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

  async equip (bot, { item, slot }, context) {
    const held = bot.inventory.items().find(stack => stack.name === item)
    if (!held) throw new ActionError(`you have no ${item}`)
    const destination = slot ?? 'hand'
    if (!['hand', 'off-hand', 'head', 'torso', 'legs', 'feet'].includes(destination)) {
      throw new ActionError('slot must be hand, off-hand, head, torso, legs or feet')
    }
    await bot.equip(held, destination)
    return { equipped: item, slot: destination }
  },

  async attack (bot, { target }, context) {
    // A mob or animal the observation listed (by its id), attacked until it dies or time runs out.
    const entity = bot.entities[int(target, 'target')]
    if (!entity || !entity.isValid) throw new ActionError(`there is no creature ${target} in sight`)
    if (!lineOfSight(bot, eyes(bot), entity.position.offset(0, (entity.height ?? 1) / 2, 0))) {
      throw new ActionError(`you cannot see ${target}`)
    }
    await equipBestWeapon(bot)
    let hits = 0
    while (entity.isValid && bot.entities[entity.id] && !context.signal.aborted) {
      if (entity.position.distanceTo(bot.entity.position) > 3) {
        const goal = new goals.GoalNear(entity.position.x, entity.position.y, entity.position.z, 2)
        await Promise.race([travel(bot, goal, context), sleep(1500)]).catch(error => { if (error instanceof Interrupted) throw error })
        continue
      }
      await bot.lookAt(entity.position.offset(0, (entity.height ?? 1) * 0.8, 0), true)
      bot.attack(entity)
      hits++
      await sleep(650) // the attack cooldown of most weapons
    }
    if (context.signal.aborted && bot.entities[entity.id]) throw new Interrupted()
    return { defeated: entity.name, hits }
  },

  async shoot (bot, { target }, context) {
    // An arrow at a creature (or an end crystal) the observation listed, aimed above it for the arrow's drop.
    const entity = bot.entities[int(target, 'target')]
    if (!entity || !entity.isValid) throw new ActionError(`there is no creature ${target} in sight`)
    const bow = bot.inventory.items().find(stack => stack.name === 'bow')
    if (!bow) throw new ActionError('you have no bow')
    if (countOf(bot, 'arrow') === 0) throw new ActionError('you have no arrows')
    // Where to point: at where the target will be when the arrow arrives (it keeps the velocity it showed during the
    // draw), raised for the arrow's drop.
    const aim = (velocity) => {
      const center = entity.position.offset(0, (entity.height ?? 1) / 2, 0)
      const flat = Math.hypot(center.x - bot.entity.position.x, center.z - bot.entity.position.z)
      const seconds = flat / 50 // a full draw leaves at 60 blocks a second and slows in the air
      return center.plus(velocity.scaled(seconds)).offset(0, 10 * seconds * seconds, 0) // gravity: 20 blocks/s²
    }
    const still = new Vec3(0, 0, 0)
    if (!lineOfSight(bot, eyes(bot), entity.position.offset(0, (entity.height ?? 1) / 2, 0))) {
      throw new ActionError(`you cannot see ${target}`)
    }
    await bot.equip(bow, 'hand')
    let shots = 0
    while (entity.isValid && bot.entities[entity.id] && countOf(bot, 'arrow') > 0 && !context.signal.aborted && shots < 3) {
      await bot.lookAt(aim(still), true)
      bot.activateItem()
      await sleep(800)
      const before = entity.position.clone()
      await sleep(300) // 1.1 seconds: a full draw
      await bot.lookAt(aim(entity.position.minus(before).scaled(1 / 0.3)), true)
      bot.deactivateItem()
      shots++
      await sleep(300)
    }
    const alive = Boolean(bot.entities[entity.id])
    return { shot: entity.name, arrows: shots, target_still_there: alive, arrows_left: countOf(bot, 'arrow') }
  },

  async use (bot, { item, x, y, z }, context) {
    // Use the held item (or `item`) on a block you see within reach, or in the air: flint and steel on obsidian,
    // a bucket on water or lava, an eye of ender in the air or on an end portal frame.
    if (item) {
      const stack = bot.inventory.items().find(entry => entry.name === item)
      if (!stack) throw new ActionError(`you have no ${item}`)
      await bot.equip(stack, 'hand')
    }
    const name = bot.heldItem?.name ?? 'your hand'
    if (x === undefined) {
      if (name === 'ender_eye') return throwEye(bot)
      bot.activateItem()
      await sleep(300)
      bot.deactivateItem()
      return { used: name }
    }
    const position = new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z'))
    const block = bot.blockAt(position)
    if (!block) throw new ActionError('nothing is there')
    if (position.offset(0.5, 0.5, 0.5).distanceTo(eyes(bot)) > REACH) throw new ActionError('that is out of reach')
    const blocking = whatHides(bot, block)
    if (blocking !== null) throw new ActionError(`you cannot see it from here: ${blocking} is in the way`)
    const before = inventoryCounts(bot)
    if (name.endsWith('bucket')) { // buckets are used on what the player looks at
      await bot.lookAt(position.offset(0.5, name === 'bucket' ? 0.5 : 1.0, 0.5), true)
      bot.activateItem()
      await sleep(400)
      bot.deactivateItem()
      return { used: name, on: block.name, now_holding: bot.heldItem?.name ?? null }
    }
    await bot.lookAt(position.offset(0.5, 0.5, 0.5), true)
    await bot.activateBlock(block)
    await sleep(300)
    return { used: name, on: block.name, at: { x: position.x, y: position.y, z: position.z }, spent: spent(before, bot) }
  },

  async place_at (bot, { item, x, y, z }, context) {
    // Place a block at a free position the bot sees within reach, against a solid neighbour. (A player clicks a
    // face of the neighbour; the bot needs only to see the free position, so it can build above its eye level.)
    const held = bot.inventory.items().find(stack => stack.name === item)
    if (!held) throw new ActionError(`you have no ${item}`)
    const target = new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z'))
    const there = bot.blockAt(target)
    if (!there || there.boundingBox !== 'empty') throw new ActionError(`(${target.x}, ${target.y}, ${target.z}) is not free`)
    if (target.offset(0.5, 0.5, 0.5).distanceTo(eyes(bot)) > REACH) throw new ActionError('that spot is out of reach')
    if (standsIn(bot, target)) throw new ActionError('someone is standing there')
    if (!seesCell(bot, target)) throw new ActionError('you cannot see that position from here')
    for (const offset of [DIRECTIONS.down, DIRECTIONS.north, DIRECTIONS.south, DIRECTIONS.east, DIRECTIONS.west, DIRECTIONS.up]) {
      const against = bot.blockAt(target.plus(offset))
      if (!against || against.boundingBox !== 'block') continue
      await bot.equip(held, 'hand')
      await bot.placeBlock(against, offset.scaled(-1))
      return { placed: item, x: target.x, y: target.y, z: target.z }
    }
    throw new ActionError('there is no block next to that position to place it against')
  },

  async sleep (bot, { x, y, z }, context) {
    const block = visibleInReach(bot, new Vec3(int(x, 'x'), int(y, 'y'), int(z, 'z')), true)
    if (!block.name.endsWith('_bed')) throw new ActionError(`${block.name} is not a bed`)
    if (block.position.offset(0.5, 0.5, 0.5).distanceTo(bot.entity.position) > 2) {
      await travel(bot, new goals.GoalNear(block.position.x, block.position.y, block.position.z, 1), context)
    }
    await bot.sleep(block)
    return { sleeping: true, note: 'you wake when the night is over, if every player sleeps' }
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
    bot.pathfinder.setGoal(null) // clears a stop left over from the last freeze, which would end this path at once
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

// Throw an eye of ender and report where it went: it flies toward the nearest stronghold, and sinks when the
// stronghold is close below.
async function throwEye (bot) {
  const known = new Set(Object.keys(bot.entities))
  const start = bot.entity.position.clone()
  await bot.look(bot.entity.yaw, 0.6, true) // up a little, so the eye is not thrown into the ground
  bot.activateItem()
  let eye = null
  for (let i = 0; i < 20 && eye === null; i++) {
    await sleep(100)
    eye = Object.values(bot.entities).find(entity => !known.has(String(entity.id)) && /eye_of_ender|ender_eye/.test(entity.name ?? '')) ?? null
  }
  if (eye === null) return { used: 'ender_eye', flew: null, note: 'the eye was not seen to fly; is there a stronghold in this world?' }
  let last = eye.position.clone()
  for (let i = 0; i < 25 && bot.entities[eye.id]; i++) { await sleep(100); last = eye.position.clone() }
  const dx = last.x - start.x; const dz = last.z - start.z; const flat = Math.hypot(dx, dz)
  if (flat < 1.5) return { used: 'ender_eye', flew: 'straight down: the stronghold is below you; dig down with stairs' }
  return { used: 'ender_eye', flew: compass(dx, dz), toward: { dx: round(dx / flat), dz: round(dz / flat) }, note: 'the stronghold is in that direction; walk that way and throw again' }
}

function compass (dx, dz) {
  const names = ['east', 'south-east', 'south', 'south-west', 'west', 'north-west', 'north', 'north-east']
  return names[((Math.round(Math.atan2(dz, dx) / (Math.PI / 4)) % 8) + 8) % 8]
}

// What the inventory lost since `before`, as {item: count}.
function spent (before, bot) {
  const after = inventoryCounts(bot)
  const result = {}
  for (const [name, count] of Object.entries(before)) {
    if ((after[name] ?? 0) < count) result[name] = count - (after[name] ?? 0)
  }
  return result
}

async function equipBestWeapon (bot) {
  const ranking = ['netherite_sword', 'diamond_sword', 'iron_sword', 'netherite_axe', 'diamond_axe', 'stone_sword',
    'iron_axe', 'golden_sword', 'wooden_sword', 'stone_axe']
  for (const name of ranking) {
    const weapon = bot.inventory.items().find(stack => stack.name === name)
    if (weapon) { if (bot.heldItem?.name !== name) await bot.equip(weapon, 'hand'); return }
  }
}

async function equipBestTool (bot, block) {
  let best = null; let fastest = block.digTime(null, false, false, false, [], [])
  for (const item of bot.inventory.items()) {
    const time = block.digTime(item.type, false, false, false, [], [])
    if (time < fastest && block.canHarvest(item.type)) { best = item; fastest = time }
  }
  if (best && (!bot.heldItem || bot.heldItem.type !== best.type)) await bot.equip(best, 'hand')
}

function visibleInReach (bot, position, anyShape = false) {
  const block = bot.blockAt(position)
  if (!block || (!anyShape && block.boundingBox !== 'block') || block.name.endsWith('air')) {
    throw new ActionError(`there is no solid block at (${position.x}, ${position.y}, ${position.z})`)
  }
  const center = position.offset(0.5, 0.5, 0.5)
  const origin = eyes(bot)
  if (center.distanceTo(origin) > REACH) throw new ActionError(`(${position.x}, ${position.y}, ${position.z}) is out of reach (${round(center.distanceTo(origin))} blocks; reach is ${REACH})`)
  const blocking = whatHides(bot, block)
  if (blocking !== null) {
    throw new ActionError(`you cannot see (${position.x}, ${position.y}, ${position.z}) from here: ${blocking} is in the way`)
  }
  return block
}

// Whether a ray from the eyes to `point` (on or just inside `block`) reaches it without hitting another block first.
function seesPoint (bot, point, block) {
  const origin = eyes(bot)
  const offset = point.minus(origin)
  const distance = offset.norm()
  if (distance < 1e-6) return true
  const hit = firstHit(bot, origin, offset.scaled(1 / distance), distance + 0.2)
  return hit === null || hit.position.equals(block.position)
}

// null if the bot sees `block` (its centre or the middle of any of its sides); otherwise the name of what hides it.
function whatHides (bot, block) {
  const center = block.position.offset(0.5, 0.5, 0.5)
  const points = [center, ...Object.values(DIRECTIONS).map(direction => center.plus(direction.scaled(0.45)))]
  if (points.some(point => seesPoint(bot, point, block))) return null
  const origin = eyes(bot)
  const offset = center.minus(origin)
  const hit = firstHit(bot, origin, offset.scaled(1 / offset.norm()), offset.norm())
  return hit ? hit.name : 'something'
}

// Whether the bot sees into the free cell at `position`: its middle, or a point near any of its sides.
function seesCell (bot, position) {
  const origin = eyes(bot)
  const center = position.offset(0.5, 0.5, 0.5)
  return [center, ...Object.values(DIRECTIONS).map(direction => center.plus(direction.scaled(0.4)))].some(point => {
    const offset = point.minus(origin)
    const distance = offset.norm()
    return distance < 1e-6 || firstHit(bot, origin, offset.scaled(1 / distance), distance) === null
  })
}

// Whether a player's body (feet to head) is in the block at `position`.
function standsIn (bot, position) {
  for (const entity of Object.values(bot.entities)) {
    if (entity.type !== 'player') continue
    for (const dy of [0, 1, 1.8]) if (entity.position.offset(0, dy, 0).floored().equals(position)) return true
  }
  return false
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
  const occupied = new Set() // blocks a player's body is in (feet and head)
  for (const entity of Object.values(bot.entities)) {
    if (entity.type !== 'player') continue
    for (const dy of [0, 1, 1.8]) occupied.add(key(entity.position.offset(0, dy, 0).floored()))
  }
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
