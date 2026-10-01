'use strict'
// What a bot may perceive: only what it could see with its own eyes.
//
// The client's world data already lacks hidden ores (Paper's anti-xray sends ores no air touches as stone), but it
// still holds caves, chests and ores behind walls. Observations therefore come from rays cast from the bot's eyes in
// every direction: a block is seen if some ray hits it first, an entity if nothing opaque lies between the eyes and
// it. Players outside the team are never reported.

const { Vec3 } = require('vec3')

const RANGE = 24 // blocks
const FAR_RANGE = 96 // large or glowing things in the open are seen from much farther
const FAR = new Set(['ender_dragon', 'end_crystal', 'ghast'])
const MAX_KINDS = 14 // kinds of notable blocks per observation
const RAYS = 2400 // directions, spread evenly over the sphere
const NOTABLE = /(_ore$|^ancient_debris$|^chest$|^trapped_chest$|^barrel$|^crafting_table$|^furnace$|^blast_furnace$|^lava$|^water$|^diamond_block$|^iron_block$|^spawner$|^torch$|^wall_torch$|^obsidian$|^crying_obsidian$|^nether_portal$|^end_portal$|^end_portal_frame$|^nether_bricks$|^bed$|_bed$|^gravel$|^sand$|_log$|^bedrock$|^end_stone$)/
const SEE_THROUGH = new Set(['air', 'cave_air', 'void_air', 'water', 'glass', 'glass_pane', 'torch', 'wall_torch',
  'light', 'short_grass', 'tall_grass', 'fern', 'vine', 'glow_lichen', 'cave_vines', 'cave_vines_plant', 'snow'])
const ANIMALS = new Set(['cow', 'pig', 'sheep', 'chicken', 'rabbit', 'mooshroom', 'goat'])
const HOSTILE = new Set(['zombie', 'skeleton', 'creeper', 'spider', 'cave_spider', 'enderman', 'witch', 'slime',
  'drowned', 'husk', 'stray', 'silverfish', 'bogged', 'zombie_villager', 'phantom', 'blaze', 'ghast',
  'wither_skeleton', 'magma_cube', 'piglin_brute', 'hoglin', 'zoglin', 'endermite', 'shulker', 'ender_dragon',
  'end_crystal', // not a creature, but a target: it heals the dragon and explodes when hit
  'pillager', 'vindicator', 'evoker', 'ravager', 'guardian', 'elder_guardian', 'breeze'])
const DIRECTIONS = {
  north: new Vec3(0, 0, -1), south: new Vec3(0, 0, 1), east: new Vec3(1, 0, 0), west: new Vec3(-1, 0, 0),
  up: new Vec3(0, 1, 0), down: new Vec3(0, -1, 0)
}

// Directions spread over the sphere (a Fibonacci lattice), computed once.
const SPHERE = (() => {
  const result = []
  const golden = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < RAYS; i++) {
    const y = 1 - (i / (RAYS - 1)) * 2
    const radius = Math.sqrt(1 - y * y)
    const theta = golden * i
    result.push(new Vec3(Math.cos(theta) * radius, y, Math.sin(theta) * radius))
  }
  return result
})()

function eyes (bot) {
  return bot.entity.position.offset(0, bot.entity.eyeHeight ?? 1.62, 0)
}

function opaque (block) {
  return block !== null && !SEE_THROUGH.has(block.name)
}

// The first opaque block along a ray, stepping through the voxel grid (Amanatides & Woo).
function firstHit (bot, origin, direction, range) {
  let x = Math.floor(origin.x); let y = Math.floor(origin.y); let z = Math.floor(origin.z)
  const step = [Math.sign(direction.x), Math.sign(direction.y), Math.sign(direction.z)]
  const delta = [Math.abs(1 / direction.x), Math.abs(1 / direction.y), Math.abs(1 / direction.z)]
  const boundary = (position, cell, sign) => sign > 0 ? cell + 1 - position : position - cell
  const max = [boundary(origin.x, x, step[0]) * delta[0], boundary(origin.y, y, step[1]) * delta[1],
    boundary(origin.z, z, step[2]) * delta[2]]
  let travelled = 0
  while (travelled <= range) {
    if (max[0] < max[1] && max[0] < max[2]) { x += step[0]; travelled = max[0]; max[0] += delta[0] } else if (max[1] < max[2]) { y += step[1]; travelled = max[1]; max[1] += delta[1] } else { z += step[2]; travelled = max[2]; max[2] += delta[2] }
    if (travelled > range) break
    const block = bot.blockAt(new Vec3(x, y, z), false)
    if (block === null) return null // an unloaded chunk: nothing seen
    if (opaque(block)) return block
  }
  return null
}

function lineOfSight (bot, from, to) {
  const offset = to.minus(from)
  const distance = offset.norm()
  if (distance < 1e-6) return true
  const hit = firstHit(bot, from, offset.scaled(1 / distance), distance)
  return hit === null || hit.position.offset(0.5, 0.5, 0.5).distanceTo(from) >= distance - 0.5
}

function key (position) { return `${position.x},${position.y},${position.z}` }

// Everything a bot can see now. `memory` (a Map of positions it has seen) is updated, so actions can check that
// their targets were seen.
function look (bot, memory) {
  const origin = eyes(bot)
  const seen = new Map()
  for (const direction of SPHERE) {
    const block = firstHit(bot, origin, direction, RANGE)
    if (block !== null) seen.set(key(block.position), block)
  }
  for (const [position, block] of seen) memory.set(position, block.name)
  return seen
}

function relative (bot, position) {
  const here = bot.entity.position.floored()
  return { x: position.x, y: position.y, z: position.z, dx: position.x - here.x, dy: position.y - here.y, dz: position.z - here.z }
}

function observe (bot, team, memory) {
  const seen = look(bot, memory)
  const here = bot.entity.position.floored()
  const counts = {}
  const kinds = new Map() // block name → the ones in sight, nearest first
  for (const block of seen.values()) {
    counts[block.name] = (counts[block.name] ?? 0) + 1
    if (NOTABLE.test(block.name)) {
      if (!kinds.has(block.name)) kinds.set(block.name, [])
      kinds.get(block.name).push({ ...relative(bot, block.position), distance: round(block.position.offset(0.5, 0.5, 0.5).distanceTo(bot.entity.position)) })
    }
  }
  // One entry per kind: the nearest one, how many are in sight, and the next few (a forest is one line, not thirty).
  const notable = []
  for (const [name, found] of kinds) {
    found.sort((a, b) => a.distance - b.distance)
    notable.push({ block: name, count: found.length, ...found[0], also: found.slice(1, 4).map(({ x, y, z }) => ({ x, y, z })) })
  }
  notable.sort((a, b) => a.distance - b.distance)

  const origin = eyes(bot)
  const surroundings = {}
  for (const [name, direction] of Object.entries(DIRECTIONS)) {
    const hit = firstHit(bot, origin, direction, RANGE)
    surroundings[name] = hit === null ? { open: `more than ${RANGE}` } : { open: round(hit.position.offset(0.5, 0.5, 0.5).distanceTo(origin) - 0.5), then: hit.name }
  }

  const teammates = []; const items = []; const mobs = []; const animals = []
  for (const entity of Object.values(bot.entities)) {
    const range = FAR.has(entity.name) ? FAR_RANGE : RANGE
    if (entity === bot.entity || entity.position.distanceTo(bot.entity.position) > range) continue
    const center = entity.position.offset(0, (entity.height ?? 1) / 2, 0)
    if (entity.type === 'player') {
      if (!team.has(entity.username)) continue // only teammates exist, as far as agents know
      if (!lineOfSight(bot, origin, center)) continue
      teammates.push({ name: entity.username, ...relative(bot, entity.position.floored()) })
    } else if (entity.name === 'item') {
      if (!lineOfSight(bot, origin, center)) continue
      const item = entity.getDroppedItem?.()
      if (item) items.push({ item: item.name, count: item.count, ...relative(bot, entity.position.floored()) })
    } else if (HOSTILE.has(entity.name) || ANIMALS.has(entity.name)) {
      if (!lineOfSight(bot, origin, center)) continue
      const seen = { id: entity.id, ...relative(bot, entity.position.floored()), distance: round(entity.position.distanceTo(bot.entity.position)) }
      if (HOSTILE.has(entity.name)) mobs.push({ mob: entity.name, ...seen })
      else animals.push({ animal: entity.name, ...seen })
    }
  }

  const feet = bot.blockAt(here)
  return {
    self: {
      name: bot.username,
      position: { x: here.x, y: here.y, z: here.z },
      dimension: String(bot.game?.dimension ?? 'overworld').replace('minecraft:', ''),
      health: round(bot.health ?? 0),
      food: bot.food ?? 0,
      holding: bot.heldItem ? bot.heldItem.name : null,
      wearing: armor(bot),
      inventory: inventory(bot),
      near: nearbyStations(bot)
    },
    world: {
      time: timeOfDay(bot),
      light: feet ? light(bot, feet) : null,
      sky: feet ? feet.skyLight >= 15 : null,
      biome: feet?.biome?.name ?? null
    },
    surroundings,
    visible_blocks: Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([block, count]) => ({ block, count })),
    notable: notable.slice(0, MAX_KINDS),
    teammates,
    items,
    mobs: mobs.slice(0, 10),
    animals: animals.slice(0, 10)
  }
}

// Worn armor, by slot (inventory slots 5 to 8 are head, torso, legs and feet).
function armor (bot) {
  const slots = { head: 5, torso: 6, legs: 7, feet: 8 }
  const worn = {}
  for (const [slot, index] of Object.entries(slots)) {
    const item = bot.inventory.slots[index]
    if (item) worn[slot] = item.name
  }
  const offHand = bot.inventory.slots[45]
  if (offHand) worn['off-hand'] = offHand.name
  return worn
}

// The light where the bot stands: from blocks, or from the sky (which gives little at night).
function light (bot, feet) {
  const ticks = bot.time?.timeOfDay ?? 0
  const night = ticks >= 13000 && ticks < 23000
  return Math.max(feet.light ?? 0, night ? Math.min(feet.skyLight ?? 0, 4) : (feet.skyLight ?? 0))
}

function timeOfDay (bot) {
  const ticks = bot.time?.timeOfDay ?? 0
  const phase = ticks < 12000 ? 'day' : ticks < 13800 ? 'dusk' : ticks < 22200 ? 'night' : 'dawn'
  return { ticks, phase }
}

function inventory (bot) {
  const totals = {}
  for (const item of bot.inventory.items()) totals[item.name] = (totals[item.name] ?? 0) + item.count
  return totals
}

// Crafting tables, furnaces and chests within reach (4.5 blocks) and in sight: what crafting and smelting can use.
function nearbyStations (bot) {
  const result = []
  const origin = eyes(bot)
  for (const name of ['crafting_table', 'furnace', 'chest']) {
    const id = bot.registry.blocksByName[name]?.id
    if (id === undefined) continue
    for (const position of bot.findBlocks({ matching: id, maxDistance: 5, count: 4 })) {
      if (position.offset(0.5, 0.5, 0.5).distanceTo(origin) <= 4.5 && lineOfSight(bot, origin, position.offset(0.5, 0.5, 0.5))) {
        result.push({ station: name, x: position.x, y: position.y, z: position.z })
      }
    }
  }
  return result
}

function round (value) { return Math.round(value * 10) / 10 }

module.exports = { observe, look, lineOfSight, firstHit, eyes, key, opaque, RANGE, DIRECTIONS }
