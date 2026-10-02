'use strict'
// A made-up world for tests: a bot's view of it needs no server.

const { Vec3 } = require('vec3')

const SOLID = new Set(['stone', 'chest', 'deepslate_diamond_ore'])

// Stone everywhere, except what `carve` empties and `put` places.
function world () {
  const blocks = new Map()
  const at = (x, y, z) => `${x},${y},${z}`
  return {
    carve (x1, x2, y1, y2, z1, z2) {
      for (let x = x1; x <= x2; x++) for (let y = y1; y <= y2; y++) for (let z = z1; z <= z2; z++) blocks.set(at(x, y, z), 'air')
    },
    put (x, y, z, name) { blocks.set(at(x, y, z), name) },
    name (x, y, z) { return blocks.get(at(x, y, z)) ?? 'stone' }
  }
}

function botIn (blocks, position) {
  const registry = { blocksByName: {} }
  for (const name of ['stone', 'chest', 'deepslate_diamond_ore', 'water', 'air']) {
    registry.blocksByName[name] = { id: name.length, boundingBox: SOLID.has(name) ? 'block' : 'empty' }
  }
  return {
    username: 'ada',
    entity: { position, eyeHeight: 1.62, height: 1.8 },
    entities: {},
    health: 20,
    food: 20,
    heldItem: null,
    inventory: { items: () => [], slots: [] },
    game: { dimension: 'overworld' },
    time: { timeOfDay: 1000 },
    registry,
    findBlocks: () => [],
    blockAt (point) {
      const cell = point.floored()
      const name = blocks.name(cell.x, cell.y, cell.z)
      return { name, position: cell, boundingBox: SOLID.has(name) ? 'block' : 'empty', skyLight: 0, biome: { name: 'plains' } }
    }
  }
}

// A 7 by 7 room, three high, with a corridor out of its east wall; a second room to the south with a chest in it,
// not connected; diamond ore in the west wall; water in a corner.
function scene () {
  const blocks = world()
  blocks.carve(-3, 3, 64, 66, -3, 3)
  blocks.carve(4, 10, 64, 65, 0, 0)
  blocks.carve(-3, 3, 64, 66, 5, 7)
  blocks.put(0, 64, 6, 'chest')
  blocks.put(-4, 65, 0, 'deepslate_diamond_ore')
  blocks.put(2, 64, 2, 'water')
  return blocks
}

module.exports = { world, botIn, scene, Vec3 }
