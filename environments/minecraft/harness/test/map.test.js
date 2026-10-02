'use strict'
// The map an agent is shown, on a made-up world (no server): what is in it, and what is not.

const test = require('node:test')
const assert = require('node:assert')
const { Vec3 } = require('vec3')
const { observe } = require('../lib/observe')

const { scene, botIn } = require('./world')

// The name the map gives the cell at an offset from the bot (undefined: not seen).
function cell (map, dx, dy, dz) {
  const layer = map.layers.find(each => each.dy === dy)
  const index = layer.cells[(dz + map.radius) * (2 * map.radius + 1) + dx + map.radius]
  return index === -1 ? undefined : map.palette[index].name
}

test('the map holds what the bot sees and nothing it does not', () => {
  const blocks = scene()
  const { map } = observe(botIn(blocks, new Vec3(0.5, 64, 0.5)), new Set(['ada']), new Map(), new Set())
  assert.deepStrictEqual(map.center, { x: 0, y: 64, z: 0 })
  assert.strictEqual(map.radius, 6)
  assert.deepStrictEqual(map.layers.map(layer => layer.dy), [2, 1, 0, -1, -2])
  assert.strictEqual(map.layers[0].cells.length, 13 * 13)
  assert.strictEqual(cell(map, 0, 0, 0), 'air') // where it stands
  assert.strictEqual(cell(map, 3, 0, 3), 'air') // the room's corner
  assert.strictEqual(cell(map, 6, 0, 0), 'air') // down the corridor: the opening shows
  assert.strictEqual(cell(map, 4, 0, -1), 'stone') // the wall beside the opening
  assert.strictEqual(cell(map, 0, -1, 0), 'stone') // the floor
  assert.strictEqual(cell(map, 0, 0, 4), 'stone') // the south wall
  assert.strictEqual(cell(map, -4, 1, 0), 'deepslate_diamond_ore') // ore in the wall, at head height
  assert.strictEqual(cell(map, 2, 0, 2), 'water')
  assert.strictEqual(cell(map, 0, 0, 5), undefined) // the room behind the wall
  assert.strictEqual(cell(map, 0, 0, 6), undefined) // and its chest
  assert.strictEqual(cell(map, 0, -2, 0), undefined) // under the floor
  assert.ok(!map.palette.some(entry => entry.name === 'chest'))
  assert.deepStrictEqual(map.palette.find(entry => entry.name === 'water'), { name: 'water', solid: false })
  assert.deepStrictEqual(map.palette.find(entry => entry.name === 'stone'), { name: 'stone', solid: true })
})

test('the map remembers what was seen, and follows changes it sees', () => {
  const blocks = scene()
  const memory = new Map()
  const air = new Set()
  observe(botIn(blocks, new Vec3(0.5, 64, 0.5)), new Set(['ada']), memory, air)
  // From inside the corridor the room's far corners are out of sight; they stay on the map.
  let { map } = observe(botIn(blocks, new Vec3(5.5, 64, 0.5)), new Set(['ada']), memory, air)
  assert.deepStrictEqual(map.center, { x: 5, y: 64, z: 0 })
  assert.strictEqual(cell(map, -6, 0, -3), 'air') // the room's north-west side, seen earlier
  assert.strictEqual(cell(map, -5, 0, 5), undefined) // never seen
  // The south wall is opened: seen again from the room, the cell is empty and the second room shows.
  blocks.put(0, 64, 4, 'air')
  blocks.put(0, 65, 4, 'air')
  ;({ map } = observe(botIn(blocks, new Vec3(0.5, 64, 0.5)), new Set(['ada']), memory, air))
  assert.strictEqual(cell(map, 0, 0, 4), 'air')
  assert.strictEqual(cell(map, 0, 0, 6), 'chest')
})

test('an observation lists the nearest few of each kind in sight, and says where the rest are', () => {
  const bot = botIn(scene(), new Vec3(0.5, 64, 0.5))
  const row = (index, z) => new Vec3((index % 6) - 2.5, 64, z + 0.5) // six to a row across the room
  for (let index = 0; index < 12; index++) {
    bot.entities[index + 1] = { id: index + 1, name: 'zombie', position: row(index, index < 6 ? -2 : -3), height: 1.9 }
    bot.entities[index + 21] = { id: index + 21, name: 'cow', position: row(index, index < 6 ? 1 : 3), height: 1.4 }
  }
  for (let index = 0; index < 10; index++) {
    const item = { name: 'diamond', count: index + 1 }
    bot.entities[index + 41] = { id: index + 41, name: 'item', position: row(index, index < 6 ? -1 : 2), height: 0.25, getDroppedItem: () => item }
  }
  const seen = observe(bot, new Set(['ada']), new Map(), new Set())
  assert.deepStrictEqual([seen.mobs.length, seen.animals.length, seen.items.length], [6, 6, 8])
  assert.deepStrictEqual(Object.keys(seen.mobs[0]), ['mob', 'id', 'x', 'y', 'z', 'distance'])
  // The map shows ten creatures of a kind and every item: those past the end of a list come as places only.
  assert.deepStrictEqual([seen.unlisted.mobs.length, seen.unlisted.animals.length, seen.unlisted.items.length], [4, 4, 2])
  assert.deepStrictEqual(seen.unlisted.items, [{ x: -1, y: 64, z: 2 }, { x: 0, y: 64, z: 2 }])
  const away = place => Math.hypot(place.x, place.z) // from the bot, which stands in the cell at the origin
  for (const kind of ['mobs', 'animals']) { // the nearest are the ones listed
    assert.ok(Math.max(...seen[kind].map(away)) <= Math.min(...seen.unlisted[kind].map(away)), kind)
  }
  assert.deepStrictEqual(seen.world, { time: { phase: 'day' }, sky: false, biome: 'plains' })
})
