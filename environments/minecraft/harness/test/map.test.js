'use strict'
// The map an agent is shown, on a made-up world (no server): what is in it, and what is not.

const test = require('node:test')
const assert = require('node:assert')
const { Vec3 } = require('vec3')
const { observe } = require(process.env.OBSERVE ?? '../lib/observe')

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
