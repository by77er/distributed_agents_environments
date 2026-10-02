'use strict'
// Parts of the actions that need no server: what burns and for how long, what is worn where, a window that never
// opens, and rays along an axis.

const test = require('node:test')
const assert = require('node:assert')
const { Vec3 } = require('vec3')
const { ACTIONS, ActionError, burnTicks, opened, WORN } = require('../lib/actions')
const { firstHit } = require('../lib/observe')
const { world, botIn } = require('./world')

test('fuels burn as long as in the game, and what does not burn is not fuel', () => {
  assert.strictEqual(burnTicks('coal'), 1600) // eight items
  assert.strictEqual(burnTicks('charcoal'), 1600)
  assert.strictEqual(burnTicks('oak_planks'), 300) // one and a half
  assert.strictEqual(burnTicks('dark_oak_log'), 300)
  assert.strictEqual(burnTicks('stick'), 100) // half an item
  assert.strictEqual(burnTicks('cobblestone'), 0)
  assert.strictEqual(burnTicks('raw_iron'), 0)
})

test('armor goes where it is worn', async () => {
  const place = item => Object.keys(WORN).find(slot => WORN[slot].test(item))
  assert.deepStrictEqual(['iron_helmet', 'diamond_chestplate', 'iron_leggings', 'leather_boots'].map(place), ['head', 'torso', 'legs', 'feet'])
  assert.strictEqual(place('iron_pickaxe'), undefined)
  const equipped = []
  const bot = {
    inventory: { items: () => [{ name: 'iron_helmet' }, { name: 'iron_pickaxe' }] },
    equip: async (item, slot) => { equipped.push([item.name, slot]) }
  }
  assert.deepStrictEqual(await ACTIONS.equip(bot, { item: 'iron_helmet' }), { equipped: 'iron_helmet', slot: 'head' })
  assert.deepStrictEqual(await ACTIONS.equip(bot, { item: 'iron_pickaxe' }), { equipped: 'iron_pickaxe', slot: 'hand' })
  await assert.rejects(ACTIONS.equip(bot, { item: 'iron_pickaxe', slot: 'head' }), /cannot be worn on the head/)
  await assert.rejects(ACTIONS.equip(bot, { item: 'iron_helmet', slot: 'torso' }), ActionError)
  assert.deepStrictEqual(equipped, [['iron_helmet', 'head'], ['iron_pickaxe', 'hand']])
})

test('a window that does not open is an error the agent can read, and is closed if it opens late', async () => {
  const real = setTimeout
  global.setTimeout = (run, ms) => real(run, ms >= 1000 ? 20 : ms) // (four seconds pass in twenty milliseconds)
  try {
    let closed = false
    const late = new Promise(resolve => real(() => resolve({ close: () => { closed = true } }), 60))
    await assert.rejects(opened(late, 'chest'), /the chest does not open/)
    await late
    await new Promise(resolve => real(resolve, 5))
    assert.strictEqual(closed, true)
    const window = { close: () => { throw new Error('closed a window that opened in time') } }
    assert.strictEqual(await opened(Promise.resolve(window), 'chest'), window)
  } finally { global.setTimeout = real }
})

test('a ray along an axis from a whole coordinate stops at the wall', () => {
  const blocks = world()
  blocks.carve(0, 4, 64, 65, 0, 0) // a corridor east from the origin; stone beyond x = 4
  const bot = botIn(blocks, new Vec3(0.5, 64, 0.5))
  const hit = firstHit(bot, new Vec3(1, 64.5, 0.5), new Vec3(1, 0, 0), 24)
  assert.deepStrictEqual([hit.name, hit.position.x], ['stone', 5])
  const up = firstHit(bot, new Vec3(2, 64, 0), new Vec3(0, 1, 0), 24) // every coordinate whole
  assert.deepStrictEqual([up.name, up.position.y], ['stone', 66])
})
