'use strict'
// The game data, corrected: how long blocks take to break with the right tool.

const test = require('node:test')
const assert = require('node:assert')
const { fixMaterials } = require('../lib/data')

test('pickaxes are as fast on ores and obsidian as in the game', () => {
  const registry = require('prismarine-registry')('1.21.11')
  const Block = require('prismarine-block')(registry)
  fixMaterials(registry)
  const seconds = (block, tool) => {
    const item = tool ? registry.itemsByName[tool].id : null
    return new Block(registry.blocksByName[block].id, 0, 0).digTime(item, false, false, false, [], []) / 1000
  }
  assert.strictEqual(seconds('obsidian', 'diamond_pickaxe'), 9.4)
  assert.strictEqual(seconds('deepslate_diamond_ore', 'iron_pickaxe'), 1.15)
  assert.strictEqual(seconds('iron_ore', 'stone_pickaxe'), 1.15)
  assert.strictEqual(seconds('obsidian', null), 250) // by hand: minutes, and nothing drops
  assert.strictEqual(seconds('stone', 'wooden_pickaxe'), 1.15) // (blocks that were right stay right)
  assert.strictEqual(fixMaterials(registry), 0) // nothing is left to correct
})
