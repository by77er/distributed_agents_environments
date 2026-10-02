'use strict'
// The game data, corrected: how long blocks take to break with the right tool.

const test = require('node:test')
const assert = require('node:assert')
const fs = require('node:fs')
const path = require('node:path')
const { fixMaterials } = require('../lib/data')

// The game version the servers run: the Python side names it, and gives it to the harness when it connects.
const paper = fs.readFileSync(path.join(__dirname, '../../minecraft_swarm/paper.py'), 'utf8')
const VERSION = /^PAPER_VERSION = "([^"]+)"$/m.exec(paper)[1]

test('pickaxes are as fast on ores and obsidian as in the game', () => {
  const registry = require('prismarine-registry')(VERSION)
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
