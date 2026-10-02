'use strict'
// Corrections to the game data the bots run on.

// Blocks that need a better pickaxe than a wooden one (most ores, obsidian, ancient debris, blocks of metal) come
// from minecraft-data with the material `incorrect_for_wooden_tool` (or `_stone_`, `_iron_`), for which it has no
// table of tool speeds. A bot then thinks its pickaxe no faster than a hand: diamond ore takes it 6.75 seconds with an
// iron pickaxe instead of 1.15, obsidian 75 seconds with a diamond one instead of 9.4. They are all mined with
// pickaxes; which pickaxe can harvest them is a separate table (`harvestTools`), left as it is.
function fixMaterials (registry) {
  let fixed = 0
  for (const block of registry.blocksArray) {
    if (typeof block.material === 'string' && block.material.startsWith('incorrect_for_')) {
      block.material = 'mineable/pickaxe'
      fixed++
    }
  }
  return fixed
}

module.exports = { fixMaterials }
