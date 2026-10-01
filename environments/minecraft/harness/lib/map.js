'use strict'
// A map of what a bot has seen around itself: for each height near it, a grid of the blocks it has seen there.
//
// Only what its own eyes saw is in it: `memory` holds the blocks its rays hit or passed through, `air` the empty cells
// they passed through. A cell neither holds is unknown, whatever the client's world data says.

const AIR = new Set(['air', 'cave_air', 'void_air', 'light']) // (`light` is the invisible block that lights staged rooms)
const RADIUS = 6 // blocks each way: a 13 by 13 grid
const HEIGHTS = [2, 1, 0, -1, -2] // relative to the feet: above the head, head, feet, floor, under the floor
const MAX_AIR = 150000 // remembered empty cells per bot
const KEEP_AIR = 32 // when that is exceeded: keep those within this many blocks

function key (x, y, z) { return `${x},${y},${z}` }

// { center, radius, palette: [{ name, solid }], layers: [{ dy, y, cells }] }: `cells` holds, row by row from north to
// south and west to east within a row, an index into `palette`, or -1 for a cell the bot has not seen.
function localMap (bot, memory, air, radius = RADIUS, heights = HEIGHTS) {
  const here = bot.entity.position.floored()
  const palette = []
  const indices = new Map()
  const index = name => {
    let found = indices.get(name)
    if (found === undefined) {
      found = palette.length
      palette.push({ name, solid: bot.registry.blocksByName[name]?.boundingBox === 'block' })
      indices.set(name, found)
    }
    return found
  }
  const layers = heights.map(dy => {
    const cells = []
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const at = key(here.x + dx, here.y + dy, here.z + dz)
        const name = memory.get(at) ?? (air.has(at) ? 'air' : undefined)
        cells.push(name === undefined ? -1 : index(name))
      }
    }
    return { dy, y: here.y + dy, cells }
  })
  return { center: { x: here.x, y: here.y, z: here.z }, radius, palette, layers }
}

// Record a cell a ray passed through (empty, or something see-through such as water).
function remember (memory, air, x, y, z, name) {
  const at = key(x, y, z)
  if (AIR.has(name)) {
    air.add(at)
    memory.delete(at) // what was there is gone
  } else {
    memory.set(at, name)
    air.delete(at)
  }
}

// Forget far empty cells once there are too many (solid blocks are kept: there are far fewer of them).
function trim (bot, air) {
  if (air.size <= MAX_AIR) return
  const here = bot.entity.position
  for (const at of air) {
    const [x, y, z] = at.split(',').map(Number)
    if (Math.abs(x - here.x) > KEEP_AIR || Math.abs(y - here.y) > KEEP_AIR || Math.abs(z - here.z) > KEEP_AIR) air.delete(at)
  }
}

module.exports = { localMap, remember, trim, AIR, RADIUS, HEIGHTS }
