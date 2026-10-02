'use strict'
// The limits that the actions keep and the prompts state: ../../limits.json, which the Python side reads too
// (minecraft_swarm/limits.py), so that what agents are told of an action is what it does.
//
//   reach_blocks      how far from its eyes a bot mines, places and uses
//   move_blocks       the most blocks one `move` walks
//   walk_dig_seconds  walking digs through a block only if the bot's tools break it within this
//   scaffolding       the blocks walking bridges and pillars with
//   wait_seconds      how long `wait` waits
//   smelt_seconds     what a furnace takes for one item
//   fuels             the fuels agents are told of, as they are named to them
//   window_seconds    the most game time a turn's window runs: an action that takes longer is cut off there
//   chat_characters   the length a chat message is cut to

const LIMITS = require('../../limits.json')

const TICKS_PER_SECOND = 20
const NUMBERS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
  'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']

// A small whole number as a sentence has it ("five seconds").
function spelled (number) { return NUMBERS[number] ?? String(number) }

// Names as a sentence lists them: "a, b and c" (or "a, b or c").
function listed (names, last) {
  return names.length < 2 ? names.join('') : `${names.slice(0, -1).join(', ')} ${last} ${names.at(-1)}`
}

module.exports = { LIMITS, TICKS_PER_SECOND, spelled, listed }
