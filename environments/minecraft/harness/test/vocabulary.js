'use strict'
// Prints what the harness takes and keeps, as JSON (for the Python side's tests, which hold the prompts against it):
// the actions it handles, the slots and directions they take, the blocks it treats as containers and as furnaces, how
// far it sees large things, and the limits it read.
const { ACTIONS, SLOTS } = require('../lib/actions')
const { DIRECTIONS, CONTAINERS, FURNACES, FAR_RANGE } = require('../lib/observe')
const { LIMITS } = require('../lib/limits')

process.stdout.write(JSON.stringify({
  actions: Object.keys(ACTIONS),
  slots: SLOTS,
  directions: Object.keys(DIRECTIONS),
  containers: [...CONTAINERS],
  furnaces: [...FURNACES],
  far_range: FAR_RANGE,
  limits: LIMITS
}))
