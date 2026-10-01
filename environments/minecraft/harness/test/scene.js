'use strict'
// Prints what a bot observes in the made-up world of map.test.js, as JSON (for the Python side's tests).
const { Vec3 } = require('vec3')
const { observe } = require('../lib/observe')
const { scene, botIn } = require('./world')

const bot = botIn(scene(), new Vec3(0.5, 64, 0.5))
bot.entities = {
  1: { type: 'player', username: 'ben', position: new Vec3(2.5, 64, -1.5), height: 1.8 },
  2: { name: 'item', position: new Vec3(-2.5, 64, 2.5), height: 0.25, getDroppedItem: () => ({ name: 'diamond', count: 3 }) }
}
process.stdout.write(JSON.stringify(observe(bot, new Set(['ada', 'ben']), new Map(), new Set())))
