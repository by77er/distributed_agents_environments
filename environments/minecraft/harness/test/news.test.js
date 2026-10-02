'use strict'
// What a bot is told besides what it sees: an observation asked for twice tells the same, and what was told goes
// when the game next runs.

const test = require('node:test')
const assert = require('node:assert')
const { News } = require('../lib/news')

test('an observation repeated tells the same, and a thaw drops what was told', () => {
  const news = new News(20)
  news.hear('ben', 'I go east')
  news.died = true
  const first = news.tell()
  assert.deepStrictEqual(first, { messages: [{ from: 'ben', message: 'I go east' }], died: true })
  assert.deepStrictEqual(news.tell(), first) // nothing was used up
  news.hear('cy', 'ore here') // said after the observation, before the game runs
  news.forget()
  assert.deepStrictEqual(news.tell(), { messages: [{ from: 'cy', message: 'ore here' }], died: false })
  news.forget()
  assert.deepStrictEqual(news.tell(), { messages: [], died: false })
})

test('what was never told is kept through a thaw', () => {
  const news = new News(20)
  news.forget()
  news.hear('ben', 'one')
  news.died = true
  news.forget() // no observation since: nothing was told
  assert.deepStrictEqual(news.tell(), { messages: [{ from: 'ben', message: 'one' }], died: true })
})

test('the oldest messages go first when there are too many, told or not', () => {
  const news = new News(3)
  for (const message of ['one', 'two']) news.hear('ben', message)
  news.tell()
  for (const message of ['three', 'four']) news.hear('ben', message) // `one` goes: it was told
  news.forget() // and so does `two`
  assert.deepStrictEqual(news.tell().messages.map(said => said.message), ['three', 'four'])
  for (const message of ['five', 'six']) news.hear('ben', message)
  news.forget()
  assert.deepStrictEqual(news.tell().messages.map(said => said.message), ['five', 'six'])
})
