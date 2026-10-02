'use strict'
// What a bot is told at its next observation besides what it sees: what its teammates said, and that it died.
//
// Telling uses nothing up: an observation asked for again answers the same (and adds what has arrived since), so one
// repeated after a failure loses nothing. What has been told is dropped when the game next runs (`forget`).

class News {
  constructor (limit) {
    this.limit = limit // messages kept; the oldest goes first
    this.messages = []
    this.died = false
    this.told = { messages: 0, died: false }
  }

  hear (from, message) {
    this.messages.push({ from, message })
    if (this.messages.length > this.limit) {
      this.messages.shift()
      this.told.messages = Math.max(0, this.told.messages - 1)
    }
  }

  // The news as an observation gives it.
  tell () {
    this.told = { messages: this.messages.length, died: this.died }
    return { messages: [...this.messages], died: this.died }
  }

  // Drop what observations have told; what arrived after the last of them stays for the next.
  forget () {
    this.messages.splice(0, this.told.messages)
    if (this.told.died) this.died = false
    this.told = { messages: 0, died: false }
  }
}

module.exports = { News }
