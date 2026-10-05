# Minecraft memory

**Status: built.** The memory of the Minecraft team's worlds, measured on real servers and episodes, and the settings
chosen from the measurements: G1 held to the heap the live set needs, two malloc arenas, uncompressed packets and a
smaller young generation for Node. Several worlds in one Paper server was considered and is not built; the reasons are
[below](#several-worlds-in-one-server). A design note: see [Design notes](README.md) for the others.

Code: `minecraft_team.paper` (`HEAP`, `JVM`, `JAVA_ENVIRONMENT`), `minecraft_team.harness` (`NODE_FLAGS`),
`config/server.properties` · See [the Minecraft team](../products/minecraft-team.md#the-worlds), [the cluster
config](../guide/cluster.md)

A world is a Paper server of its own and a Node process with one mineflayer bot per agent; the Python side is the
episode and the bridge to both. A pool of worlds (`[sandboxes.minecraft]`) runs `size` of them at once, and a run's
demand counts each pool's `size` times its `memory_gib`
([what a run needs](../libraries/rollout-train/launching.md#what-a-run-needs)).

## How it was measured

| | |
|---|---|
| Machine | WSL2, 20 cores, 23 GB; OpenJDK 21.0.11, Node 24, Paper 1.21.11 build 132 |
| Worlds | Servers copied from the template of seed 12345, two at a time at most |
| Episodes | The episode program (`TeamEpisode`) through a local runner and a pool of the worlds, four bots, with a scripted policy: each agent `move`s 32 blocks a turn in a direction of its own, turning every fourth turn. Four bots walking apart load the most chunks a team can |
| Tasks | `t001` (diamonds in a staged room: it ends once they are held, after about 40 seconds), `t046` (beside a nether fortress), `t037` (the surface, normal difficulty), 15 turns each; `t037` and `t057` (the game, from a bare spawn) for 40 turns, by when the bots have walked far past the template's generated area |
| Sampled | Every second, each process's resident memory (RSS, its anonymous and file parts) and proportional share (PSS); every 15 seconds each server's heap (`jcmd GC.heap_info`) and native memory (`jcmd VM.native_memory summary`, with native memory tracking on, which holds 8 to 10 MiB itself); each Node process's heap, from `process.memoryUsage()` every 5 seconds. The live set: a full collection forced every 15 seconds, in runs of their own |
| Before | The code at the start of this work: G1 at its defaults, `-Xmx1536M`, packets compressed, Node at its defaults |

Figures are MiB of peak RSS unless they say otherwise.

## Before

| | Paper | Node | A world |
|---|---|---|---|
| A server after loading its world, no players | 956 to 1,107 | | |
| `t001`, a staged room | 1,363 | 329 | 1,692 |
| `t046`, beside a fortress | 1,634 | 515 | 2,149 |
| `t037`, the surface | 1,973 to 2,106 | 559 to 650 | 2,620 to 2,670 |
| `t037` and `t057`, 40 turns, roaming | 1,541 and 2,133 | 643 and 621 | 2,184 and 2,754 |

The cluster config counted 1.5 GiB a world; every episode measured used more.

Where a server's memory went, at its peak in `t037`:

| Part | MiB | What it is |
|---|---|---|
| Heap committed | 985 to 1,089 | G1 grows toward `-Xmx` as allocation runs; 550 to 990 of it in use, of which the live set is 250 to 560 in 15 turns, and up to 1,030 for bots roaming apart through new terrain |
| Outside the JVM's accounting | 250 to 540 | glibc's malloc arenas: up to eight a core, each keeping native memory once freed (the JIT compiler's arenas peak near 90 MiB and are freed) |
| Classes and metaspace | 165 | 29,500 classes |
| G1's own structures | 80 to 90 | Remembered sets, card table, mark bitmaps |
| Compiled code | 70 to 85 | |
| Symbols | 50 | |
| Threads and the rest | 30 to 40 | 96 threads |

A server's live set after loading its world is 214 MiB: the game's registries, recipes, advancements, loot tables and
structure templates. The rest is chunks. In 15 turns the live set grew to 420 MiB beside the fortress and to 560 MiB
on the surface; four bots walking apart through terrain the template does not hold keep about 9,000 chunks in memory
while the terrain around them is generated, and their server's live set reached 1 GiB in twelve minutes.

The Node process holds 120 to 150 MiB before its bots join (mineflayer, its protocol and the game's data for the
version), then each bot's chunks (30 to 125 MiB of buffers for four), the bots' memories of what they saw, and V8's
heap grown well past what is in use. The Python side is 26 to 46 MiB before an episode and 15 to 20 MiB more for each
episode running: its records, the latest observations and the bridges.

## What changed

| Change | Where | What it saves | Gameplay |
|---|---|---|---|
| Two malloc arenas (`MALLOC_ARENA_MAX=2`) | `JAVA_ENVIRONMENT` | Most of the 250 to 540 MiB outside the JVM's accounting: 90 to 140 MiB remain | None |
| G1 held to the heap the live set needs: `-XX:GCTimeRatio=4` (grow the heap only while collecting takes more than a fifth of the time; G1's default is a thirteenth), `-XX:MinHeapFreeRatio=20`, `-XX:MaxHeapFreeRatio=40` | `JVM` | 200 to 300 MiB of heap committed in 15-turn episodes; free heap is given back after a collection | None: pauses as short as before or shorter (most under 4 ms), no full collection |
| Packets not compressed (`network-compression-threshold=-1`) | `config/server.properties` | With the next row, 115 to 220 MiB of the Node process: zlib's buffers on both sides, and their time. The bots are on the same machine | None |
| Node's young generation 4 MiB a half, not 16 (`--max-semi-space-size=4`) | `NODE_FLAGS` | See the row above | None: young collections are more frequent and each smaller |
| Generating a template and the bootstrap server run with the servers' settings and heap (they had heaps of 2 and 1 GiB, and G1 at its defaults) | `Installation.command` | Generating a template peaked at 1,305 MiB (25 seconds). The first episodes of several new seeds generate theirs at once | None |

`HEAP` stays 1536 MiB: four bots roaming apart keep up to 1 GiB live.

## After

| | Paper | Node | A world |
|---|---|---|---|
| A server after loading its world, no players | 882 (800 after a collection) | | |
| `t001`, a staged room | 908 | 212 | 1,120 |
| `t046`, beside a fortress | 1,005 to 1,206 | 264 to 296 | 1,280 to 1,500 |
| `t037`, the surface | 1,244 to 1,336 | 534 to 539 | 1,780 to 1,870 |
| `t037` and `t057`, 40 turns, roaming | 1,456 and 1,843 | 603 and 571 | 2,059 and 2,414 |

Where a task was played more than once with these settings (three times for the servers, twice for Node), the table
gives the range: the bots' paths differ from run to run, and so does the terrain they load. A world takes 0.55 to
0.85 GiB less in 15-turn episodes: a staged task about 1.1 GiB, a task in the nether 1.25 to 1.45 GiB, four bots walking
apart on the surface 1.75 to 1.85 GiB. Bots that roam for long take 0.1 to 0.35 GiB less: most of their memory is
their server's live set, which no setting here shrinks. The longest pause was 20 ms, with no full collection, against
25 to 190 ms before.

### A pool

Each world is a JVM and a Node process of its own: what they share is the JDK's and Paper's files (34 MiB of each
server's RSS is file pages, and its PSS is within 20 MiB of its RSS), so a pool's memory is its `size` times a world's,
and the driver's Python 15 to 20 MiB more for each episode.

| Pool of 6 (the chart's `size`) | Before | After |
|---|---|---|
| Staged tasks | 9.9 GiB | 6.6 GiB |
| Survival, 15 turns | 12.6 to 15.6 GiB | 7.5 to 11 GiB |
| The cluster config's `memory_gib` | 1.5 a world, 9 GiB | 1.75 a world, 10.5 GiB |

`memory_gib = 1.75` is about the peak of four bots walking apart on the surface: above every staged task and the
nether, below bots that roam for long. The cluster config's `[guards] runs_gib` keeps a runner from claiming another
episode while less than that is free, which covers the peaks a pool's count does not.

## What was considered and not done

### Several worlds in one server

Every world in one Paper server, each with its own players, would hold the server's fixed part once: the 214 MiB live
after loading, about 330 MiB of classes, symbols, code and the collector's structures, and the Node process's 150 MiB
of modules if the bots shared one. Over a pool of 6 that is up to 2.5 GiB. It is not built, because an episode's
isolation would not hold:

- **Time.** Each episode freezes its world while its agents think and runs it for a window when they have acted, on
  its own schedule. Paper's tick manager (`ServerTickManager`), which the plugin freezes, steps and runs, is the
  server's: there is no freezing one world while another runs. A world left running while its agents think is a
  different game (a creeper walks up).
- **One main thread.** One thread ticks every world of a server: a world generating terrain, or one episode's lag,
  slows every other, while each episode's bots act in real time.
- **Players are the server's.** Names are drawn for each start from the same list, and two episodes with an `ada`
  cannot both join. Chat and the player list are the server's.
- **Dimensions.** Each episode needs an overworld, a nether and an end of its own, with portals linked between them
  and a dragon fight of its own; Paper links portals between the main world and its own nether and end only.
- **The plugin.** Its state is the server's: one team, one baseline of advancements, one log of events, one dragon.

### Others

| Considered | Why not |
|---|---|
| Smaller view and simulation distance | 6 chunks is the least that holds the 96 blocks the dragon, end crystals and ghasts are seen from (`tests/test_agreement.py` holds it), and mobs spawn and furnaces smelt only within the simulation distance |
| A smaller view distance asked for by the bots in the overworld, where nothing is seen beyond 24 blocks | Halves the Node process's chunks, but the pathfinder plans only through chunks the bot holds: a `move_to` 64 to 96 blocks away would plan differently |
| A world border at the template's edge | Bounds the roaming that costs most, but the tasks go thousands of blocks (strongholds, the nether) |
| No spawn chunks, entity and tick settings | The spawn chunks are 25 at the origin; the rest changes the game |
| Serial collector | Less than G1 at its defaults, but up to 200 MiB more than G1 held to its live set, and full collections of 0.3 to 0.7 seconds, three or four in a 15-turn episode: the server stops for up to 14 ticks while the bots act in real time |
| Parallel collector (two threads) | More memory than either (its sizing grows the heap), 62 MiB of its own structures, and a young collection of 2.7 seconds |
| Class data sharing (a dynamic archive of Paper's classes) | Metaspace and symbols fall by 140 MiB, but the archive's mapping, 136 MiB, takes their place: two idle servers' PSS was 937 and 907 MiB without it, 943 and 942 with it, mapped at the address it was made for (a diagnostic option). At JDK 21's default, a random address, every page of it is rewritten and each process's own |
| One Node process for several worlds | Saves 150 MiB a world, but one event loop for every bot: the pathfinder plans for up to two seconds at a time, and every bot of every world would stop while it does; and one crash would end every episode |
| A cap on Node's heap | The bots' memories of what they saw grow as they explore (V8's heap reached 350 MiB in use); a cap would end long episodes |
| Python | Nothing large is held per episode |
