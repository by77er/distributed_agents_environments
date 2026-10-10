package dev.rollout.groundtruth;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.papermc.paper.event.player.AsyncChatEvent;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import org.bukkit.Bukkit;
import org.bukkit.Difficulty;
import org.bukkit.GameMode;
import org.bukkit.GameRule;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.ServerTickManager;
import org.bukkit.World;
import org.bukkit.attribute.Attribute;
import org.bukkit.attribute.AttributeInstance;
import org.bukkit.block.Block;
import org.bukkit.block.BlockState;
import org.bukkit.block.Container;
import org.bukkit.block.DoubleChest;
import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityPickupItemEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerCommandPreprocessEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.InventoryHolder;
import org.bukkit.inventory.InventoryView;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Ground truth and tick control for RL episodes, behind an HTTP API on 127.0.0.1 only (port from the system property
 * {@code rollout.control.port}). Agents never see this API: it is for the worlds ({@code minecraft_team.worlds}) and
 * the servers ({@code minecraft_team.paper}).
 *
 * <ul>
 *   <li>GET /health: ready, and who is online.</li>
 *   <li>GET /state: every player's position, health, food, inventory and diamonds; the team's total diamonds, the
 *       advancements it earned since the baseline, what it got hold of since then (picked up, crafted, smelted), and
 *       the most the dragon was hurt.</li>
 *   <li>GET /tick, POST /tick {"action": "freeze" | "step" | "run" | "stop", "ticks": n}: a step runs n
 *       ticks of a frozen game and answers when they have run; a run starts n ticks and answers at once, and a stop
 *       ends it early and says how many ran. While the game is frozen and not stepping, team members are held as
 *       they were: the game itself does not freeze players, whose hunger, air, fire and health would run on in
 *       real time while agents think.</li>
 *   <li>POST /episode: set up the team (one scoreboard team without friendly fire; clear, kit with armor worn,
 *       teleport to any dimension, respawn there, game mode) and the worlds (difficulty, time, rules).</li>
 *   <li>GET /ores?x&amp;y&amp;z&amp;radius: diamond ores near a point, for choosing starts (never agents).</li>
 *   <li>GET /events?after=n: what happened (chat, ores mined, items picked up, deaths, joins, advancements, hits on
 *       creatures, moves the server refused, the dragon's death).</li>
 *   <li>POST /baseline: remember each team member's advancements now; /state then reports only newer ones (a kit can
 *       itself grant advancements, which an episode should not be rewarded for).</li>
 *   <li>Setup, for building tasks: POST /setup/carve (a lit, empty box with a floor, sealed against liquids and
 *       falling blocks), /setup/items (dropped items), /setup/chest (a chest with contents), /setup/block (one
 *       block), /setup/spawn (a creature), /setup/food (a player's hunger), /setup/generate (generate
 *       the overworld's chunks around the origin and save them, for a server template);
 *       GET /setup/stand (safe places to stand), /setup/surface (the ground's height), /setup/locate (the nearest
 *       structure), /setup/blocks (blocks of one type near a point).</li>
 * </ul>
 *
 * Team members cannot run commands: every command they send is cancelled.
 */
public final class GroundTruthPlugin extends JavaPlugin implements Listener {
    private static final int MAX_EVENTS = 10_000;
    private static final int MAX_ORE_RADIUS = 64;

    private final Gson gson = new GsonBuilder().disableHtmlEscaping().create();
    private final Set<String> team = ConcurrentHashMap.newKeySet();
    private final Map<String, Integer> lastKnownDiamonds = new ConcurrentHashMap<>();
    /** What each team member held when they left, item by item (`/holdings`). */
    private final Map<String, Map<String, Integer>> lastKnownHeld = new ConcurrentHashMap<>();
    /** Where team members placed containers since the baseline: what is stored in them is the team's. */
    private final Set<Location> placedContainers = ConcurrentHashMap.newKeySet();
    private final Map<String, Set<String>> baselines = new ConcurrentHashMap<>();
    private final Set<String> teamEarned = ConcurrentHashMap.newKeySet();
    private final Map<String, Integer> teamObtained = new ConcurrentHashMap<>();
    private final Map<String, Long> lastFailedMove = new ConcurrentHashMap<>();
    private final Map<java.util.UUID, Held> held = new HashMap<>();
    private long runStarted = 0;
    private volatile boolean dragonKilled = false;
    private volatile double dragonDamage = 0.0;
    private final ConcurrentLinkedDeque<JsonObject> events = new ConcurrentLinkedDeque<>();
    private final AtomicLong eventSequence = new AtomicLong();
    private HttpServer http;

    @FunctionalInterface
    private interface Route {
        JsonElement handle(String method, Map<String, String> query, JsonObject body) throws Exception;
    }

    @Override
    public void onEnable() {
        int port = Integer.getInteger("rollout.control.port", 0);
        if (port == 0) {
            getLogger().severe("set -Drollout.control.port to enable the control API");
            return;
        }
        try {
            http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 64);
        } catch (IOException error) {
            throw new IllegalStateException("cannot listen on 127.0.0.1:" + port, error);
        }
        http.setExecutor(Executors.newFixedThreadPool(8));
        route("/health", this::health);
        route("/state", this::state);
        route("/holdings", this::holdings);
        route("/tick", this::tick);
        route("/episode", this::episode);
        route("/ores", this::ores);
        route("/events", this::events);
        route("/baseline", this::baseline);
        route("/setup/carve", this::carve);
        route("/setup/items", this::dropItems);
        route("/setup/chest", this::chest);
        route("/setup/block", this::setBlock);
        route("/setup/stand", this::standingSpots);
        route("/setup/surface", this::surface);
        route("/setup/locate", this::locate);
        route("/setup/blocks", this::findBlocks);
        route("/setup/spawn", this::spawnEntity);
        route("/setup/food", this::setFood);
        route("/setup/generate", this::generate);
        getServer().getPluginManager().registerEvents(this, this);
        Bukkit.getScheduler().runTaskTimer(this, this::holdPlayers, 1L, 1L);  // (the scheduler runs while frozen)
        http.start();
        getLogger().info("control API on 127.0.0.1:" + port);
    }

    @Override
    public void onDisable() {
        if (http != null) {
            http.stop(0);
        }
    }

    // Routes

    private JsonElement health(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            JsonObject result = new JsonObject();
            result.addProperty("ready", true);
            result.addProperty("version", Bukkit.getMinecraftVersion());
            result.addProperty("server", System.getProperty("rollout.server.name", ""));
            JsonArray online = new JsonArray();
            for (Player player : Bukkit.getOnlinePlayers()) {
                online.add(player.getName());
            }
            result.add("online", online);
            return result;
        });
    }

    private JsonElement state(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            JsonObject result = new JsonObject();
            result.add("tick", tickState());
            JsonArray players = new JsonArray();
            int teamDiamonds = 0;
            Set<String> counted = new java.util.HashSet<>();
            Set<String> teamAdvancements = new java.util.TreeSet<>(teamEarned);  // members who left keep theirs
            for (Player player : Bukkit.getOnlinePlayers()) {
                int diamonds = diamonds(player);
                String name = player.getName().toLowerCase(Locale.ROOT);
                JsonObject described = describe(player, diamonds);
                if (team.contains(name)) {
                    teamDiamonds += diamonds;
                    counted.add(name);
                    Set<String> earned = newAdvancements(player);
                    teamAdvancements.addAll(earned);
                    JsonArray list = new JsonArray();
                    earned.forEach(list::add);
                    described.add("advancements", list);
                }
                players.add(described);
            }
            for (String member : team) {  // members who left keep what they held when they left
                if (!counted.contains(member)) {
                    teamDiamonds += lastKnownDiamonds.getOrDefault(member, 0);
                }
            }
            result.add("players", players);
            result.addProperty("team_diamonds", teamDiamonds);
            JsonArray advancements = new JsonArray();
            teamAdvancements.forEach(advancements::add);
            result.add("team_advancements", advancements);
            JsonObject obtained = new JsonObject();
            new java.util.TreeMap<>(teamObtained).forEach(obtained::addProperty);
            result.add("team_obtained", obtained);
            result.addProperty("dragon_killed", dragonKilled);
            result.addProperty("dragon_damage", dragonKilled ? 1.0 : dragonDamage);
            JsonArray members = new JsonArray();
            team.forEach(members::add);
            result.add("team", members);
            return result;
        });
    }

    /**
     * Every item the team holds, by its id: what members carry (`held`: inventory with armor and offhand, cursor and
     * crafting grid; members who left, what they held then) and what is stored in the containers members placed since
     * the baseline (`stored`; a double chest counted once), with how many of those containers still stand.
     */
    private JsonElement holdings(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            Map<String, Integer> carried = new TreeMap<>();
            Set<String> counted = new java.util.HashSet<>();
            for (Player player : Bukkit.getOnlinePlayers()) {
                String name = player.getName().toLowerCase(Locale.ROOT);
                if (team.contains(name)) {
                    counted.add(name);
                    held(player).forEach((item, count) -> carried.merge(item, count, Integer::sum));
                }
            }
            for (String member : team) {
                if (!counted.contains(member)) {
                    lastKnownHeld.getOrDefault(member, Map.of()).forEach((item, count) -> carried.merge(item, count, Integer::sum));
                }
            }
            Map<String, Integer> stored = new TreeMap<>();
            Set<String> seen = new java.util.HashSet<>();
            int containers = 0;
            for (Location at : placedContainers) {
                BlockState state = at.getBlock().getState();
                if (!(state instanceof Container container)) {
                    continue;  // broken, or replaced by something else
                }
                Inventory inventory = container.getInventory();
                InventoryHolder holder = inventory.getHolder();
                Location where = holder instanceof DoubleChest both ? both.getLocation() : at;
                if (!seen.add(where.getWorld().getName() + ":" + where.getX() + ":" + where.getY() + ":" + where.getZ())) {
                    continue;  // the other half of a double chest
                }
                containers++;
                for (ItemStack stack : inventory.getContents()) {
                    count(stored, stack);
                }
            }
            JsonObject result = new JsonObject();
            JsonObject heldJson = new JsonObject();
            carried.forEach(heldJson::addProperty);
            result.add("held", heldJson);
            JsonObject storedJson = new JsonObject();
            stored.forEach(storedJson::addProperty);
            result.add("stored", storedJson);
            result.addProperty("containers", containers);
            return result;
        });
    }

    private JsonElement tick(String method, Map<String, String> query, JsonObject body) throws Exception {
        ServerTickManager ticks = Bukkit.getServerTickManager();
        if (method.equals("GET")) {
            return onMainThread(this::tickState);
        }
        String action = body.has("action") ? body.get("action").getAsString() : "";
        switch (action) {
            case "freeze" -> onMainThread(() -> { ticks.setFrozen(true); return null; });
            case "step" -> {
                int count = body.get("ticks").getAsInt();
                if (count < 1 || count > 20 * 60 * 10) {
                    throw new IllegalArgumentException("ticks must be between 1 and 12000");
                }
                long before = onMainThread(() -> overworld().getGameTime());
                boolean started = onMainThread(() -> {
                    if (!ticks.isFrozen()) {
                        ticks.setFrozen(true);
                    }
                    releasePlayers();
                    return ticks.stepGameIfFrozen(count);
                });
                if (!started) {
                    throw new IllegalStateException("the game could not step");
                }
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5 + count / 5);
                while (onMainThread(() -> ticks.isStepping() || ticks.getFrozenTicksToRun() > 0)) {
                    if (System.nanoTime() > deadline) {
                        throw new IllegalStateException("stepping did not finish in time");
                    }
                    Thread.sleep(5);
                }
                JsonObject result = onMainThread(this::tickState);
                result.addProperty("stepped", onMainThread(() -> overworld().getGameTime()) - before);
                return result;
            }
            case "run" -> {  // start the ticks and answer at once: whoever asked watches the bots and stops it
                int count = body.get("ticks").getAsInt();
                if (count < 1 || count > 20 * 60 * 10) {
                    throw new IllegalArgumentException("ticks must be between 1 and 12000");
                }
                boolean started = onMainThread(() -> {
                    if (!ticks.isFrozen()) {
                        ticks.setFrozen(true);
                    }
                    releasePlayers();
                    runStarted = overworld().getGameTime();
                    return ticks.stepGameIfFrozen(count);
                });
                if (!started) {
                    throw new IllegalStateException("the game could not step");
                }
            }
            case "stop" -> {
                return onMainThread(() -> {
                    ticks.stopStepping();
                    JsonObject result = tickState();
                    result.addProperty("stepped", overworld().getGameTime() - runStarted);
                    return result;
                });
            }
            default -> throw new IllegalArgumentException("action must be freeze, step, run or stop");
        }
        return onMainThread(this::tickState);
    }

    // Players while the game is frozen. A tick freeze stops the world and its creatures, not players: they would
    // go hungry, heal, drown and burn in real time while the agents think (half a minute a turn, a quarter of an
    // hour during an update). So when the stepping ends each team member is noted as it is, and held so until the
    // game steps again.

    private record Held(int air, int fire, int food, float saturation, float exhaustion,
            java.util.Collection<org.bukkit.potion.PotionEffect> effects) {}

    private static boolean paused() {
        ServerTickManager ticks = Bukkit.getServerTickManager();
        return ticks.isFrozen() && !ticks.isStepping();
    }

    private boolean heldStill(org.bukkit.entity.Entity entity) {
        return entity instanceof Player player && paused() && team.contains(player.getName().toLowerCase(Locale.ROOT));
    }

    /** Every server heartbeat: note the team when the game stops stepping, and keep it as noted while it is paused. */
    private void holdPlayers() {
        if (!paused()) {
            held.clear();
            return;
        }
        for (Player player : Bukkit.getOnlinePlayers()) {
            if (!team.contains(player.getName().toLowerCase(Locale.ROOT)) || player.isDead()) {
                continue;
            }
            Held was = held.get(player.getUniqueId());
            if (was == null) {
                held.put(player.getUniqueId(), new Held(player.getRemainingAir(), player.getFireTicks(),
                        player.getFoodLevel(), player.getSaturation(), player.getExhaustion(),
                        new java.util.ArrayList<>(player.getActivePotionEffects())));
                continue;
            }
            player.setRemainingAir(was.air());
            player.setFireTicks(was.fire());
            player.setFoodLevel(was.food());
            player.setSaturation(was.saturation());
            player.setExhaustion(was.exhaustion());
        }
    }

    /** As the game is about to step: effects get back the time they had when it stopped. */
    private void releasePlayers() {
        for (Player player : Bukkit.getOnlinePlayers()) {
            Held was = held.remove(player.getUniqueId());
            if (was == null || player.isDead()) {
                continue;
            }
            for (org.bukkit.potion.PotionEffect effect : was.effects()) {
                player.addPotionEffect(effect, true);
            }
        }
    }

    @EventHandler(priority = EventPriority.LOWEST, ignoreCancelled = true)
    public void onDamageWhilePaused(org.bukkit.event.entity.EntityDamageEvent event) {
        if (heldStill(event.getEntity())) {
            event.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.LOWEST, ignoreCancelled = true)
    public void onHealingWhilePaused(org.bukkit.event.entity.EntityRegainHealthEvent event) {
        if (heldStill(event.getEntity())) {
            event.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.LOWEST, ignoreCancelled = true)
    public void onHungerWhilePaused(org.bukkit.event.entity.FoodLevelChangeEvent event) {
        if (heldStill(event.getEntity()) && event.getItem() == null) {  // (eating is an action, and has ended)
            event.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.LOWEST, ignoreCancelled = true)
    public void onExhaustionWhilePaused(org.bukkit.event.entity.EntityExhaustionEvent event) {
        if (heldStill(event.getEntity())) {
            event.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.LOWEST, ignoreCancelled = true)
    public void onAirWhilePaused(org.bukkit.event.entity.EntityAirChangeEvent event) {
        if (heldStill(event.getEntity())) {
            event.setCancelled(true);
        }
    }

    /** Generate the overworld's chunks within `radius` chunks of the origin, and save them (for a server template:
     * servers copied from it then hold the same chunks, where each generating its own gave slightly different
     * worlds from one seed). */
    private JsonElement generate(String method, Map<String, String> query, JsonObject body) throws Exception {
        int radius = body.get("radius").getAsInt();
        if (radius < 0 || radius > 40) {
            throw new IllegalArgumentException("radius is in chunks, at most 40");
        }
        World world = overworld();
        java.util.List<java.util.concurrent.CompletableFuture<Void>> pending = new java.util.ArrayList<>();
        int chunks = 0;
        for (int cx = -radius; cx <= radius; cx++) {
            for (int cz = -radius; cz <= radius; cz++) {
                java.util.concurrent.CompletableFuture<Void> done = new java.util.concurrent.CompletableFuture<>();
                int x = cx, z = cz;
                Bukkit.getScheduler().runTask(this, () -> world.getChunkAtAsync(x, z, true, chunk -> done.complete(null)));
                pending.add(done);
                chunks++;
                if (pending.size() >= 256) {  // a row or so at a time, so that memory holds
                    java.util.concurrent.CompletableFuture.allOf(pending.toArray(new java.util.concurrent.CompletableFuture[0]))
                            .get(300, TimeUnit.SECONDS);
                    pending.clear();
                }
            }
        }
        java.util.concurrent.CompletableFuture.allOf(pending.toArray(new java.util.concurrent.CompletableFuture[0]))
                .get(300, TimeUnit.SECONDS);
        onMainThread(() -> { world.save(); return null; });
        JsonObject result = new JsonObject();
        result.addProperty("chunks", chunks);
        return result;
    }

    private JsonElement episode(String method, Map<String, String> query, JsonObject body) throws Exception {
        if (!method.equals("POST")) {
            throw new IllegalArgumentException("POST an episode");
        }
        return onMainThread(() -> {
            held.clear();  // what is held while the game is frozen is the team as set up here
            team.clear();
            lastKnownDiamonds.clear();
            lastKnownHeld.clear();
            for (JsonElement name : body.getAsJsonArray("team")) {
                team.add(name.getAsString().toLowerCase(Locale.ROOT));
            }
            JsonObject spawn = body.getAsJsonObject("spawn");
            World world = Bukkit.getWorld(spawn.has("world") ? spawn.get("world").getAsString() : "world");
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            for (World each : Bukkit.getWorlds()) {  // the nether and the end play by the same rules
                if (body.has("difficulty")) {
                    each.setDifficulty(Difficulty.valueOf(body.get("difficulty").getAsString().toUpperCase(Locale.ROOT)));
                }
                if (body.has("gamerules")) {
                    for (Map.Entry<String, JsonElement> rule : body.getAsJsonObject("gamerules").entrySet()) {
                        setGameRule(each, rule.getKey(), rule.getValue());
                    }
                }
            }
            if (body.has("time")) {
                overworld().setTime(body.get("time").getAsLong());
            }
            GameMode gameMode = GameMode.valueOf(body.has("gamemode") ? body.get("gamemode").getAsString().toUpperCase(Locale.ROOT) : "SURVIVAL");
            double x = spawn.get("x").getAsDouble(), y = spawn.get("y").getAsDouble(), z = spawn.get("z").getAsDouble();
            JsonArray placed = new JsonArray();
            JsonArray missing = new JsonArray();
            Map<String, JsonObject> placements = new HashMap<>();  // per player: x, y, z and a kit of their own
            if (body.has("placements")) {
                for (JsonElement placement : body.getAsJsonArray("placements")) {
                    JsonObject entry = placement.getAsJsonObject();
                    placements.put(entry.get("name").getAsString().toLowerCase(Locale.ROOT), entry);
                }
            }
            // One scoreboard team: teammates cannot hurt each other (their arrows pass through) or push each other.
            org.bukkit.scoreboard.Scoreboard scoreboard = Bukkit.getScoreboardManager().getMainScoreboard();
            org.bukkit.scoreboard.Team team = scoreboard.getTeam("team");
            if (team == null) {
                team = scoreboard.registerNewTeam("team");
            }
            team.setAllowFriendlyFire(false);
            team.setOption(org.bukkit.scoreboard.Team.Option.COLLISION_RULE, org.bukkit.scoreboard.Team.OptionStatus.NEVER);
            int index = 0;
            for (JsonElement entry : body.getAsJsonArray("team")) {
                Player player = Bukkit.getPlayerExact(entry.getAsString());
                if (player == null) {
                    missing.add(entry.getAsString());
                    continue;
                }
                team.addEntry(player.getName());
                JsonObject placement = placements.get(entry.getAsString().toLowerCase(Locale.ROOT));
                player.getInventory().clear();
                player.setItemOnCursor(null);
                JsonArray kit = placement != null && placement.has("kit") ? placement.getAsJsonArray("kit")
                        : body.has("kit") ? body.getAsJsonArray("kit") : new JsonArray();
                for (ItemStack stack : stacks(kit)) {
                    if (!wear(player, stack)) {
                        player.getInventory().addItem(stack);
                    }
                }
                player.updateInventory();  // the client sees the cleared and refilled slots
                player.setGameMode(gameMode);
                AttributeInstance maxHealth = player.getAttribute(Attribute.MAX_HEALTH);
                player.setHealth(maxHealth != null ? maxHealth.getValue() : 20.0);
                player.setFoodLevel(20);
                player.setSaturation(5.0f);
                player.setFireTicks(0);
                player.setFallDistance(0);
                if (placement != null) {
                    World target = placement.has("world") ? Bukkit.getWorld(placement.get("world").getAsString()) : world;
                    if (target == null) {
                        throw new IllegalArgumentException("no such world " + placement.get("world").getAsString());
                    }
                    Location start = new Location(target, placement.get("x").getAsDouble(), placement.get("y").getAsDouble(),
                            placement.get("z").getAsDouble(), (float) (index * 90), 0f);
                    player.teleport(start);
                    player.setRespawnLocation(start, true);  // whoever dies starts again from here
                } else {
                    double offset = index - 1.5;  // stand side by side, one block apart
                    player.teleport(new Location(world, x + offset, y, z, (float) (index * 90), 0f));
                }
                placed.add(player.getName());
                index++;
            }
            events.clear();
            record("episode", null, new JsonObject());
            JsonObject result = new JsonObject();
            result.add("placed", placed);
            result.add("missing", missing);
            return result;
        });
    }

    private JsonElement ores(String method, Map<String, String> query, JsonObject body) throws Exception {
        int radius = Math.min(Integer.parseInt(query.getOrDefault("radius", "32")), MAX_ORE_RADIUS);
        int cx = Integer.parseInt(query.get("x")), cy = Integer.parseInt(query.get("y")), cz = Integer.parseInt(query.get("z"));
        String worldName = query.getOrDefault("world", "world");
        return onMainThread(() -> {
            World world = Bukkit.getWorld(worldName);
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            JsonArray found = new JsonArray();
            int minY = Math.max(world.getMinHeight(), cy - radius), maxY = Math.min(world.getMaxHeight() - 1, cy + radius);
            for (int x = cx - radius; x <= cx + radius; x++) {
                for (int z = cz - radius; z <= cz + radius; z++) {
                    for (int y = minY; y <= maxY; y++) {
                        Block block = world.getBlockAt(x, y, z);
                        Material type = block.getType();
                        if (type != Material.DIAMOND_ORE && type != Material.DEEPSLATE_DIAMOND_ORE) {
                            continue;
                        }
                        JsonObject ore = new JsonObject();
                        ore.addProperty("x", x);
                        ore.addProperty("y", y);
                        ore.addProperty("z", z);
                        found.add(ore);
                    }
                }
            }
            JsonObject result = new JsonObject();
            result.add("ores", found);
            return result;
        });
    }

    private JsonElement events(String method, Map<String, String> query, JsonObject body) {
        long after = Long.parseLong(query.getOrDefault("after", "-1"));
        JsonArray selected = new JsonArray();
        for (JsonObject event : events) {
            if (event.get("seq").getAsLong() > after) {
                selected.add(event);
            }
        }
        JsonObject result = new JsonObject();
        result.add("events", selected);
        return result;
    }

    private JsonElement carve(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            World world = world(body);
            int x0 = body.get("x").getAsInt(), y0 = body.get("y").getAsInt(), z0 = body.get("z").getAsInt();
            int width = body.get("width").getAsInt(), height = body.get("height").getAsInt(), depth = body.get("depth").getAsInt();
            if (width < 1 || height < 2 || depth < 1 || width * height * depth > 32 * 32 * 32) {
                throw new IllegalArgumentException("a box is at least 1 x 2 x 1 and at most 32768 blocks");
            }
            boolean light = !body.has("light") || body.get("light").getAsBoolean();
            Material floor = body.has("floor") ? Material.matchMaterial(body.get("floor").getAsString()) : Material.STONE;
            for (int dx = 0; dx < width; dx++) {
                for (int dz = 0; dz < depth; dz++) {
                    Block below = world.getBlockAt(x0 + dx, y0 - 1, z0 + dz);
                    if (!below.getType().isSolid() || below.getType().hasGravity() || below.getType() == Material.MAGMA_BLOCK) {
                        below.setType(floor == null ? Material.STONE : floor, false);
                    }
                    for (int dy = 0; dy < height; dy++) {
                        world.getBlockAt(x0 + dx, y0 + dy, z0 + dz).setType(Material.AIR, false);
                    }
                }
            }
            // The box's shell: what would flow or fall into the room (water, lava, gravel, sand) becomes the floor's
            // material. A room carved beside water with a gravel floor once drowned a team in a "safe" task.
            Material wall = floor == null ? Material.STONE : floor;
            for (int dx = -1; dx <= width; dx++) {
                for (int dy = -1; dy <= height; dy++) {
                    for (int dz = -1; dz <= depth; dz++) {
                        boolean inside = dx >= 0 && dx < width && dy >= 0 && dy < height && dz >= 0 && dz < depth;
                        if (inside) {
                            continue;
                        }
                        Block shell = world.getBlockAt(x0 + dx, y0 + dy, z0 + dz);
                        Material type = shell.getType();
                        boolean waterlogged = shell.getBlockData() instanceof org.bukkit.block.data.Waterlogged logged
                                && logged.isWaterlogged();
                        if (shell.isLiquid() || waterlogged || type.hasGravity() || type == Material.MAGMA_BLOCK) {
                            shell.setType(wall, false);
                        }
                    }
                }
            }
            if (light) {  // invisible light blocks in the top corners
                for (int[] corner : new int[][] {{0, 0}, {width - 1, 0}, {0, depth - 1}, {width - 1, depth - 1}}) {
                    world.getBlockAt(x0 + corner[0], y0 + height - 1, z0 + corner[1]).setType(Material.LIGHT, false);
                }
            }
            return new JsonObject();
        });
    }

    private JsonElement dropItems(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            World world = world(body);
            Location location = new Location(world, body.get("x").getAsDouble() + 0.5, body.get("y").getAsDouble() + 0.1,
                    body.get("z").getAsDouble() + 0.5);
            int dropped = 0;
            for (ItemStack stack : stacks(body.getAsJsonArray("items"))) {
                Item item = world.dropItem(location, stack);
                item.setVelocity(new org.bukkit.util.Vector(0, 0, 0));
                item.setPickupDelay(0);
                dropped += stack.getAmount();
            }
            JsonObject result = new JsonObject();
            result.addProperty("dropped", dropped);
            return result;
        });
    }

    private JsonElement chest(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            Block block = world(body).getBlockAt(body.get("x").getAsInt(), body.get("y").getAsInt(), body.get("z").getAsInt());
            block.setType(Material.CHEST, false);
            org.bukkit.block.Chest chest = (org.bukkit.block.Chest) block.getState();
            for (ItemStack stack : stacks(body.has("items") ? body.getAsJsonArray("items") : new JsonArray())) {
                chest.getBlockInventory().addItem(stack);
            }
            return new JsonObject();
        });
    }

    private JsonElement setBlock(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            Material material = Material.matchMaterial(body.get("block").getAsString());
            if (material == null || !material.isBlock()) {
                throw new IllegalArgumentException("unknown block " + body.get("block").getAsString());
            }
            world(body).getBlockAt(body.get("x").getAsInt(), body.get("y").getAsInt(), body.get("z").getAsInt()).setType(material, false);
            return new JsonObject();
        });
    }

    /** Spawn a creature (staged fights, and tests). */
    private JsonElement spawnEntity(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            org.bukkit.entity.EntityType type = org.bukkit.entity.EntityType.valueOf(body.get("entity").getAsString().toUpperCase(Locale.ROOT));
            Location location = new Location(world(body), body.get("x").getAsDouble() + 0.5, body.get("y").getAsDouble(),
                    body.get("z").getAsDouble() + 0.5);
            org.bukkit.entity.Entity entity = location.getWorld().spawnEntity(location, type);
            if (entity instanceof org.bukkit.entity.LivingEntity living) {
                living.setRemoveWhenFarAway(false);
                if (body.has("ai") && !body.get("ai").getAsBoolean()) {
                    living.setAI(false);  // it stands where it is put
                }
            }
            JsonObject result = new JsonObject();
            result.addProperty("id", entity.getEntityId());
            return result;
        });
    }

    /** A player's hunger (20 is full), for staging and tests. */
    private JsonElement setFood(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            Player player = Bukkit.getPlayerExact(body.get("name").getAsString());
            if (player == null) {
                throw new IllegalArgumentException("not online");
            }
            player.setFoodLevel(body.get("food").getAsInt());
            player.setSaturation(0f);
            held.clear();  // what is held while the game is frozen is the player as set up
            return new JsonObject();
        });
    }

    /** Places to stand near a point: air at the feet and head, a solid floor, and no lava within two blocks. */
    private JsonElement standingSpots(String method, Map<String, String> query, JsonObject body) throws Exception {
        int radius = Math.min(Integer.parseInt(query.getOrDefault("radius", "16")), MAX_ORE_RADIUS);
        int limit = Integer.parseInt(query.getOrDefault("limit", "64"));
        int cx = Integer.parseInt(query.get("x")), cy = Integer.parseInt(query.get("y")), cz = Integer.parseInt(query.get("z"));
        String worldName = query.getOrDefault("world", "world");
        return onMainThread(() -> {
            World world = Bukkit.getWorld(worldName);
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            java.util.List<int[]> found = new java.util.ArrayList<>();
            int minY = Math.max(world.getMinHeight() + 1, cy - radius), maxY = Math.min(world.getMaxHeight() - 2, cy + radius);
            for (int x = cx - radius; x <= cx + radius; x++) {
                for (int z = cz - radius; z <= cz + radius; z++) {
                    for (int y = minY; y <= maxY; y++) {
                        if (safeToStand(world, x, y, z)) {
                            found.add(new int[] {x, y, z});
                        }
                    }
                }
            }
            found.sort(java.util.Comparator.comparingDouble(spot ->
                    Math.pow(spot[0] - cx, 2) + Math.pow(spot[1] - cy, 2) + Math.pow(spot[2] - cz, 2)));
            JsonArray spots = new JsonArray();
            for (int[] spot : found.subList(0, Math.min(limit, found.size()))) {
                JsonObject entry = new JsonObject();
                entry.addProperty("x", spot[0]);
                entry.addProperty("y", spot[1]);
                entry.addProperty("z", spot[2]);
                spots.add(entry);
            }
            JsonObject result = new JsonObject();
            result.add("spots", spots);
            return result;
        });
    }

    /** The nearest structure (fortress, stronghold, end_city) to a point: where tasks of the late game start. */
    private JsonElement locate(String method, Map<String, String> query, JsonObject body) throws Exception {
        String worldName = query.getOrDefault("world", "world");
        String name = query.get("structure").toUpperCase(Locale.ROOT);
        int x = Integer.parseInt(query.getOrDefault("x", "0")), z = Integer.parseInt(query.getOrDefault("z", "0"));
        int radius = Integer.parseInt(query.getOrDefault("radius", "100"));  // in chunks
        return onMainThread(() -> {
            World world = Bukkit.getWorld(worldName);
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            org.bukkit.generator.structure.Structure structure;
            try {
                structure = (org.bukkit.generator.structure.Structure)
                        org.bukkit.generator.structure.Structure.class.getField(name).get(null);
            } catch (ReflectiveOperationException error) {
                throw new IllegalArgumentException("unknown structure " + name);
            }
            org.bukkit.util.StructureSearchResult found =
                    world.locateNearestStructure(new Location(world, x, 64, z), structure, radius, false);
            JsonObject result = new JsonObject();
            if (found != null) {
                result.addProperty("x", found.getLocation().getBlockX());
                result.addProperty("y", found.getLocation().getBlockY());
                result.addProperty("z", found.getLocation().getBlockZ());
            }
            return result;
        });
    }

    /** Blocks of one type near a point (e.g. end portal frames, to start beside the stronghold's portal). */
    private JsonElement findBlocks(String method, Map<String, String> query, JsonObject body) throws Exception {
        String worldName = query.getOrDefault("world", "world");
        boolean logs = "#logs".equals(query.get("block"));  // any kind of log
        Material material = logs ? Material.OAK_LOG : Material.matchMaterial(query.get("block"));
        int radius = Math.min(Integer.parseInt(query.getOrDefault("radius", "32")), MAX_ORE_RADIUS);
        int cx = Integer.parseInt(query.get("x")), cy = Integer.parseInt(query.get("y")), cz = Integer.parseInt(query.get("z"));
        int limit = Integer.parseInt(query.getOrDefault("limit", "64"));
        if (material == null) {
            throw new IllegalArgumentException("unknown block " + query.get("block"));
        }
        return onMainThread(() -> {
            World world = Bukkit.getWorld(worldName);
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            JsonArray found = new JsonArray();
            int minY = Math.max(world.getMinHeight(), cy - radius), maxY = Math.min(world.getMaxHeight() - 1, cy + radius);
            for (int x = cx - radius; x <= cx + radius && found.size() < limit; x++) {
                for (int z = cz - radius; z <= cz + radius && found.size() < limit; z++) {
                    for (int y = minY; y <= maxY && found.size() < limit; y++) {
                        Material here = world.getBlockAt(x, y, z).getType();
                        if (logs ? org.bukkit.Tag.LOGS.isTagged(here) : here == material) {
                            JsonObject block = new JsonObject();
                            block.addProperty("x", x);
                            block.addProperty("y", y);
                            block.addProperty("z", z);
                            found.add(block);
                        }
                    }
                }
            }
            JsonObject result = new JsonObject();
            result.add("blocks", found);
            return result;
        });
    }

    /** The height of the highest solid block at a column, to start on the surface. */
    private JsonElement surface(String method, Map<String, String> query, JsonObject body) throws Exception {
        int x = Integer.parseInt(query.get("x")), z = Integer.parseInt(query.get("z"));
        String worldName = query.getOrDefault("world", "world");
        return onMainThread(() -> {
            World world = Bukkit.getWorld(worldName);
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            JsonObject result = new JsonObject();
            result.addProperty("y", world.getHighestBlockYAt(x, z, org.bukkit.HeightMap.MOTION_BLOCKING_NO_LEAVES));
            return result;
        });
    }

    private static boolean safeToStand(World world, int x, int y, int z) {
        Material floor = world.getBlockAt(x, y - 1, z).getType();
        if (!floor.isSolid() || floor == Material.MAGMA_BLOCK || floor == Material.POWDER_SNOW
                || floor == Material.CACTUS || floor.name().endsWith("CAMPFIRE")
                || org.bukkit.Tag.LEAVES.isTagged(floor)) {  // (a canopy is not the ground the team stands on)
            return false;
        }
        if (!world.getBlockAt(x, y, z).isPassable() || !world.getBlockAt(x, y + 1, z).isPassable()) {
            return false;
        }
        for (int dy = 0; dy <= 1; dy++) {
            Material in = world.getBlockAt(x, y + dy, z).getType();
            if (HARMFUL.contains(in)) {
                return false;
            }
        }
        if (world.getBlockAt(x, y, z).isLiquid() || world.getBlockAt(x, y + 1, z).isLiquid()) {
            return false;
        }
        for (int dx = -2; dx <= 2; dx++) {
            for (int dy = -2; dy <= 2; dy++) {
                for (int dz = -2; dz <= 2; dz++) {
                    if (world.getBlockAt(x + dx, y + dy, z + dz).getType() == Material.LAVA) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static final Set<Material> HARMFUL = Set.of(Material.FIRE, Material.SOUL_FIRE, Material.COBWEB,
            Material.SWEET_BERRY_BUSH, Material.WITHER_ROSE, Material.NETHER_PORTAL, Material.END_PORTAL);

    private static World world(JsonObject body) {
        World world = Bukkit.getWorld(body.has("world") ? body.get("world").getAsString() : "world");
        if (world == null) {
            throw new IllegalArgumentException("no such world");
        }
        return world;
    }

    /** Put a piece of armor on, if that slot is empty. */
    private static boolean wear(Player player, ItemStack stack) {
        String name = stack.getType().name();
        org.bukkit.inventory.PlayerInventory inventory = player.getInventory();
        if (name.endsWith("_HELMET") && empty(inventory.getHelmet())) {
            inventory.setHelmet(stack);
        } else if (name.endsWith("_CHESTPLATE") && empty(inventory.getChestplate())) {
            inventory.setChestplate(stack);
        } else if (name.endsWith("_LEGGINGS") && empty(inventory.getLeggings())) {
            inventory.setLeggings(stack);
        } else if (name.endsWith("_BOOTS") && empty(inventory.getBoots())) {
            inventory.setBoots(stack);
        } else {
            return false;
        }
        return true;
    }

    private static boolean empty(ItemStack stack) {
        return stack == null || stack.getType().isAir();
    }

    private static java.util.List<ItemStack> stacks(JsonArray items) {
        java.util.List<ItemStack> result = new java.util.ArrayList<>();
        for (JsonElement item : items) {
            JsonObject stack = item.getAsJsonObject();
            Material material = Material.matchMaterial(stack.get("item").getAsString());
            if (material == null || !material.isItem()) {
                throw new IllegalArgumentException("unknown item " + stack.get("item").getAsString());
            }
            int count = stack.has("count") ? stack.get("count").getAsInt() : 1;
            while (count > 0) {  // stacks of at most the item's maximum size
                int size = Math.min(count, material.getMaxStackSize());
                result.add(new ItemStack(material, size));
                count -= size;
            }
        }
        return result;
    }

    private JsonElement baseline(String method, Map<String, String> query, JsonObject body) throws Exception {
        return onMainThread(() -> {
            baselines.clear();
            teamEarned.clear();
            teamObtained.clear();
            placedContainers.clear();
            dragonKilled = false;
            dragonDamage = 0.0;
            JsonObject result = new JsonObject();
            for (Player player : Bukkit.getOnlinePlayers()) {
                String name = player.getName().toLowerCase(Locale.ROOT);
                if (team.contains(name)) {
                    Set<String> done = doneAdvancements(player);
                    baselines.put(name, done);
                    result.addProperty(player.getName(), done.size());
                }
            }
            return result;
        });
    }

    /** Advancements a player has completed (not the recipe unlocks Minecraft also tracks as advancements). */
    private static Set<String> doneAdvancements(Player player) {
        Set<String> done = new java.util.TreeSet<>();
        java.util.Iterator<org.bukkit.advancement.Advancement> all = Bukkit.advancementIterator();
        while (all.hasNext()) {
            org.bukkit.advancement.Advancement advancement = all.next();
            String key = advancement.getKey().getKey();
            if (key.startsWith("recipes/")) {
                continue;
            }
            if (player.getAdvancementProgress(advancement).isDone()) {
                done.add(key);
            }
        }
        return done;
    }

    private Set<String> newAdvancements(Player player) {
        Set<String> done = doneAdvancements(player);
        done.removeAll(baselines.getOrDefault(player.getName().toLowerCase(Locale.ROOT), Set.of()));
        return done;
    }

    // Listeners

    @EventHandler(priority = EventPriority.MONITOR)
    public void onAdvancement(org.bukkit.event.player.PlayerAdvancementDoneEvent event) {
        String key = event.getAdvancement().getKey().getKey();
        if (!key.startsWith("recipes/")) {
            JsonObject data = new JsonObject();
            data.addProperty("advancement", key);
            record("advancement", event.getPlayer(), data);
            String name = event.getPlayer().getName().toLowerCase(Locale.ROOT);
            if (team.contains(name) && baselines.containsKey(name)) {
                teamEarned.add(key);
            }
        }
    }

    /** The most the dragon has been hurt so far, as a share of its health (it heals at its crystals). */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDragonDamage(org.bukkit.event.entity.EntityDamageEvent event) {
        org.bukkit.entity.Entity hurt = event.getEntity();
        if (hurt instanceof org.bukkit.entity.EnderDragonPart part) {
            hurt = part.getParent();
        }
        if (hurt instanceof org.bukkit.entity.EnderDragon dragon) {
            AttributeInstance maxHealth = dragon.getAttribute(Attribute.MAX_HEALTH);
            double most = maxHealth != null ? maxHealth.getValue() : 200.0;
            double left = Math.max(0.0, dragon.getHealth() - event.getFinalDamage());
            dragonDamage = Math.max(dragonDamage, 1.0 - left / most);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onEntityDeath(org.bukkit.event.entity.EntityDeathEvent event) {
        if (event.getEntityType() == org.bukkit.entity.EntityType.ENDER_DRAGON) {
            dragonKilled = true;
            record("dragon_killed", event.getEntity().getKiller(), new JsonObject());
        }
    }

    /** A team member hurt a creature, by hand or with an arrow. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onHurt(org.bukkit.event.entity.EntityDamageByEntityEvent event) {
        org.bukkit.entity.Entity source = event.getDamager();
        if (source instanceof org.bukkit.entity.Projectile projectile && projectile.getShooter() instanceof Player shooter) {
            source = shooter;
        }
        if (!(source instanceof Player player) || !team.contains(player.getName().toLowerCase(Locale.ROOT))
                || event.getEntity() instanceof Player) {
            return;
        }
        JsonObject data = new JsonObject();
        data.addProperty("entity", event.getEntityType().name().toLowerCase(Locale.ROOT));
        data.addProperty("damage", Math.round(event.getFinalDamage() * 10) / 10.0);
        data.addProperty("with", event.getDamager() instanceof org.bukkit.entity.Projectile ? "arrow" : "hand");
        record("hurt", player, data);
    }

    /** Whoever threw an item cannot pick it back up for five seconds: tossed toward a teammate, it is the teammate's. */
    @EventHandler(priority = EventPriority.NORMAL, ignoreCancelled = true)
    public void onPickupOfOwnThrow(EntityPickupItemEvent event) {
        Item item = event.getItem();
        if (event.getEntity() instanceof Player player && player.getUniqueId().equals(item.getThrower())
                && item.getTicksLived() < 100) {
            event.setCancelled(true);
        }
    }

    /** A move the server refused (it puts the player back): recorded once a second per player, for diagnosis. */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onFailedMove(io.papermc.paper.event.player.PlayerFailMoveEvent event) {
        Player player = event.getPlayer();
        long now = overworld().getGameTime();
        Long last = lastFailedMove.get(player.getName());
        if (last != null && now - last < 20) {
            return;
        }
        lastFailedMove.put(player.getName(), now);
        JsonObject data = new JsonObject();
        data.addProperty("reason", event.getFailReason().name().toLowerCase(Locale.ROOT));
        data.addProperty("from", event.getFrom().toVector().toString());
        data.addProperty("to", event.getTo().toVector().toString());
        record("move_refused", player, data);
    }

    @EventHandler(priority = EventPriority.LOWEST)
    public void onCommand(PlayerCommandPreprocessEvent event) {
        if (team.contains(event.getPlayer().getName().toLowerCase(Locale.ROOT))) {
            event.setCancelled(true);  // agents act only through the game, never through commands
            JsonObject data = new JsonObject();
            data.addProperty("command", event.getMessage());
            record("command_refused", event.getPlayer(), data);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBreak(BlockBreakEvent event) {
        String type = event.getBlock().getType().name();
        if (type.endsWith("_ORE")) {
            JsonObject data = new JsonObject();
            data.addProperty("block", type.toLowerCase(Locale.ROOT));
            data.addProperty("x", event.getBlock().getX());
            data.addProperty("y", event.getBlock().getY());
            data.addProperty("z", event.getBlock().getZ());
            record("mined", event.getPlayer(), data);
        }
    }

    /** What the team has got hold of since the baseline: picked up, crafted, or taken from a furnace. */
    private void obtained(Player player, Material item, int count) {
        String name = player.getName().toLowerCase(Locale.ROOT);
        if (team.contains(name) && baselines.containsKey(name)) {
            teamObtained.merge(item.name().toLowerCase(Locale.ROOT), count, Integer::sum);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onCraft(org.bukkit.event.inventory.CraftItemEvent event) {
        if (event.getWhoClicked() instanceof Player player) {
            ItemStack made = event.getRecipe().getResult();
            obtained(player, made.getType(), made.getAmount());
            JsonObject data = new JsonObject();
            data.addProperty("item", made.getType().name().toLowerCase(Locale.ROOT));
            record("crafted", player, data);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onSmelted(org.bukkit.event.inventory.FurnaceExtractEvent event) {
        obtained(event.getPlayer(), event.getItemType(), event.getItemAmount());
        JsonObject data = new JsonObject();
        data.addProperty("item", event.getItemType().name().toLowerCase(Locale.ROOT));
        data.addProperty("count", event.getItemAmount());
        record("smelted", event.getPlayer(), data);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPickup(EntityPickupItemEvent event) {
        if (event.getEntity() instanceof Player player) {
            Item item = event.getItem();
            obtained(player, item.getItemStack().getType(), item.getItemStack().getAmount());
            JsonObject data = new JsonObject();
            data.addProperty("item", item.getItemStack().getType().name().toLowerCase(Locale.ROOT));
            data.addProperty("count", item.getItemStack().getAmount());
            record("picked_up", player, data);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        JsonObject data = new JsonObject();
        data.addProperty("diamonds_dropped", event.getKeepInventory() ? 0 : diamonds(event.getEntity()));
        record("died", event.getEntity(), data);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onChat(AsyncChatEvent event) {
        JsonObject data = new JsonObject();
        data.addProperty("message", PlainTextComponentSerializer.plainText().serialize(event.message()));
        record("chat", event.getPlayer(), data);
    }

    @EventHandler
    public void onJoin(PlayerJoinEvent event) {
        record("joined", event.getPlayer(), new JsonObject());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPlace(BlockPlaceEvent event) {
        String name = event.getPlayer().getName().toLowerCase(Locale.ROOT);
        if (team.contains(name) && event.getBlockPlaced().getState() instanceof Container) {
            placedContainers.add(event.getBlockPlaced().getLocation());
        }
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        String name = event.getPlayer().getName().toLowerCase(Locale.ROOT);
        if (team.contains(name)) {
            lastKnownDiamonds.put(name, diamonds(event.getPlayer()));
            lastKnownHeld.put(name, held(event.getPlayer()));
        }
        record("left", event.getPlayer(), new JsonObject());
    }

    // Ground truth

    /** Diamonds a player holds: inventory (with armor and offhand), cursor and crafting grid; a block counts 9. */
    static int diamonds(Player player) {
        int total = 0;
        for (ItemStack stack : player.getInventory().getContents()) {
            total += diamondValue(stack);
        }
        total += diamondValue(player.getItemOnCursor());
        InventoryView view = player.getOpenInventory();
        switch (view.getType()) {
            case CRAFTING, WORKBENCH -> {
                ItemStack[] grid = view.getTopInventory().getContents();
                for (int slot = 1; slot < grid.length; slot++) {  // slot 0 is the result, not yet held
                    total += diamondValue(grid[slot]);
                }
            }
            default -> { }
        }
        return total;
    }

    /** Every item a player holds, by its id: inventory (with armor and offhand), cursor and crafting grid. */
    static Map<String, Integer> held(Player player) {
        Map<String, Integer> found = new TreeMap<>();
        for (ItemStack stack : player.getInventory().getContents()) {
            count(found, stack);
        }
        count(found, player.getItemOnCursor());
        InventoryView view = player.getOpenInventory();
        switch (view.getType()) {
            case CRAFTING, WORKBENCH -> {
                ItemStack[] grid = view.getTopInventory().getContents();
                for (int slot = 1; slot < grid.length; slot++) {  // slot 0 is the result, not yet held
                    count(found, grid[slot]);
                }
            }
            default -> { }
        }
        return found;
    }

    private static void count(Map<String, Integer> found, ItemStack stack) {
        if (stack != null && !stack.getType().isAir() && stack.getAmount() > 0) {
            found.merge(stack.getType().getKey().getKey(), stack.getAmount(), Integer::sum);
        }
    }

    private static int diamondValue(ItemStack stack) {
        if (stack == null) {
            return 0;
        }
        return switch (stack.getType()) {
            case DIAMOND -> stack.getAmount();
            case DIAMOND_BLOCK -> 9 * stack.getAmount();
            default -> 0;
        };
    }

    private JsonObject describe(Player player, int diamonds) {
        JsonObject result = new JsonObject();
        result.addProperty("name", player.getName());
        result.addProperty("uuid", player.getUniqueId().toString());
        result.addProperty("team", team.contains(player.getName().toLowerCase(Locale.ROOT)));
        result.addProperty("gamemode", player.getGameMode().name().toLowerCase(Locale.ROOT));
        Location location = player.getLocation();
        result.addProperty("world", location.getWorld().getName());
        result.addProperty("x", location.getX());
        result.addProperty("y", location.getY());
        result.addProperty("z", location.getZ());
        result.addProperty("health", player.getHealth());
        result.addProperty("food", player.getFoodLevel());
        result.addProperty("diamonds", diamonds);
        Map<String, Integer> inventory = new TreeMap<>();
        for (ItemStack stack : player.getInventory().getContents()) {
            if (stack != null && !stack.getType().isAir()) {
                inventory.merge(stack.getType().name().toLowerCase(Locale.ROOT), stack.getAmount(), Integer::sum);
            }
        }
        result.add("inventory", gson.toJsonTree(inventory));
        return result;
    }

    private JsonObject tickState() {
        ServerTickManager ticks = Bukkit.getServerTickManager();
        JsonObject result = new JsonObject();
        result.addProperty("frozen", ticks.isFrozen());
        result.addProperty("stepping", ticks.isStepping());
        result.addProperty("rate", ticks.getTickRate());
        result.addProperty("game_time", overworld().getGameTime());
        return result;
    }

    private static World overworld() {
        return Bukkit.getWorlds().get(0);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static void setGameRule(World world, String name, JsonElement value) {
        GameRule rule = GameRule.getByName(name);
        if (rule == null) {  // the API's constant names (DO_DAYLIGHT_CYCLE) outlive the game's renamed keys
            try {
                rule = (GameRule) GameRule.class.getField(name.toUpperCase(Locale.ROOT)).get(null);
            } catch (ReflectiveOperationException error) {
                throw new IllegalArgumentException("unknown game rule " + name);
            }
        }
        Object converted = rule.getType() == Boolean.class ? (Object) value.getAsBoolean() : (Object) value.getAsInt();
        world.setGameRule(rule, converted);
    }

    private void record(String kind, Player player, JsonObject data) {
        data.addProperty("seq", eventSequence.incrementAndGet());
        data.addProperty("kind", kind);
        if (player != null) {
            data.addProperty("player", player.getName());
        }
        data.addProperty("game_time", overworld().getGameTime());
        events.addLast(data);
        while (events.size() > MAX_EVENTS) {
            events.pollFirst();
        }
    }

    // Plumbing

    private <T> T onMainThread(Callable<T> work) throws Exception {
        if (Bukkit.isPrimaryThread()) {
            return work.call();
        }
        return Bukkit.getScheduler().callSyncMethod(this, work).get(30, TimeUnit.SECONDS);
    }

    private void route(String path, Route handler) {
        http.createContext(path, exchange -> {
            int status = 200;
            JsonElement response;
            try {
                String text = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                JsonObject body = text.isBlank() ? new JsonObject() : JsonParser.parseString(text).getAsJsonObject();
                response = handler.handle(exchange.getRequestMethod(), query(exchange.getRequestURI()), body);
                if (response == null) {
                    response = new JsonObject();
                }
            } catch (IllegalArgumentException | IllegalStateException | NullPointerException error) {
                status = 400;
                response = error(error);
            } catch (Exception error) {
                status = 500;
                response = error(error);
            }
            send(exchange, status, response);
        });
    }

    private JsonObject error(Exception error) {
        JsonObject result = new JsonObject();
        result.addProperty("error", error.getClass().getSimpleName() + ": " + error.getMessage());
        return result;
    }

    private void send(HttpExchange exchange, int status, JsonElement body) throws IOException {
        byte[] bytes = gson.toJson(body).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream stream = exchange.getResponseBody()) {
            stream.write(bytes);
        }
    }

    private static Map<String, String> query(URI uri) {
        Map<String, String> result = new HashMap<>();
        String raw = uri.getRawQuery();
        if (raw == null) {
            return result;
        }
        for (String pair : raw.split("&")) {
            String[] parts = pair.split("=", 2);
            result.put(java.net.URLDecoder.decode(parts[0], StandardCharsets.UTF_8),
                    parts.length > 1 ? java.net.URLDecoder.decode(parts[1], StandardCharsets.UTF_8) : "");
        }
        return result;
    }
}
