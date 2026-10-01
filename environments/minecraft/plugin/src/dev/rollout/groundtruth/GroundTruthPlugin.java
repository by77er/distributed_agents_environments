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
import org.bukkit.block.BlockFace;
import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.entity.EntityPickupItemEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerCommandPreprocessEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
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
 * {@code rollout.control.port}). Agents never see this API: it is for the environment service.
 *
 * <ul>
 *   <li>GET /health: ready, and who is online.</li>
 *   <li>GET /state: every player's position, health, food, inventory and diamonds; the team's total diamonds.</li>
 *   <li>GET /tick, POST /tick {"action": "freeze" | "unfreeze" | "step", "ticks": n}: a step runs n ticks of a
 *       frozen game and answers when they have run.</li>
 *   <li>POST /episode: set up the team (clear, kit, teleport, game mode) and the world (difficulty, time, rules).</li>
 *   <li>GET /ores?x&amp;y&amp;z&amp;radius&amp;exposed: diamond ores near a point, for choosing starts (never agents).</li>
 *   <li>GET /events?after=n: what happened (chat, ores mined, items picked up, deaths, joins).</li>
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
        route("/tick", this::tick);
        route("/episode", this::episode);
        route("/ores", this::ores);
        route("/events", this::events);
        getServer().getPluginManager().registerEvents(this, this);
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
            for (Player player : Bukkit.getOnlinePlayers()) {
                int diamonds = diamonds(player);
                String name = player.getName().toLowerCase(Locale.ROOT);
                if (team.contains(name)) {
                    teamDiamonds += diamonds;
                    counted.add(name);
                }
                players.add(describe(player, diamonds));
            }
            for (String member : team) {  // members who left keep what they held when they left
                if (!counted.contains(member)) {
                    teamDiamonds += lastKnownDiamonds.getOrDefault(member, 0);
                }
            }
            result.add("players", players);
            result.addProperty("team_diamonds", teamDiamonds);
            JsonArray members = new JsonArray();
            team.forEach(members::add);
            result.add("team", members);
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
            case "unfreeze" -> onMainThread(() -> { ticks.setFrozen(false); return null; });
            case "step" -> {
                int count = body.get("ticks").getAsInt();
                if (count < 1 || count > 20 * 60 * 10) {
                    throw new IllegalArgumentException("ticks must be between 1 and 12000");
                }
                long before = onMainThread(() -> overworld().getFullTime());
                boolean started = onMainThread(() -> {
                    if (!ticks.isFrozen()) {
                        ticks.setFrozen(true);
                    }
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
                result.addProperty("stepped", onMainThread(() -> overworld().getFullTime()) - before);
                return result;
            }
            default -> throw new IllegalArgumentException("action must be freeze, unfreeze or step");
        }
        return onMainThread(this::tickState);
    }

    private JsonElement episode(String method, Map<String, String> query, JsonObject body) throws Exception {
        if (!method.equals("POST")) {
            throw new IllegalArgumentException("POST an episode");
        }
        return onMainThread(() -> {
            team.clear();
            lastKnownDiamonds.clear();
            for (JsonElement name : body.getAsJsonArray("team")) {
                team.add(name.getAsString().toLowerCase(Locale.ROOT));
            }
            JsonObject spawn = body.getAsJsonObject("spawn");
            World world = Bukkit.getWorld(spawn.has("world") ? spawn.get("world").getAsString() : "world");
            if (world == null) {
                throw new IllegalArgumentException("no such world");
            }
            if (body.has("difficulty")) {
                world.setDifficulty(Difficulty.valueOf(body.get("difficulty").getAsString().toUpperCase(Locale.ROOT)));
            }
            if (body.has("time")) {
                world.setTime(body.get("time").getAsLong());
            }
            if (body.has("gamerules")) {
                for (Map.Entry<String, JsonElement> rule : body.getAsJsonObject("gamerules").entrySet()) {
                    setGameRule(world, rule.getKey(), rule.getValue());
                }
            }
            GameMode gameMode = GameMode.valueOf(body.has("gamemode") ? body.get("gamemode").getAsString().toUpperCase(Locale.ROOT) : "SURVIVAL");
            double x = spawn.get("x").getAsDouble(), y = spawn.get("y").getAsDouble(), z = spawn.get("z").getAsDouble();
            JsonArray placed = new JsonArray();
            JsonArray missing = new JsonArray();
            int index = 0;
            for (JsonElement entry : body.getAsJsonArray("team")) {
                Player player = Bukkit.getPlayerExact(entry.getAsString());
                if (player == null) {
                    missing.add(entry.getAsString());
                    continue;
                }
                player.getInventory().clear();
                player.setItemOnCursor(null);
                if (body.has("kit")) {
                    for (JsonElement item : body.getAsJsonArray("kit")) {
                        JsonObject stack = item.getAsJsonObject();
                        Material material = Material.matchMaterial(stack.get("item").getAsString());
                        if (material == null) {
                            throw new IllegalArgumentException("unknown item " + stack.get("item").getAsString());
                        }
                        player.getInventory().addItem(new ItemStack(material, stack.has("count") ? stack.get("count").getAsInt() : 1));
                    }
                }
                player.setGameMode(gameMode);
                AttributeInstance maxHealth = player.getAttribute(Attribute.MAX_HEALTH);
                player.setHealth(maxHealth != null ? maxHealth.getValue() : 20.0);
                player.setFoodLevel(20);
                player.setSaturation(5.0f);
                player.setFireTicks(0);
                player.setFallDistance(0);
                double offset = index - 1.5;  // stand side by side, one block apart
                player.teleport(new Location(world, x + offset, y, z, (float) (index * 90), 0f));
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
        boolean exposedOnly = Boolean.parseBoolean(query.getOrDefault("exposed", "false"));
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
                        boolean exposed = isExposed(block);
                        if (exposedOnly && !exposed) {
                            continue;
                        }
                        JsonObject ore = new JsonObject();
                        ore.addProperty("x", x);
                        ore.addProperty("y", y);
                        ore.addProperty("z", z);
                        ore.addProperty("exposed", exposed);
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

    // Listeners

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

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPickup(EntityPickupItemEvent event) {
        if (event.getEntity() instanceof Player player) {
            Item item = event.getItem();
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

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        String name = event.getPlayer().getName().toLowerCase(Locale.ROOT);
        if (team.contains(name)) {
            lastKnownDiamonds.put(name, diamonds(event.getPlayer()));
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

    private static boolean isExposed(Block block) {
        for (BlockFace face : new BlockFace[] {BlockFace.UP, BlockFace.DOWN, BlockFace.NORTH, BlockFace.SOUTH, BlockFace.EAST, BlockFace.WEST}) {
            Material neighbour = block.getRelative(face).getType();
            if (neighbour.isAir() || neighbour == Material.WATER || neighbour == Material.LAVA) {
                return true;
            }
        }
        return false;
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
        result.addProperty("game_time", overworld().getFullTime());
        return result;
    }

    private static World overworld() {
        return Bukkit.getWorlds().get(0);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static void setGameRule(World world, String name, JsonElement value) {
        GameRule rule = GameRule.getByName(name);
        if (rule == null) {
            throw new IllegalArgumentException("unknown game rule " + name);
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
        data.addProperty("game_time", overworld().getFullTime());
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
