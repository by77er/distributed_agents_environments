"""What the policy is asked: concepts to explain to a ten-year-old, and short passages to summarize. Each list is split
in two: what training draws from, and what only the eval plays (`judging.environment`)."""

EXPLAIN_TRAIN = (
    "photosynthesis",
    "why the sky is blue",
    "how vaccines work",
    "what gravity is",
    "why we have seasons",
    "how a rainbow forms",
    "what electricity is",
    "how the heart pumps blood",
    "why ice floats",
    "what an echo is",
    "how bees make honey",
    "what a volcano is",
    "why leaves change colour in autumn",
    "how a magnet works",
    "what the internet is",
    "why the moon has phases",
)
"""Concepts training draws from."""
EXPLAIN_EVAL = (
    "how plants drink water",
    "what a black hole is",
    "why we need sleep",
    "how sound travels",
    "what recycling does",
    "why bread rises",
)
"""Concepts only the eval asks about."""

SUMMARIZE_TRAIN = (
    (
        "the lighthouse keeper",
        "For forty years Mara kept the lighthouse on Gull Point. Every evening she climbed the ninety steps, "
        "cleaned the great lens and lit the lamp. When the coast guard replaced the lamp with an automatic "
        "light, the town expected her to leave. Instead she stayed, turning the keeper's cottage into a small "
        "museum about the ships the light had guided home, and children now climb the steps with her on "
        "Saturdays.",
    ),
    (
        "the community garden",
        "An empty lot behind the bus depot had collected rubbish for a decade. Last spring a group of "
        "neighbours asked the council for permission to clear it. They hauled away three skips of junk, built "
        "raised beds from old pallets and planted beans, tomatoes and sunflowers. By August the garden fed a "
        "dozen families, and the depot's drivers had started spending their breaks on a bench by the "
        "sunflowers.",
    ),
    (
        "the late train",
        "The 6:40 train from Halden was late every morning for a month. Commuters blamed the drivers, but an "
        "engineer who rode it noticed the delay always began at the same bridge. A sensor there had been "
        "fitted upside down and kept reporting a fault, which forced trains to slow to walking pace. Once the "
        "sensor was turned the right way up, the train ran on time for the first time that year.",
    ),
    (
        "the chess club",
        "When the school's chess club had only three members, Mr Osei moved its meetings from a classroom to "
        "the library at lunchtime, where anyone passing could watch. He left a board set up with a puzzle "
        "each day and a notebook for answers. Within a term the notebook was full, the club had thirty "
        "members, and two of them reached the regional final.",
    ),
    (
        "the lost dog",
        "Biscuit, a small brown terrier, slipped his lead during a thunderstorm and vanished. His owners put "
        "up posters and searched for days. A week later a delivery driver spotted a dog sheltering under a "
        "loading bay four miles away and recognised him from a poster at the petrol station. Biscuit was thin "
        "and muddy but unhurt, and he now wears a collar with a tracker.",
    ),
    (
        "the bakery's oven",
        "The bakery on Mill Street had used the same wood-fired oven since 1920. When it cracked last winter, "
        "the owners were told a new electric oven would be cheaper to run. Regular customers raised money to "
        "repair the old one instead, and a retired bricklayer offered his time for free. The oven was relit "
        "in March, and the queue on the first morning stretched round the corner.",
    ),
    (
        "the river clean-up",
        "Students from Ridgeway College spent a weekend wading through the River Lune with nets and gloves. "
        "They pulled out bicycles, shopping trolleys and more than two thousand plastic bottles. They logged "
        "every item in a shared spreadsheet and sent the results to the council, which agreed to install bins "
        "and a fence at the two spots where most of the rubbish had entered the water.",
    ),
    (
        "the night sky festival",
        "Every October the village of Dunmore switches off its street lights for one night. Astronomers set "
        "up telescopes on the green, and visitors lie on blankets to watch for shooting stars. The festival "
        "began when a teacher complained that her pupils had never seen the Milky Way. It now draws thousands "
        "of people, and the village has applied to become an official dark sky reserve.",
    ),
    (
        "the repair café",
        "Once a month the church hall in Fenwick becomes a repair café. Volunteers with screwdrivers, sewing "
        "machines and soldering irons fix whatever people bring: lamps, jackets, toasters, toys. Nobody is "
        "charged, but owners must stay and watch so they learn how the repair was done. Last year the café "
        "mended over six hundred items that would otherwise have been thrown away.",
    ),
    (
        "the first library card",
        "Amira was seven when she got her first library card. She borrowed the limit of six books every week "
        "and kept a list of every title in a notebook. By eleven the list ran to four notebooks. When the "
        "library faced closure, she read the list aloud at a council meeting, one title at a time, until the "
        "councillors voted to keep the library open.",
    ),
)
"""Passages training draws from, each with a title."""
SUMMARIZE_EVAL = (
    (
        "the beekeeper's map",
        "When bees in the valley began to die, a beekeeper named Tomas started marking every failing hive on "
        "a paper map in his kitchen. After a season the marks formed a line along one road. He traced it to a "
        "farm that had switched to a new spray. The farmer changed back once he saw the map, and the next "
        "spring the hives along the road recovered.",
    ),
    (
        "the bus stop shelter",
        "The bus stop outside the hospital had no shelter, and patients waited in the rain. A nurse wrote to "
        "the bus company every week for a year. When nothing changed, she and her colleagues built a wooden "
        "shelter on a weekend with donated timber. The company, embarrassed by a photo in the local paper, "
        "replaced it with a proper glass shelter a month later.",
    ),
    (
        "the orchestra's lost score",
        "On the morning of its centenary concert, the town orchestra discovered that the only copy of the "
        "score for its opening piece had been thrown out with the recycling. The players spent the day "
        "rebuilding it from memory, each writing out their own part. The performance that evening was not "
        "perfect, but the audience gave the longest standing ovation in the orchestra's history.",
    ),
    (
        "the school's weather station",
        "A primary school bought a small weather station to teach pupils about measurement. The children "
        "recorded temperature, rain and wind every day and posted the readings online. After two years their "
        "records were the most complete in the area, and a university climate team asked to use them in a "
        "study of how the town's rainfall was changing.",
    ),
)
"""Passages only the eval asks to summarize."""
