
# BFS finds the shortest path
# Dijkstra chooses a longer avoiding tiles close to holes.
map_DijkstraVsBFS = [
    "SHFFFFFF",
    "FHHHHHHF",
    "FHFFFFHF",
    "FHFFFFHF",
    "FHHHHHFF",
    "FFFFFFFF",
    "FFFFFFHF",
    "FFFFFFHG",
]

map_DijkstraVsBFS_2 = [
    ['S', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F'],
    ['F', 'H', 'H', 'F', 'F', 'H', 'H', 'F', 'H', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'H', 'H', 'F', 'F'],
    ['F', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['F', 'F', 'H', 'H', 'H', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'H'],
    ['H', 'F', 'F', 'H', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'H', 'F', 'F', 'F', 'H'],
    ['F', 'F', 'H', 'H', 'H', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
    ['F', 'F', 'H', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'F'],
    ['H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'H', 'H', 'H', 'F', 'H', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'H', 'F'],
    ['F', 'F', 'F', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['H', 'F', 'F', 'H', 'F', 'F', 'F', 'H', 'F', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
    ['H', 'F', 'F', 'F', 'H', 'F', 'F', 'H', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['H', 'F', 'F', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'H', 'F'],
    ['F', 'H', 'H', 'F', 'H', 'F', 'H', 'F', 'H', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'F'],
    ['H', 'F', 'H', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'H', 'F', 'F', 'F', 'H', 'F', 'H', 'H', 'F', 'F'],
    ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'H', 'F', 'H', 'F'],
    ['H', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'H', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
    ['F', 'F', 'H', 'H', 'F', 'F', 'H', 'F', 'H', 'H', 'H', 'H', 'F', 'F', 'F', 'H', 'H', 'F', 'F', 'G'],
]