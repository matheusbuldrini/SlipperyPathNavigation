
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

map_4x4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

map_8x8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
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


map_bad_for_AndOrSolver = [
            "SFFFHHFFFFFFFFH",
            "FFHHHFFHHHFFHFH",
            "HFHFHHFFHFFFHHF",
            "FFFHHHFFFFHFHFF",
            "HFHHFFFHFFFFFHH",
            "FFFFFHFFFFFHHHF",
            "HHFFFFHFFFFFHFF",
            "FFFFFFFFFFFFFFF",
            "FFFFFFHFFFHFFFH",
            "FFFFHFHHFHHFFHH",
            "HFHFFFFHFFFHHHF",
            "FHFFFHFFHFFFFHF",
            "HFFFFHFFFFFHFHH",
            "HFFFFFFFHFFHFHF",
            "HFHFFFFFHFHFFFG",
        ]