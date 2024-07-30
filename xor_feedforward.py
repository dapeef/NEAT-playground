"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os

import neat
import neat_utils

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    # p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="./checkpoints/xor/"))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\n--- Best genome: ---\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\n--- Output: ---')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    neat_utils.draw_net(config, winner, True, node_names=node_names)
    neat_utils.draw_net(config, winner, True, node_names=node_names, prune_unused=True, filename="pruned-network")
    neat_utils.plot_stats(stats, ylog=False, view=True)
    neat_utils.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('./checkpoints/4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    run("xor_config.txt")