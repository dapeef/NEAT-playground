import pygame
import neat
import random
import neat_utils
import pickle


SPEED_MULTIPLIER = 8

WIN_WIDTH, WIN_HEIGHT = 500, 500
BACKGROUND_COLOUR = (0, 0, 0)
FPS = 60

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 10
PADDLE_COLOUR = (255, 255, 255)
PADDLE_SPEED = 100 # px/sec
PADDLE_SPEED *= SPEED_MULTIPLIER

BALL_RADIUS = 10
BALL_COLOUR = (255, 0, 0)
BALL_SPEED = 100 # px/sec
BALL_SPEED *= SPEED_MULTIPLIER

REWARD_PER_TIME = 0.1 # /sec
REWARD_PER_BOUNCE = 10
PENALTY_PER_DEATH = 5

FITNESS_THRESHOLD = 500


class Paddle:
    def __init__(self) -> None:
        self.position : pygame.math.Vector2 = pygame.math.Vector2(random.randint(PADDLE_WIDTH//2, WIN_WIDTH - PADDLE_WIDTH//2), WIN_HEIGHT-PADDLE_HEIGHT//2)
    
    def move(self, direction:int, dt:float):
        # direction: -1 = left, 0 = no movement, 1 = right
        self.position.x += direction * PADDLE_SPEED * dt
    
    def draw(self, screen:pygame.surface.Surface):
        rect = pygame.Rect(
            int(self.position.x - PADDLE_WIDTH/2),
            int(self.position.y - PADDLE_HEIGHT/2),
            PADDLE_WIDTH,
            PADDLE_HEIGHT
        )
        pygame.draw.rect(screen, PADDLE_COLOUR, rect, width=2, border_radius=PADDLE_HEIGHT//4)


class Ball:
    def __init__(self, paddle:Paddle) -> None:
        self.paddle = paddle

        self.position : pygame.math.Vector2 = pygame.math.Vector2(
            random.randint(BALL_RADIUS, WIN_WIDTH - BALL_RADIUS),
            random.randint(BALL_RADIUS, WIN_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS)
        )

        start_angle : float = random.randint(20, 160)
        self.velocity : pygame.math.Vector2 = pygame.math.Vector2(BALL_SPEED, 0).rotate(-start_angle)

        self.time_since_bounce = pygame.math.Vector2(WIN_WIDTH, WIN_HEIGHT) / BALL_SPEED
    
    def move(self, dt:float) -> bool:
        self.time_since_bounce += pygame.math.Vector2(1, 1) * dt

        # Hit left or right wall
        if (self.position.x < BALL_RADIUS or self.position.x > WIN_WIDTH - BALL_RADIUS) and self.time_since_bounce.x >= WIN_WIDTH/BALL_SPEED*.5:
            self.velocity.x *= -1

            self.time_since_bounce.x = 0

        # Hit top wall
        if self.position.y < BALL_RADIUS and self.time_since_bounce.y >= WIN_HEIGHT/BALL_SPEED*.5:
            self.velocity.y *= -1

            self.time_since_bounce.y = 0
        
        # Hit paddle
        has_hit_paddle = False
        dead = False
        if self.position.y > WIN_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS and self.time_since_bounce.y >= WIN_HEIGHT/BALL_SPEED*.5:
            if self.paddle.position.x - PADDLE_WIDTH/2 < self.position.x < self.paddle.position.x + PADDLE_WIDTH/2:
                has_hit_paddle = True

                angle = 50 * (self.position.x - self.paddle.position.x) / (PADDLE_WIDTH/2)
                self.velocity = pygame.math.Vector2(0, -BALL_SPEED).rotate(angle)

                self.time_since_bounce.y = 0
            else:
                dead = True


        self.position += self.velocity * dt

        return has_hit_paddle, dead

    def draw(self, screen:pygame.surface.Surface) -> None:
        pygame.draw.circle(screen,
                           color=BALL_COLOUR,
                           center=self.position,
                           radius=BALL_RADIUS)



def eval_genomes(genomes:tuple[int,neat.DefaultGenome], config:neat.Config):
    # Initialise games
    balls : list[Ball] = []
    paddles : list[Paddle] = []
    ges : list[neat.DefaultGenome] = []
    nets : list[neat.nn.FeedForwardNetwork] = []

    for id, genome in genomes:
        paddles.append(Paddle())
        balls.append(Ball(paddles[-1]))
        ges.append(genome)
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))

        genome.fitness = 0
    
    # Initialise pygame window
    surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    # Mainloop
    running = True
    loops = 0
    elapsed_time = 0
    while running:
        # Event loop
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Wait for clock
        dt = clock.tick(FPS)/1000
        elapsed_time += dt
        loops += 1

        # print(f"loops: {loops} | dt: {dt} | elapsed time: {elapsed_time} | Ball position: {balls[0].position}")

        # Background fill
        surface.fill(BACKGROUND_COLOUR)

        fitnesses : list[float] = []

        for i, ball in enumerate(balls):
            ges[i].fitness += REWARD_PER_TIME * dt
            fitnesses.append(ges[i].fitness)
            
            # Goal complete
            if ges[i].fitness >= FITNESS_THRESHOLD:
                running = False

            # inputs = (
            #     ball.position.x,
            #     ball.position.y,
            #     ball.velocity.x,
            #     ball.velocity.y,
            #     paddles[i].position.x,
            #     paddles[i].position.y
            # )
            inputs = (
                # ball.velocity.x,
                # ball.velocity.y,
                paddles[i].position.x - ball.position.x,
                paddles[i].position.y - ball.position.y
            )
            output = nets[i].activate(inputs)
            direction = output.index(max(output)) - 1

            paddles[i].move(direction, dt)
            paddles[i].draw(surface)

            paddle_bounce, dead = ball.move(dt)
            ball.draw(surface)

            if paddle_bounce:
                ges[i].fitness += REWARD_PER_BOUNCE # Carrots!
            
            # Dead
            if dead:
                ges[i].fitness -= PENALTY_PER_DEATH * (abs(ball.position.x - paddles[i].position.x) / WIN_WIDTH)

                balls.pop(i)
                paddles.pop(i)
                nets.pop(i)
                ges.pop(i)
        
        pygame.display.set_caption(f"Max fitness: {max(fitnesses):.2f}")

        pygame.display.update()
            
        if len(balls) == 0:
            running = False


def run(config_file):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the overarching population object
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix="./checkpoints/pong/"))

    # Train the network
    winner = p.run(eval_genomes)

    # Display the winning genome.
    print('\n--- Best genome: ---\n{!s}'.format(winner))
    
    node_names = {
        # -1: "Ball x speed",
        # -2: "Ball y speed",
        -1: "x diff",
        -2: "y diff",
        0: "Left",
        1: "Stay",
        2: "Right"
    }
    neat_utils.draw_net(config, winner, True, node_names=node_names)
    neat_utils.draw_net(config, winner, True, node_names=node_names, prune_unused=True, filename="pruned-network")
    neat_utils.plot_stats(stats, ylog=False, view=True)
    neat_utils.plot_species(stats, view=True)

    with open("pong_genome", "w") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    run("pong_config.txt")