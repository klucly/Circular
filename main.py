import math
import pygame
import numpy as np
import random
from time import perf_counter_ns

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
RADIUS = 300
MAX_VERTICES = 10
LINE_WIDTH = 2
MIN_RADIUS = 50

class Main:
    def __init__(self) -> None:
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.radius = RADIUS
        self.vertices = 3
        self.pattern = 1
        self.cur_raw_coords = []
        self.tick = 1

        self.render(self.vertices, self.pattern, self.radius)

        while self.mainloop(): pass

    def mainloop(self) -> None:
        time1 = perf_counter_ns()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                self.generate_structure()

        self.display.fill((0,0,0))
        coord_count = 0
        for raw_coords, radius, speed in self.cur_raw_coords:
            k = 1
            for c in raw_coords.shape:
                k *= c
            coord_count += k
            coords = self.to_screen_coords(raw_coords+speed*self.tick/120, radius)
            self.draw_coords(coords)
            pygame.draw.circle(self.display, (255, 255, 255), (WINDOW_WIDTH//2, WINDOW_HEIGHT//2), radius, LINE_WIDTH)

        self.tick += 1
        dtime = (perf_counter_ns()-time1)/1e6
        

        pygame.display.update()
        self.clock.tick(60)
        pygame.display.set_caption(f"Circular {dtime}ms, {coord_count}")
        return True
    

    def generate_raw_coords(self, vertices: int, pattern: int) -> list:
        step = pattern / vertices
        coordinates = [[0, step]]

        position = step

        while position > 1e-3 and position < 1-1e-3:
            position = (position + step) % 1.
            coordinates[0].append(position)
        coordinates[0].pop(-1)

        positions_count = len(coordinates[0])

        if positions_count < vertices:
            base = np.array(coordinates[0])
            for i in range(1, pattern):
                coordinates.append((base + i / vertices))

        coordinates = np.array(coordinates)

        return coordinates

    def to_screen_coords(self, coords, radius):
        return np.vectorize(lambda x: 1j**(4*x)*radius+WINDOW_WIDTH//2+WINDOW_HEIGHT//2*1j)(coords)

    def draw_coords(self, coords):
        for line_set in coords:
            for i, coord in enumerate(line_set):
                coord: complex
                if i == line_set.shape[0]-1:
                    pygame.draw.line(self.display, (255, 255, 255), (coord.imag, coord.real), (line_set[0].imag, line_set[0].real), LINE_WIDTH)
                    continue
                
                pygame.draw.line(self.display, (255, 255, 255), (coord.imag, coord.real), (line_set[i+1].imag, line_set[i+1].real), LINE_WIDTH)
                
    def render(self, vertices: int, pattern: int, radius: float) -> np.ndarray:
        raw_coords = self.generate_raw_coords(vertices, pattern)
        coords = self.to_screen_coords(raw_coords, radius)

        self.draw_coords(coords)
        return raw_coords
    
    def generate_structure(self):
        normal_round = lambda x: int(x + 0.5)

        self.display.fill((0,0,0))
        self.vertices = random.randint(3, MAX_VERTICES)
        self.pattern = random.randrange(1, normal_round(self.vertices/2), 1)

        self.cur_raw_coords = [(self.render(self.vertices, self.pattern, self.radius), self.radius, (random.random()*2-1)/self.vertices*3)]

        prev_vertices = self.vertices
        prev_pattern = self.pattern
        prev_inner = self.radius
        inner = 100

        while inner > MIN_RADIUS:
            innervertices = random.randint(3, MAX_VERTICES)
            innerpattern = random.randrange(1, normal_round(innervertices/2), 1)

            inner = math.cos(math.pi*prev_pattern/prev_vertices) * prev_inner

            self.cur_raw_coords.append((self.render(innervertices, innerpattern, inner), inner, ((random.random()*2-1)/self.vertices*3)))
            
            prev_inner = inner
            prev_vertices = innervertices
            prev_pattern = innerpattern

if __name__ == "__main__":
    Main()
