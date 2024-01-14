import random
import numpy as np
import pygame
from default_vars import WINDOW_WIDTH, WINDOW_HEIGHT, RADIUS, MAX_VERTICES, COMPLEX_WINDOW_CENTER
from default_vars import FIGURE_AMOUNT, FIGURE_COMPLEXITY

class Pattern_generator: ...
class Layer: ...


Baked_positions = list[np.ndarray[complex]]
Pos_2d = tuple[float, float]


class Window_manager:
    def __init__(self) -> None:
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.tick = 1
        self.keep_alive_signal = True
        self.fps = 30
        self.render_styles = {"width": 3}
        self.bg_color = 0, 0, 0

    def event_loop_update(self, pattern: Pattern_generator) -> None:
        for event in pygame.event.get():
            self.single_event_update(event, pattern)

    def single_event_update(self, event: pygame.event.Event, pattern: Pattern_generator) -> None:
        match event.type:
            case pygame.QUIT:
                self.stop()
            case pygame.KEYDOWN:
                self.update_key_down(event.key, pattern)

    def stop(self) -> None:
        self.keep_alive_signal = False

    def update_key_down(self, key, pattern: Pattern_generator) -> None:
        if key == pygame.K_SPACE:
            pattern.generate_random_pattern()

    def render(self, baked: Baked_positions) -> None:
        for connected_figure in baked:
            self._render_connected_figure(connected_figure)

    def _render_connected_figure(self, connected_figure: np.ndarray[complex]) -> None:
        for pos1, pos2 in zip(connected_figure, connected_figure[1:]):
            self._draw_complex_line(pos1, pos2, **self.render_styles)
        self._draw_complex_line(connected_figure[0], connected_figure[-1], **self.render_styles)
    
    def _draw_complex_line(self, pos1: complex, pos2: complex, color=(255,255,255), **line_kwargs) -> None:
        self.draw_line(self._decomplex(pos1), self._decomplex(pos2), color=color, **line_kwargs)
    
    def draw_line(self, pos1: Pos_2d, pos2: Pos_2d, color=(255,255,255), **line_kwargs) -> None:
        pygame.draw.line(self.display, color, pos1, pos2, **line_kwargs)

    @staticmethod
    def _decomplex(x: complex) -> Pos_2d:
        return x.real, x.imag

    def update_scene(self) -> None:
        self.tick += 1
        pygame.display.update()
        self.clock.tick(self.fps)
        real_fps = self.clock.get_fps()
        pygame.display.set_caption(f"FPS: {real_fps:.1f}")
        self.display.fill(self.bg_color)


class Pattern_generator:
    def __init__(self) -> None:
        self.radius = RADIUS
        self.vertices_index = 3
        self.pattern_index = 1
        self.layers = []
        self.offset = 0

    def generate_random_pattern(self) -> None:
        layers = self.generate_aligned_random_layers(FIGURE_AMOUNT)
        self.set_random_offset(layers)
        self.layers = layers

    def set_random_offset(self, layers: list[Layer]) -> None:
        max_vertices_index = self.get_max_vertices_index(layers)
        self.offset = random.randint(0, max_vertices_index * 2) / max_vertices_index / 2

    @staticmethod
    def get_max_vertices_index(layers: list[Layer]) -> int:
        max_vertices_index = 0
        for layer in layers:
            if layer.vertices_index > max_vertices_index:
                max_vertices_index = layer.vertices_index

        return max_vertices_index

    @staticmethod
    def generate_aligned_random_layers(amount: int) -> list[Layer]:
        layers = []
        layers.append(Layer.generate_layer_basis())

        for _ in range(amount - 1):
            aligned_layer = Layer.generate_aligned(layers)
            layers.append(aligned_layer)

        return layers

    @staticmethod
    def generate_random_layers(amount: int) -> list[Layer]:
        return [
            Layer.generate_random_layer()
            for _ in range(amount)
        ]
    
    def bake(self, offset: float) -> Baked_positions:
        baked_result = []
        for layer in self.layers:
            baked_result += self.bake_layer(layer, offset)
        return baked_result

    @classmethod
    def bake_layer(cls, layer: Layer, offset: float) -> Baked_positions:
        baked_figures = []
        for connected_shape in layer.virtual_coords:
            baked_figures.append(cls._bake_virtual(connected_shape + offset, RADIUS))
        return baked_figures

    @staticmethod
    @np.vectorize
    def _bake_virtual(pos: float, radius: float) -> complex:
        relative_normalized_complex_pos = 1j**(4*pos)
        relative_complex_pos = relative_normalized_complex_pos * radius
        complex_pos = relative_complex_pos + COMPLEX_WINDOW_CENTER
        return complex_pos


class Layer:
    def __init__(self, vertices_index: int, pattern_index: int, _raw=False) -> None:
        self.vertices_index = vertices_index
        self.pattern_index = pattern_index
        self.radius = RADIUS

        if not _raw:
            self.virtual_coords = self.generate_virtual_coords()
        else:
            self.virtual_coords = None

    @classmethod
    def generate_random_layer(cls) -> Layer:
        layer = Layer._raw()
        cls.randomize_layer(layer)
        layer.virtual_coords = layer.generate_virtual_coords()
        return layer
    
    @classmethod
    def generate_layer_basis(cls) -> Layer:
        vertices_basis = cls.generate_vertices_basis_for_aligned_layers()
        layer = Layer._raw()
        layer.vertices_index = vertices_basis
        layer.randomize_pattern_index()
        layer.virtual_coords = layer.generate_virtual_coords()

        return layer

    @classmethod
    def generate_vertices_basis_for_aligned_layers(cls) -> int:
        max_for_basis = int(MAX_VERTICES**(1/FIGURE_COMPLEXITY))

        vertices_basis = 1
        for _ in range(FIGURE_COMPLEXITY):
            vertices_basis *= random.randint(2, max_for_basis)

        return vertices_basis

    @classmethod
    def _raw(cls) -> Layer:
        layer = Layer(0, 0, True)
        return layer

    def randomize_layer(layer) -> None:
        layer.randomize_vertices_index()
        layer.randomize_pattern_index()
    
    def randomize_vertices_index(self) -> None:
        self.vertices_index = random.randint(3, MAX_VERTICES)
    
    def randomize_pattern_index(self) -> None:
        # patternIndex repeats after being a half
        # of verticesIndex, if it is exactly a half,
        # a boring pattern of a star occurs, so we ignore it.
        possible_max = self.max_for_pattern_index()
        self.pattern_index = random.randrange(1, possible_max)

    def max_for_pattern_index(self) -> int:
        return precise_math_round(self.vertices_index/2)

    def generate_virtual_coords(self) -> None:
        wrapper = Virtual_coords_wrapper(self.vertices_index, self.pattern_index)
        return wrapper.generate()

    @classmethod
    def generate_aligned(cls, previous_layers: list[Layer]) -> Layer:
        vertices_indexes_matches = cls._get_available_vertices_indexes(previous_layers)
        pattern_indexes = cls._get_pattern_indexes(previous_layers)
        layer = cls._get_random_aligned_layer_with_choice_of_vertices_indexes(
            vertices_indexes_matches, pattern_indexes
        )

        return layer
    
    @classmethod
    def _get_random_aligned_layer_with_choice_of_vertices_indexes(
            cls,
            vertices_indexes: list[int],
            taken_pattern_indexes: list[int]) -> Layer:
        choice = random.choice(vertices_indexes)
        return cls._get_random_aligned_layer_with_vertices_index(choice, taken_pattern_indexes)

    @classmethod
    def _get_random_aligned_layer_with_vertices_index(
            cls, vertices_index: int,
            taken_pattern_indexes: list[int]) -> Layer:
        layer = Layer._raw()
        layer.vertices_index = vertices_index
        layer.randomize_pattern_index_without(taken_pattern_indexes)
        layer.virtual_coords = layer.generate_virtual_coords()
        return layer
    
    def randomize_pattern_index_without(self, indexes_to_ignore: list[int]) -> None:
        max_pattern_index = self.max_for_pattern_index()
        matches = set(range(1, max_pattern_index))
        matches = list(matches.difference(indexes_to_ignore))
        if matches:
            self.pattern_index = random.choice(matches)
        else:
            self.pattern_index = random.choice(indexes_to_ignore)

    @classmethod
    def _get_available_vertices_indexes(cls, layers: list[Layer]) -> list[int]:
        indexes = cls._get_vertices_indexes(layers)
        matches = []
        for possible_match in cls._get_possible_matches():
            if cls._check_index_for_alignment(possible_match, indexes):
                matches.append(possible_match)
        return matches
    
    @classmethod
    def _get_vertices_indexes(cls, layers: list[Layer]) -> list[int]:
        vertices_indexes = []
        for layer in layers:
            vertices_indexes.append(layer.vertices_index)

        return vertices_indexes
    
    @classmethod
    def _get_pattern_indexes(cls, layers: list[Layer]) -> list[int]:
        pattern_indexes = []
        for layer in layers:
            pattern_indexes.append(layer.pattern_index)

        return pattern_indexes

    @classmethod
    def _get_possible_matches(cls) -> range:
        return range(3, MAX_VERTICES + 1)

    @classmethod
    def _check_index_for_alignment(cls, index_for_check: int, matches: list[int]) -> bool:
        for match in matches:
            if (match % index_for_check != 0 and
                index_for_check % match != 0):
                return False
        return True


class Virtual_coords_wrapper:
    def __init__(self, vertices_index: int, pattern_index: int) -> None:
        self.vertices_index = vertices_index
        self.pattern_index = pattern_index
        self.step = self.calculate_step()
        self.position = self.step
        self.ready_positions = [[0, self.step]]

    def calculate_step(self) -> float:
        return self.pattern_index / self.vertices_index

    def generate(self) -> list[np.ndarray[float]]:
        self._setup_base()
        self._multiple_figure_case()
        return self.ready_positions

    def _setup_base(self) -> None:
        self._generate_base()
        self._optimize_base()

    def _optimize_base(self) -> None:
        self.ready_positions[0] = np.array(self.ready_positions[0])

    def _generate_base(self) -> None:
        while self._pos_in_bounds(self.position):
            self._update_position()
        del self.ready_positions[0][-1]

    @classmethod
    def _pos_in_bounds(cls, position) -> bool:
        # Is in 0..1 check
        return position > 1e-3 and position < 1-1e-3

    def _update_position(self) -> None:
        self.position = (self.position + self.step) % 1.
        self.ready_positions[0].append(self.position)

    def _multiple_figure_case(self) -> None:
        if self.multiple_figures():
            self._handle_multiple_figures()

    def multiple_figures(self) -> bool:
        if self.pattern_index == 1:
            return False
        return self.vertices_index % self.pattern_index == 0
    
    def _handle_multiple_figures(self) -> None:
        base = np.array(self.ready_positions[0])
        for i in range(1, self.pattern_index):
            self.ready_positions.append((base + i / self.vertices_index))


def precise_math_round(value: float) -> int:
    return int(value + 0.5)
