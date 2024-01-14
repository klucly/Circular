from win_pattern import Window_manager, Pattern_generator


def run_app() -> None:
    window = Window_manager()
    pattern = Pattern_generator()
    start_win_pattern_loop(window, pattern)


def start_win_pattern_loop(window: Window_manager, pattern: Pattern_generator) -> None:
    while window.keep_alive_signal:
        update_win_pattern_loop(window, pattern)


def update_win_pattern_loop(window: Window_manager, pattern: Pattern_generator) -> None:
    window.event_loop_update(pattern)
    display_update(window, pattern)


def display_update(window: Window_manager, pattern: Pattern_generator) -> None:
    render_pattern(window, pattern)
    window.update_scene()


def render_pattern(window: Window_manager, pattern: Pattern_generator) -> None:
    baked = pattern.bake(pattern.offset)
    window.render(baked)


if __name__ == "__main__":
    run_app()
