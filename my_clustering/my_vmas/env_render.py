from scenarios import navigation
from my_interactive_rendering import render_interactively

render_interactively(
        "navigation",
        control_two_agents=True,
        save_render=True,
        display_info=True,
    )