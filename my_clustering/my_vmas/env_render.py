from my_clustering.my_vmas.scenarios import navigation_random
from my_interactive_rendering import render_interactively

render_interactively(
        "navigation_clustering",
        control_two_agents=True,
        save_render=True,
        display_info=True,
    )