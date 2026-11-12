import os
import pathlib
import bpy


OVERRIDE_LOOM_OUTPUT_PATHS: bool = True


_blend_file: str = os.environ["MOVIE_NIGHT_BLEND_FILE"]
blend_file: pathlib.Path = pathlib.Path(_blend_file)

assert blend_file.exists(), "Blend file %s does not exist" % blend_file.as_posix()

bpy.ops.wm.open_mainfile(filepath=blend_file.as_posix())

nodes = [
    "AYON_SCREENSHOT.GENERIC",
    "AYON_BARCODE.GENERIC",
    "AYON_POSTER.GENERIC",
    "AYON_DATE.GENERIC",
]

frames = os.environ["FRAMES"]

# Always use "hero" versions
screenshot: str = os.environ["AYON_SCREENSHOT"]
date: str = os.environ["AYON_DATE"]
poster: str = os.environ["AYON_POSTER"]
barcode: str = os.environ["AYON_BARCODE"]

textures = {
    "AYON_SCREENSHOT.GENERIC": pathlib.Path(screenshot),
    "AYON_BARCODE.GENERIC": pathlib.Path(barcode),
    "AYON_POSTER.GENERIC": pathlib.Path(poster),
    "AYON_DATE.GENERIC": pathlib.Path(date),
}

# set the texture file paths in the texture nodes
for node in nodes:
    bpy.data.images[node].filepath = textures[node].as_posix()

# loom
# - https://github.com/p2or/blender-loom#loom
# LOOM_PG_scene_settings.frame_input
# bpy.data.scenes["Scene"].loom.guess_frames()
bpy.data.scenes["Scene"].loom.frame_input = str(frames)
bpy.data.scenes["Scene"].loom.command_line = False

if OVERRIDE_LOOM_OUTPUT_PATHS:
    blend_project: str = os.environ["TMUX_SESSION"]  # the blend file without .blend extension
    movie_night_project: str = os.environ["MOVIE_NIGHT_PROJECT"]

    # `//` means `relative` to the .blend file
    bpy.data.scenes["Scene"].render.filepath = \
        f"//render/{movie_night_project}_{blend_project}/LineArt/api_022-01_192_0001.exr"  # Todo: change the file name after test
    bpy.data.scenes["Scene"].node_tree.nodes["FILE_OUTPUT.AND"].base_path = \
        f"//render/{movie_night_project}_{blend_project}/AnD/"
    bpy.data.scenes["Scene"].node_tree.nodes["FILE_OUTPUT.POSTER"].base_path = \
        f"//render/{movie_night_project}_{blend_project}/Poster/"

bpy.ops.render.image_sequence(
    frames=str(frames),
    isolate_numbers=False,
    render_silent=True,
    digits=4,
    validate_scene=False,
    render_preset='EMPTY',
)
