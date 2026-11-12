import os
import pathlib
import bpy


OVERRIDE_LOOM_OUTPUT_PATHS: bool = True


_blend_file: str = os.environ["MOVIE_NIGHT_BLEND_FILE"]
blend_file: pathlib.Path = pathlib.Path(_blend_file)

assert blend_file.exists(), "Blend file %s does not exist" % blend_file.as_posix()

bpy.ops.wm.open_mainfile(
    filepath=blend_file.as_posix()
)

frames = os.environ["FRAMES"]

# Always use "hero" versions
screenshot: str = os.environ["AYON_SCREENSHOT"]
date: str = os.environ["AYON_DATE"]
poster: str = os.environ["AYON_POSTER"]
barcode: str = os.environ["AYON_BARCODE"]

texture_set = {
    # "NODE_NAME": texture_source_path,
    "AYON_SCREENSHOT.GENERIC": pathlib.Path(screenshot),
    "AYON_BARCODE.GENERIC": pathlib.Path(barcode),
    "AYON_POSTER.GENERIC": pathlib.Path(poster),
    "AYON_DATE.GENERIC": pathlib.Path(date),
}

# loom
# - https://github.com/p2or/blender-loom#loom
# LOOM_PG_scene_settings.frame_input
# bpy.data.scenes["Scene"].loom.guess_frames()
bpy.data.scenes["Scene"].loom.frame_input = str(frames)
bpy.data.scenes["Scene"].loom.command_line = False

# Todo:
#  - [x] use absoute path for MOVIE_NIGHT_TEXTURE node
#  - [x] publish to Ayon
#  - [ ] publish `.xcf` workfile (is source for `date` and `movie night` textures) to Ayon
#        `/home/michael/Documents/Obsidian/Michael/Projects/Ap n Daun Movie Night/fonts/kalieb_luxury_font_purchased_003.xcf`
# bpy.data.images["MOVIE_NIGHT_TEXTURE"].filepath = "/data/share/projects/ayon/MovieNight/shot/publish/texture/textureMovieNightFontTexture/hero/MoN_shot_textureMovieNightFontTexture_hero.png"

if OVERRIDE_LOOM_OUTPUT_PATHS:
    movie_night_project: str = os.environ["MOVIE_NIGHT_PROJECT"]

    # Default render root is:
    # `//render`
    # `//` means `relative to the .blend file`
    #
    # Override here to set absolute path
    # so that the render does not end up
    # where the `hero` blend file lives
    # (`/data/share/projects/ayon/MovieNight/shot/publish/workfile/workfileRendering/hero/`).
    #
    # For now, we set to
    # `/data/share/projects/ayon/MovieNight/shot/work/rendering/render/`
    # but this is arbitrary. Maybe there is
    # a better target directory.
    render_root: pathlib.Path = pathlib.Path("/data/share/projects/ayon/MovieNight/shot/work/rendering/render/")
    render_root.mkdir(
        exist_ok=True,
        parents=True,
    )

    bpy.data.scenes["Scene"].render.filepath = \
        render_root.joinpath(f"{movie_night_project}/LineArt/api_022-01_192_0001.exr").as_posix()  # Todo: change the file name after test
    bpy.data.scenes["Scene"].node_tree.nodes["FILE_OUTPUT.AND"].base_path = \
        render_root.joinpath(f"{movie_night_project}/AnD/").as_posix()
    bpy.data.scenes["Scene"].node_tree.nodes["FILE_OUTPUT.POSTER"].base_path = \
        render_root.joinpath(f"{movie_night_project}/Poster/").as_posix()

# set the texture file paths in the texture nodes
# for node in nodes:
for node, texture_path in texture_set.items():
    assert texture_path.exists(), "Texture file %s does not exist" % texture_path.as_posix()
    bpy.data.images[node].filepath = texture_path.as_posix()

bpy.ops.render.image_sequence(
    frames=str(frames),
    isolate_numbers=False,
    render_silent=True,
    digits=4,
    validate_scene=False,
    render_preset='EMPTY',
)
