<!-- TOC -->
* [Movie Night - Blender Stuff](#movie-night---blender-stuff)
* [Shell](#shell)
* [Loom](#loom)
  * [Preferences](#preferences)
  * [Script](#script)
* [Blender Python API](#blender-python-api)
  * [Load `.blend` File](#load-blend-file)
* [Command Line Render](#command-line-render)
  * [With `--python-expr <expr>`](#with---python-expr-expr)
  * [With `--python <file>`](#with---python-file)
  * [Blender as a Review Player](#blender-as-a-review-player)
  * [Render Settings](#render-settings)
  * [File Output Nodes](#file-output-nodes)
* [Known Issues](#known-issues)
<!-- TOC -->

---

# Movie Night - Blender Stuff

Resources:
- [Stack Overflow - How can i find the python triggered by the GUI?](https://blender.stackexchange.com/a/164008)
- [Python API Documentation (4.4.x)](https://docs.blender.org/api/4.4/)

Enable Python Tooltips
- Preferences -> Interface
  - [x] Developer Extrase
  - [x] Python Tooltips

Blender Context Menu:
- Copy Data Path: `Shift` + `Ctrl` + `C`
- Copy Full Data Path: `Shift` + `Ctrl` + `Alt` + `C`

Target Image Nodes:
- `bpy.data.images["AYON_SCREENSHOT.GENERIC"]`
- `bpy.data.images["AYON_BARCODE.GENERIC"]`
- `bpy.data.images["AYON_POSTER.GENERIC"]`
- `bpy.data.images["AYON_DATE.GENERIC"]`

Target attribute:
- `filepath`

# Shell

```shell
nix develop github:michimussato/flake-prod#blender-4-4-3-devshell --command fhs
```

# Loom

## Preferences

Render -> Loom -> Loom Preferences

## Script

Default Script output path:
- `/home/michael/.local/share/AYON/addons/blender_1.1.9/ayon_blender/blender_addon/loom-render-temp.sh`

> [!TIP]
> 
> For reference:
> 
> ```shell
> #! /bin/sh
> export BLENDER_USER_SCRIPTS="/home/michael/.local/share/AYON/addons/blender_1.1.9/ayon_blender/blender_addon"
> "/data/share/tools/blender-4.4.3-linux-x64/blender" -b "/data/share/projects/ayon/MovieNight/shot/work/rendering/MoN_shot_rendering_v010_4-4-3.blend" --python-expr "import bpy;bpy.ops.render.image_sequence(frames='1001-1059', isolate_numbers=False,render_silent=True, digits=4, render_preset='EMPTY')" -t 8
> read -n1 -r -p "Press any key to continue..." key
> exit
> ```

# Blender Python API

## Load `.blend` File

```shell
MOVIE_NIGHT_BLEND_FILE="/data/share/projects/ayon/MovieNight/shot/work/rendering/MoN_shot_rendering_v034_4-4-3.blend"
```

> [!TIP]
>
> **Application Handlers**
>
> Reference:
> - [Persistent Handler](https://docs.blender.org/api/4.4/bpy.app.handlers.html#persistent-handler-example)
> 
> ```python
> import bpy
> from bpy.app.handlers import persistent
> 
> 
> @persistent
> def load_handler(dummy):
>     print("Load Handler:", bpy.data.filepath)
> 
> 
> bpy.app.handlers.load_post.append(load_handler)
> ```

# Command Line Render

```python
import bpy

bpy.ops.render.image_sequence(
    frames='1001-1059', 
    isolate_numbers=False,
    render_silent=True, 
    digits=4, 
    render_preset='EMPTY',
)
```

## With `--python-expr <expr>`

```shell
export TMUX_SESSION="MoN_shot_rendering_v034_4-4-3"
export BLENDER_USER_SCRIPTS="/home/michael/.local/share/AYON/addons/blender_1.1.9/ayon_blender/blender_addon"
export MOVIE_NIGHT_BLEND_FILE="/data/share/projects/ayon/MovieNight/shot/work/rendering/${TMUX_SESSION}.blend"

export AYON_SCREENSHOT="/data/share/projects/ayon/MovieNight/shot/publish/texture/textureBadSantaScreenshot/hero/MoN_shot_textureBadSantaScreenshot_hero.jpg"
export AYON_DATE="/data/share/projects/ayon/MovieNight/shot/publish/texture/textureBadSantaDate/hero/MoN_shot_textureBadSantaDate_hero.png"
export AYON_BARCODE="/data/share/projects/ayon/MovieNight/shot/publish/texture/textureBadSantaBarcode/hero/MoN_shot_textureBadSantaBarcode_hero.png"
export AYON_POSTER="/data/share/projects/ayon/MovieNight/shot/publish/texture/textureBadSantaPoster/hero/MoN_shot_textureBadSantaPoster_hero.jpg"

tmux new -d -s ${TMUX_SESSION} 'nix develop github:michimussato/flake-prod#blender-4-4-3-devshell --command fhs'
tmux a -t ${TMUX_SESSION}

blender \
    --background \
    ${MOVIE_NIGHT_BLEND_FILE} \
    --python-expr "import bpy; bpy.ops.render.image_sequence(frames='1001-1059', isolate_numbers=False, render_silent=True, digits=4, render_preset='EMPTY')" \
    --threads 8 \
    2>&1 | tee "/data/share/projects/ayon/MovieNight/shot/work/rendering/render/${TMUX_SESSION}_render.log"
```

## With `--python <file>`

```shell
export PY_TMP=$(mktemp --suffix=".py")

tee ${PY_TMP} < "./blender.py" | bat --file-name ${PY_TMP} --language Python -

echo "${PY_TMP}"
```

```shell
export TMUX_SESSION="MoN_shot_rendering_v035_4-4-3"
export BLENDER_USER_SCRIPTS="/home/michael/.local/share/AYON/addons/blender_1.1.9/ayon_blender/blender_addon"
export MOVIE_NIGHT_BLEND_FILE="/data/share/projects/ayon/MovieNight/shot/work/rendering/${TMUX_SESSION}.blend"

export FRAMES="1001-1059"

# Texture Sets must be published on to Ayon
# Only 'hero' versions can be used

# Example Texture Set: The Wolf of Wall Street
export MOVIE_NIGHT_PROJECT="TheWolfOfWallStreet"
# Example Texture Set: Bad Santa
export MOVIE_NIGHT_PROJECT="BadSanta"

export AYON_SCREENSHOT="/data/share/projects/ayon/MovieNight/shot/publish/texture/texture${MOVIE_NIGHT_PROJECT}Screenshot/hero/MoN_shot_texture${MOVIE_NIGHT_PROJECT}Screenshot_hero.jpg"
export AYON_DATE="/data/share/projects/ayon/MovieNight/shot/publish/texture/texture${MOVIE_NIGHT_PROJECT}Date/hero/MoN_shot_texture${MOVIE_NIGHT_PROJECT}Date_hero.png"
export AYON_BARCODE="/data/share/projects/ayon/MovieNight/shot/publish/texture/texture${MOVIE_NIGHT_PROJECT}Barcode/hero/MoN_shot_texture${MOVIE_NIGHT_PROJECT}Barcode_hero.png"
export AYON_POSTER="/data/share/projects/ayon/MovieNight/shot/publish/texture/texture${MOVIE_NIGHT_PROJECT}Poster/hero/MoN_shot_texture${MOVIE_NIGHT_PROJECT}Poster_hero.jpg"

file ${AYON_SCREENSHOT}
file ${AYON_DATE}
file ${AYON_BARCODE}
file ${AYON_POSTER}

tmux new -d -s ${TMUX_SESSION} 'nix develop github:michimussato/flake-prod#blender-4-4-3-devshell --command fhs'
tmux a -t ${TMUX_SESSION}


RENDER_LOG="/data/share/projects/ayon/MovieNight/shot/work/rendering/render/${MOVIE_NIGHT_PROJECT}_${TMUX_SESSION}/render.log"
mkdir -p $(dirname ${RENDER_LOG})

set +x
env | tee ${RENDER_LOG}

blender \
    --background \
    ${MOVIE_NIGHT_BLEND_FILE} \
    --python "${PY_TMP}" \
    --threads 8 \
    2>&1 | tee --append ${RENDER_LOG}
set -x

# Todo:
#  - [ ] possibly set `--render-output <output_dir>` via commandline manually
#  - [ ] possibly set frame range via commandline manually

# Argument Order:
#         Arguments are executed in the order they are given. eg:
#         # blender --background test.blend --render-frame 1 --render-output "/tmp"
#         ...will not render to '/tmp' because '--render-frame 1' renders before the output path is set.
#         # blender --background --render-output /tmp test.blend --render-frame 1
#         ...will not render to '/tmp' because loading the blend-file overwrites the render output that was set.
#         # blender --background test.blend --render-output /tmp --render-frame 1
#         ...works as expected.
```

## Blender as a Review Player

Reference:
- [Animation Player](https://docs.blender.org/manual/en/4.4/render/output/animation_player.html)

```shell
nix develop github:michimussato/flake-prod#blender-4-4-3-devshell --command fhs
blender \
    -a \
    -f 24 1.0 \
    -j 1 \
    -s 1001 \
    -e 1059 \
    /data/share/projects/ayon/MovieNight/shot/work/rendering/render/MoN_shot_rendering_v034_4-4-3/Poster/Image{1001..1059}.png
```

## Render Settings

| Field       | Data Path                              | Default Value |   |
|-------------|----------------------------------------|---------------|---|
| Frame Start | `bpy.data.scenes["Scene"].frame_start` | `1001`        |   |
| Frame End   | `bpy.data.scenes["Scene"].frame_end`   | `1059`        |   |
|             |                                        |               |   |


## File Output Nodes

Render Out
- Key: `bpy.data.scenes["Scene"].render.filepath`
- Value: `//render/$BLEND/LineArt/api_022-01_192_0001.exr`

AnD
- Key: `bpy.data.scenes["Scene"].node_tree.nodes["FILE_OUTPUT.AND"].base_path`
- Value: `//render/$BLEND/AnD/`

Poster 
- Key: `bpy.data.scenes["Scene"].node_tree.nodes["FILE_OUTPUT.POSTER"].base_path` 
- Value: `//render/$BLEND/Poster/`

---

# Known Issues

```
Traceback (most recent call last):
  File "/data/share/tools/blender-4.4.3-linux-x64/4.4/scripts/modules/bpy/utils/__init__.py", line 142, in _test_import
    mod = __import__(module_name)
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/michael/.local/share/AYON/addons/blender_1.1.9/ayon_blender/blender_addon/startup/init.py", line 5, in <module>
    raise ModuleNotFoundError(
ModuleNotFoundError: No module named 'ayon_core'. This usually happens if Blender is launched without the --python-use-system-env argument. Make sure you have the Blender application 'Arguments' configured correctly so that Blender has access to the launched context's PYTHONPATH.
```