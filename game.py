from gesture import gesture_handler
import pyglet

window = pyglet.window.Window(width=800, height=600, caption="Spellweave v0.5")

player = pyglet.media.Player()
song = pyglet.media.load(r"assets\loading_screen.mp3", streaming=True)
player.queue(song)
player.loop = True
player.play()

gesture_stream = gesture_handler()

gesture = {}
frame_sprite = None

label = pyglet.text.Label(
    text="No hand detected",
    font_name="Arial",
    font_size=24,
    x=window.width // 2,
    y=window.height - 50,
    anchor_x="center",
    anchor_y="center"
)


def read_camera(dt):
    global gesture
    global frame_sprite

    try:
        gesture, rgb_frame = next(gesture_stream)

    except StopIteration:
        gesture = {}
        label.text = "camera stopped 😞"
        pyglet.clock.unschedule(read_camera)
        return

    if gesture:
        label.text = str(gesture)
    else:
        label.text = "No hand detected"

    frame_height, frame_width = rgb_frame.shape[:2]
    frame_image = pyglet.image.ImageData(
        frame_width,
        frame_height,
        "RGB",
        rgb_frame.tobytes(),
        pitch=-frame_width * 3
    )

    if frame_sprite is None:
        frame_sprite = pyglet.sprite.Sprite(frame_image, x=10, y=10)
        frame_sprite.scale = 300 / frame_width
    else:
        frame_sprite.image = frame_image

@window.event
def on_draw():
    window.clear()

    if frame_sprite is not None:
        frame_sprite.draw()

    label.draw()

pyglet.clock.schedule_interval(read_camera, 1 / 60) #60 fps

if __name__ == "__main__":
    pyglet.app.run()