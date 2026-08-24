"""
2D rendering framework with headless PIL software fallback
"""
from __future__ import division
import os
import math
import sys
import numpy as np
from PIL import Image as PILImage, ImageDraw

HAVE_PYGLET = False
try:
    # pyrefly: ignore [missing-import]
    import pyglet
    # pyrefly: ignore [missing-import]
    from pyglet.gl import *
    HAVE_PYGLET = True
except Exception:
    HAVE_PYGLET = False

RAD2DEG = 57.29577951308232


def get_display(spec):
    if not HAVE_PYGLET or spec is None:
        return None
    try:
        import six
        if isinstance(spec, six.string_types):
            return pyglet.canvas.Display(spec)
    except Exception:
        pass
    return None


class Attr(object):
    def enable(self):
        pass
    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.translation = (float(translation[0]), float(translation[1]))
        self.rotation = float(rotation)
        self.scale = (float(scale[0]), float(scale[1]))

    def enable(self):
        if HAVE_PYGLET:
            try:
                glPushMatrix()
                glTranslatef(self.translation[0], self.translation[1], 0)
                glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
                glScalef(self.scale[0], self.scale[1], 1)
            except Exception:
                pass

    def disable(self):
        if HAVE_PYGLET:
            try:
                glPopMatrix()
            except Exception:
                pass

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

    def apply_to_point(self, x, y):
        # Scale -> Rotate -> Translate
        sx = x * self.scale[0]
        sy = y * self.scale[1]
        cos_t = math.cos(self.rotation)
        sin_t = math.sin(self.rotation)
        rx = sx * cos_t - sy * sin_t
        ry = sx * sin_t + sy * cos_t
        return rx + self.translation[0], ry + self.translation[1]


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        if HAVE_PYGLET:
            try:
                glColor4f(*self.vec4)
            except Exception:
                pass


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        if HAVE_PYGLET:
            try:
                glLineWidth(self.stroke)
            except Exception:
                pass


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        if HAVE_PYGLET:
            for attr in reversed(self.attrs):
                attr.enable()
            self.render1()
            for attr in self.attrs:
                attr.disable()

    def render1(self):
        pass

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1.0):
        self._color.vec4 = (r, g, b, alpha)

    def get_transforms(self):
        return [a for a in self.attrs if isinstance(a, Transform)]

    def apply_transforms(self, x, y, extra_transforms=None):
        transforms = (extra_transforms or []) + self.get_transforms()
        cx, cy = x, y
        for t in transforms:
            cx, cy = t.apply_to_point(cx, cy)
        return cx, cy

    def get_color_rgba(self):
        r, g, b, a = self._color.vec4
        return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

    def get_linewidth(self):
        for a in self.attrs:
            if isinstance(a, LineWidth):
                return int(max(1, a.stroke))
        return 1

    def draw_software(self, overlay_draw, to_screen, extra_transforms=None):
        pass


class FilledPolygon(Geom):
    def __init__(self, v):
        super(FilledPolygon, self).__init__()
        self.v = v

    def render1(self):
        if not HAVE_PYGLET:
            return
        try:
            if len(self.v) == 4:
                glBegin(GL_QUADS)
            elif len(self.v) > 4:
                glBegin(GL_POLYGON)
            else:
                glBegin(GL_TRIANGLES)
            for p in self.v:
                glVertex3f(p[0], p[1], 0)
            glEnd()

            color = (self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
            glColor4f(*color)
            glBegin(GL_LINE_LOOP)
            for p in self.v:
                glVertex3f(p[0], p[1], 0)
            glEnd()
        except Exception:
            pass

    def draw_software(self, overlay_draw, to_screen, extra_transforms=None):
        screen_points = []
        for p in self.v:
            wx, wy = self.apply_transforms(p[0], p[1], extra_transforms)
            sx, sy = to_screen(wx, wy)
            screen_points.append((sx, sy))
        if len(screen_points) >= 3:
            fill_color = self.get_color_rgba()
            outline_color = (int(fill_color[0] * 0.5), int(fill_color[1] * 0.5), int(fill_color[2] * 0.5), fill_color[3])
            overlay_draw.polygon(screen_points, fill=fill_color, outline=outline_color)


class PolyLine(Geom):
    def __init__(self, v, close):
        super(PolyLine, self).__init__()
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        if not HAVE_PYGLET:
            return
        try:
            glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
            for p in self.v:
                glVertex3f(p[0], p[1], 0)
            glEnd()
        except Exception:
            pass

    def set_linewidth(self, x):
        self.linewidth.stroke = x

    def draw_software(self, overlay_draw, to_screen, extra_transforms=None):
        screen_points = []
        for p in self.v:
            wx, wy = self.apply_transforms(p[0], p[1], extra_transforms)
            sx, sy = to_screen(wx, wy)
            screen_points.append((sx, sy))
        if len(screen_points) >= 2:
            if self.close:
                screen_points.append(screen_points[0])
            overlay_draw.line(screen_points, fill=self.get_color_rgba(), width=self.get_linewidth())


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        super(Line, self).__init__()
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        if not HAVE_PYGLET:
            return
        try:
            glBegin(GL_LINES)
            glVertex2f(*self.start)
            glVertex2f(*self.end)
            glEnd()
        except Exception:
            pass

    def draw_software(self, overlay_draw, to_screen, extra_transforms=None):
        wx1, wy1 = self.apply_transforms(self.start[0], self.start[1], extra_transforms)
        wx2, wy2 = self.apply_transforms(self.end[0], self.end[1], extra_transforms)
        sx1, sy1 = to_screen(wx1, wy1)
        sx2, sy2 = to_screen(wx2, wy2)
        overlay_draw.line([(sx1, sy1), (sx2, sy2)], fill=self.get_color_rgba(), width=self.get_linewidth())


class Compound(Geom):
    def __init__(self, gs):
        super(Compound, self).__init__()
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()

    def draw_software(self, overlay_draw, to_screen, extra_transforms=None):
        my_transforms = (extra_transforms or []) + self.get_transforms()
        for g in self.gs:
            g.draw_software(overlay_draw, to_screen, extra_transforms=my_transforms)


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class SoftwareViewer(object):
    """Pure CPU / PIL 2D renderer for headless servers and Docker environments."""
    def __init__(self, width=700, height=700, display=None):
        self.width = width
        self.height = height
        self.geoms = []
        self.onetime_geoms = []
        self.left = -1.0
        self.right = 1.0
        self.bottom = -1.0
        self.top = 1.0
        self.transform = Transform()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        self.left = float(left)
        self.right = float(right)
        self.bottom = float(bottom)
        self.top = float(top)

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def close(self):
        pass

    def render(self, return_rgb_array=False):
        # Create base white canvas
        base_img = PILImage.new("RGBA", (self.width, self.height), (255, 255, 255, 255))
        # Create overlay layer for alpha compositing
        overlay = PILImage.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        scalex = self.width / (self.right - self.left)
        scaley = self.height / (self.top - self.bottom)

        def to_screen(x, y):
            sx = (x - self.left) * scalex
            sy = self.height - (y - self.bottom) * scaley
            return sx, sy

        all_geoms = list(self.geoms) + list(self.onetime_geoms)
        for geom in all_geoms:
            geom.draw_software(draw, to_screen)

        self.onetime_geoms = []
        combined = PILImage.alpha_composite(base_img, overlay).convert("RGB")
        if return_rgb_array:
            return np.array(combined, dtype=np.uint8)
        return None

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom


class Viewer(object):
    """Hybrid Viewer: Uses Pyglet when available, otherwise falls back to SoftwareViewer."""
    def __new__(cls, width, height, display=None):
        if HAVE_PYGLET:
            try:
                inst = super(Viewer, cls).__new__(cls)
                display_obj = get_display(display)
                inst.width = width
                inst.height = height
                inst.window = pyglet.window.Window(width=width, height=height, display=display_obj)
                inst.window.on_close = inst.window_closed_by_user
                inst.geoms = []
                inst.onetime_geoms = []
                inst.transform = Transform()

                glEnable(GL_BLEND)
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glLineWidth(2.0)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                return inst
            except Exception:
                pass
        return SoftwareViewer(width=width, height=height, display=display)

    def close(self):
        try:
            self.window.close()
        except Exception:
            pass

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley),
            scale=(scalex, scaley)
        )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        try:
            glClearColor(1, 1, 1, 1)
            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()
            self.transform.enable()
            for geom in self.geoms:
                geom.render()
            for geom in self.onetime_geoms:
                geom.render()
            self.transform.disable()
            arr = None
            if return_rgb_array:
                buffer = pyglet.image.get_buffer_manager().get_color_buffer()
                image_data = buffer.get_image_data()
                arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
                arr = arr.reshape(buffer.height, buffer.width, 4)
                arr = arr[::-1, :, 0:3]
            self.window.flip()
            self.onetime_geoms = []
            return arr
        except Exception:
            sw = SoftwareViewer(width=self.width, height=self.height)
            sw.geoms = self.geoms
            sw.onetime_geoms = self.onetime_geoms
            res = sw.render(return_rgb_array=return_rgb_array)
            self.onetime_geoms = []
            return res

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom
