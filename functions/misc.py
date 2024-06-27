import math, torch


def identity(length):
    return torch.eye(3).repeat(length, 1, 1)


def polar_from_cart(x, y, z):
    r = math.hypot(x, y, z)
    theta = math.atan2(y, x)
    phi = math.atan2(math.hypot(x, y), z)
    
    return r, theta, phi


def cart_from_polar(r, theta, phi):
    # Double check this
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z
