"""
Helper for converting obj (vt, ft) texture to NMR texture format.


To use NMR, we need to supply a uv_map (F x T x T x 2).
T is a sampling factor on each triangle.
Given this w can sample from an UV image and texture the map.

SMAL already has it's own custom UV map (set via blender).
This is loaded in 'template_w_tex_uv.obj'.
This stores the UV map in the standard ft and vt format of obj, where
    vt: for each vertex, it stores the uv coordinate.
    ft: for each face, this holds the index into vt
i.e.
uv_map_for_verts = mesh.vt[mesh.ft]

This is same as F x T x T x 2 value where T = 1 (only the vertices have color).

So we need to compute the UV coordinates of the samples on each triangle.
We do this by interpolating with barycentric coordinates.

Other small things to note are that obj reads y [1-0],
not [0-1] like in image space.
"""

import numpy as np


def obj2nmr_uvmap(ft, vt, tex_size=6):
    """
    Converts obj uv_map to NMR uv_map (F x T x T x 2),
    where tex_size (T) is the sample rate on each face.
    """
    # This is F x 3 x 2
    uv_map_for_verts = vt[ft]

    # obj's y coordinate is [1-0], but image is [0-1]
    uv_map_for_verts[:, :, 1] = 1 - uv_map_for_verts[:, :, 1]

    # range [0, 1] -> [-1, 1]
    uv_map_for_verts = (2 * uv_map_for_verts) - 1

    alpha = np.arange(tex_size, dtype=np.float) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size - 1)
    import itertools
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])

    # Compute alpha, beta (this is the same order as NMR)
    v2 = uv_map_for_verts[:, 2]
    v0v2 = uv_map_for_verts[:, 0] - uv_map_for_verts[:, 2]
    v1v2 = uv_map_for_verts[:, 1] - uv_map_for_verts[:, 2]
    # Interpolate the vertex uv values: F x 2 x T*2
    uv_map = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)

    # F x T*2 x 2  -> F x T x T x 2
    uv_map = np.transpose(uv_map, (0, 2, 1)).reshape(-1, tex_size, tex_size, 2)

    return uv_map
