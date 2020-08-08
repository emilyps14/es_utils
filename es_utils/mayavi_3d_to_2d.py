import matplotlib.pyplot as plt
import numpy as np
from es_utils.mayavi_transforms import get_world_to_view_matrix, \
    get_view_to_display_matrix, apply_transform_to_points
from mayavi import mlab


def make_2d_surface(mlab_fig):
    # Make 2d surface img for future plotting
    # mlab_fig: mlab figure
    # Output:
    #   img: 2d screenshot of the figure
    #   trans_dict: transformations to go from 3d coords to 2d
    # e.g.:
    #   mlab_fig = mlab.figure(size=[800,800])
    #   brain = Brain(subject,hemi,surf,figure=mlab_fig)
    #   img,trans_dict = make_2d_surface(mlab_fig)
    comb_trans_mat = get_world_to_view_matrix(mlab_fig.scene)
    view_to_disp_mat = get_view_to_display_matrix(mlab_fig.scene)
    trans_dict = dict(comb_trans_mat=comb_trans_mat,
                      view_to_disp_mat=view_to_disp_mat)
    img = mlab.screenshot()
    return img,trans_dict

def transform_coordinates(coords, trans_dict):
    # coords: N x 3 coordinates in 3d space
    # trans_dict: output of make_2d_surface
    # Output:
    #   coords_2d: N x 2 coordinates in 2d space
    # e.g.
    #   img,trans_dict = make_2d_surface(mlab_fig)
    #   coords_2d = (coords, trans_dict)
    #   plt.imshow(img)
    #   plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
    N = coords.shape[0]

    # Transform (see mayavi_transforms.py)
    hmgns_world_coords = np.hstack((coords, np.ones((N, 1))))
    view_coords = apply_transform_to_points(hmgns_world_coords,
                                            trans_dict['comb_trans_mat'])
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))
    disp_coords = apply_transform_to_points(norm_view_coords,
                                            trans_dict['view_to_disp_mat'])
    return disp_coords[:,:2]

def scatter_on_2d_surface(coords, img, trans_dict,
                          scatter_kwargs=None, ax=None):
    # Scatter electrodes on 2d surface
    disp_coords = transform_coordinates(coords, trans_dict)

    if scatter_kwargs is None:
        scatter_kwargs = dict()

    if ax is None:
        ax = plt.gca()

    # Plot
    ax.imshow(img)
    ax.scatter(disp_coords[:, 0], disp_coords[:, 1],**scatter_kwargs)
