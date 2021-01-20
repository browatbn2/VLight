import cv2
from csl_common.utils.nn import to_numpy
from csl_common import vis


def overlay_vessels_heatmap(imgs, pred_vessel_hm):
    pred_vessel_hm = to_numpy(pred_vessel_hm)
    disp_X_recon_overlay = [vis.overlay_heatmap(imgs[i], pred_vessel_hm[i, 0], 1.0) for i in
                            range(len(pred_vessel_hm))]
    return disp_X_recon_overlay


def visualize_vessels(images, X_recon, vessel_hm, pred_vessel_hm=None, ds=None, wait=0,
                      horizontal=False, f=1.0, overlay_heatmaps_input=True, overlay_heatmaps_recon=True,
                      scores=None, nimgs=5):

    nimgs = min(nimgs, len(images))
    images = images[:nimgs]
    rows = []

    input_images = vis.to_disp_images(images[:nimgs], denorm=True)
    disp_images = vis.to_disp_images(images[:nimgs], denorm=True)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]
    rows.append(vis.make_grid(disp_images, nCols=nimgs, normalize=False))

    recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=True)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]

    if vessel_hm is not None and overlay_heatmaps_input:
        vessel_hm = to_numpy(vessel_hm[:nimgs])
        disp_images = [vis.overlay_heatmap(disp_images[i], vessel_hm[i,0], 0.5) for i in range(len(vessel_hm))]

    rows.append(vis.make_grid(disp_images, nCols=nimgs, normalize=False))

    if pred_vessel_hm is not None and overlay_heatmaps_recon:
        pred_vessel_hm = to_numpy(pred_vessel_hm[:nimgs])
        disp_X_recon_overlay = [vis.overlay_heatmap(disp_X_recon[i], pred_vessel_hm[i,0], 1.0) for i in range(len(pred_vessel_hm))]
        if scores is not None:
            disp_X_recon_overlay = vis.add_error_to_images(disp_X_recon_overlay, scores, loc='tr', format_string='{:.3f}')
        rows.append(vis.make_grid(disp_X_recon_overlay, nCols=nimgs))

    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=4)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)

    wnd_title = 'Predicted vessels '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)