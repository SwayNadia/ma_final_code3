import torch
from collections import OrderedDict
from model_and_model_component.render_ray_LinGaoyuan import render_rays


def render_single_image(
    args,
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    render_stride=1,
    featmaps=None,
    ret_alpha=False,
    single_net=False,
    sky_style_code=None,
    sky_style_model=None,
    sky_model=None,
    feature_volume = None,
    mode='val',
    use_updated_prior_depth=False,
    train_depth_prior=None,
    data_mode=None,
):
    """Render a single image and return both 2D and 3D data"""
    """
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param ret_alpha: if True, will return learned 'density' values inferred from the attention maps
    :param single_net: if True, will use single network, can be cued with both coarse and fine points
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    """

    all_ret = OrderedDict([("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())])

    N_rays = ray_batch["ray_o"].shape[0]  # 360000 in train, 1440000 in eval

    # Create dictionary to store 3D data
    three_d_data = {
        "ray_origins": [],
        "ray_directions": [],
        "sample_points": [],
        "features": [],
        "colors": [],
        "densities": [],
        "weights": [],
        "z_values": []
    }

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ["camera", "depth_range", "src_rgbs", "src_cameras"]:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i : i + chunk_size]
            else:
                chunk[k] = None

        '''
        LinGaoyuan_operation_20240918: In val process, if the data is from training dataset and use_updated_prior_depth is True,
        the updated prior depth will be used
        '''
        if train_depth_prior is not None and use_updated_prior_depth is True:
            train_depth_prior_chunk = train_depth_prior[i : i + chunk_size]
        else:
            train_depth_prior_chunk = None

        # Store 3D data
        three_d_data["ray_origins"].append(chunk["ray_o"])
        three_d_data["ray_directions"].append(chunk["ray_d"])

        ret, _ = render_rays(
            args,
            chunk,
            model,
            featmaps,
            projector=projector,
            N_samples=N_samples,
            inv_uniform=inv_uniform,
            N_importance=N_importance,
            det=det,
            white_bkgd=white_bkgd,
            ret_alpha=ret_alpha,
            single_net=single_net,
            sky_style_code=sky_style_code,
            sky_model=sky_model,
            mode=mode,
            feature_volume=feature_volume,
            use_updated_prior_depth=use_updated_prior_depth,
            train_depth_prior=train_depth_prior_chunk,
            data_mode=data_mode,
        )

        # Store additional 3D data from render_rays output
        if "sample_points" in ret:
            three_d_data["sample_points"].append(ret["sample_points"])
        if "features" in ret:
            three_d_data["features"].append(ret["features"])
        if "colors" in ret:
            three_d_data["colors"].append(ret["colors"])
        if "densities" in ret:
            three_d_data["densities"].append(ret["densities"])
        if "weights" in ret["outputs_coarse"]:
            three_d_data["weights"].append(ret["outputs_coarse"]["weights"])
        if "z_vals" in ret:
            three_d_data["z_values"].append(ret["z_vals"])

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret["outputs_coarse"]:
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k] = []

            if ret["outputs_fine"] is None:
                all_ret["outputs_fine"] = None
            else:
                for k in ret["outputs_fine"]:
                    if ret["outputs_fine"][k] is not None:
                        all_ret["outputs_fine"][k] = []

        for k in ret["outputs_coarse"]:
            if ret["outputs_coarse"][k] is not None:
                all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k].cpu())

        if ret["outputs_fine"] is not None:
            for k in ret["outputs_fine"]:
                if ret["outputs_fine"][k] is not None:
                    all_ret["outputs_fine"][k].append(ret["outputs_fine"][k].cpu())

    # Concatenate all stored 3D data
    for key in three_d_data:
        if three_d_data[key]:
            three_d_data[key] = torch.cat(three_d_data[key], dim=0)

    # Add 3D data to return value
    all_ret["three_d_data"] = three_d_data

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret["outputs_coarse"]:
        if k == "random_sigma":
            continue

        'LinGaoyuan_operation_20240906: do not cat depth_cov in validate phase'
        if k == "depth_cov":
            continue

        tmp = torch.cat(all_ret["outputs_coarse"][k], dim=0).reshape(
            (rgb_strided.shape[0], rgb_strided.shape[1], -1)
        )
        all_ret["outputs_coarse"][k] = tmp.squeeze()

    # TODO: if invalid: replace with white
    # all_ret["outputs_coarse"]["rgb"][all_ret["outputs_coarse"]["mask"] == 0] = 1.0
    if all_ret["outputs_fine"] is not None:
        for k in all_ret["outputs_fine"]:
            if k == "random_sigma":
                continue

            'LinGaoyuan_operation_20240906: do not cat depth_cov in validate phase'
            if k == "depth_cov":
                continue

            tmp = torch.cat(all_ret["outputs_fine"][k], dim=0).reshape(
                (rgb_strided.shape[0], rgb_strided.shape[1], -1)
            )

            all_ret["outputs_fine"][k] = tmp.squeeze()

    return all_ret

def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
    sky_style_code=None,
    sky_model=None,
    data_mode=None,
):
    """Log a view to the viewer and save to disk"""
    # ... existing code ...

    # Initialize viewer if not already initialized
    if not hasattr(args, 'viewer'):
        from visualization import Viewer
        args.viewer = Viewer(port=8080)
        args.viewer.viser_server.on_client_connect(args.viewer.handle_new_client)
        args.viewer.viser_server.on_client_disconnect(args.viewer.handle_disconnect)

    ret = render_single_image(
        args,
        ray_sampler=ray_sampler,
        ray_batch=ray_batch,
        model=model,
        projector=projector,
        chunk_size=args.chunk_size,
        N_samples=args.N_samples,
        inv_uniform=args.inv_uniform,
        det=True,
        N_importance=args.N_importance,
        white_bkgd=args.white_bkgd,
        render_stride=render_stride,
        featmaps=featmaps,
        ret_alpha=ret_alpha,
        single_net=single_net,
        sky_style_code=sky_style_code,
        sky_model=sky_model,
        feature_volume=feature_volume,
        data_mode=data_mode,
        use_updated_prior_depth=True
    )

    # Store 3D data for visualization
    args.viewer.current_3d_data = ret["three_d_data"]
    
    # Visualize 3D data
    args.viewer.visualize_3d_data(ret["three_d_data"])

    # ... rest of the existing code ...
