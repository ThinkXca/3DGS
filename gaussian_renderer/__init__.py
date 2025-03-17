import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
import torch.nn.functional as F

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           return_depth = False, return_normal = False, return_opacity = False):
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad() 
    except:
        pass

   
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,  
        scale_modifier=scaling_modifier,  
        viewmatrix=viewpoint_camera.world_view_transform,  
        projmatrix=viewpoint_camera.full_proj_transform,  
        sh_degree=pc.active_sh_degree,  
        campos=viewpoint_camera.camera_center,  
        prefiltered=False,
        debug=pipe.debug,  
        antialiasing=False 
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  

    means3D = pc.get_xyz  
    means2D = screenspace_points  
    opacity = pc.get_opacity  

   
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier) 
    else:
        scales = pc.get_scaling  
        rotations = pc.get_rotation  

    
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  
        else:
            shs = pc.get_features  
    else:
        colors_precomp = override_color  

   
    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

   
    return_dict =  {"render": rendered_image,  
                    "viewspace_points": screenspace_points,  
                    "visibility_filter" : radii > 0,  
                    "radii": radii}  
    
    if return_depth:
        projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach()  
        projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach()  
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1, keepdim=True) + projvect2  
        means3D_depth = means3D_depth.repeat(1, 3)  
        render_depth, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = means3D_depth,  
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_depth = render_depth.mean(dim=0)  
        return_dict.update({'render_depth': render_depth})  

    
    if return_normal:
        rotations_mat = build_rotation(rotations)  
        scales = pc.get_scaling  
        min_scales = torch.argmin(scales, dim=1)  
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]  

      
        view_dir = means3D - viewpoint_camera.camera_center
        normal = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[..., None]

        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)  
        normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)  

        render_normal, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = normal,  
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_normal = F.normalize(render_normal, dim = 0) 
        return_dict.update({'render_normal': render_normal})  

    
    if return_opacity:
        density = torch.ones_like(means3D) 

        render_opacity, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = density,  
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        return_dict.update({'render_opacity': render_opacity.mean(dim=0)})  

    return return_dict  


def render_with_mask(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, person_mask: torch.Tensor, 
                     scaling_modifier=1.0, override_color=None, 
                     return_depth=False, return_normal=False, return_opacity=False):
  
    render_pkg = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color,
                        return_depth, return_normal, return_opacity)

    
    rendered_image = render_pkg["render"]
    render_opacity = render_pkg["render_opacity"]
    
  
    person_mask = person_mask.to(torch.float32)  
    
   
    person_mask = person_mask.unsqueeze(0).repeat(3, 1, 1)  
    
   
    bg_color = bg_color.unsqueeze(-1).unsqueeze(-1)  
    bg_color = bg_color.repeat(1, rendered_image.shape[1], rendered_image.shape[2])  
    
    
    rendered_image = rendered_image * (1 - person_mask) + bg_color * person_mask
    
    
    render_opacity = render_opacity * (1 - person_mask[0]) 

   
    render_pkg.update({
        "render": rendered_image,
        "render_opacity": render_opacity
    })
    
    return render_pkg





