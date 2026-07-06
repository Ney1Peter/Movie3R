# modified from Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
from smplx.lbs import vertices2joints, vertices2landmarks
from dust3r.utils.geometry import (
    perspective_projection, 
    resize_camera_intrinsics,
    get_camera_parameters
)
from dust3r.utils.image import pad_image
import roma
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
SMPLX_DIR = os.path.join(src_dir, 'models')
MEAN_PARAMS = os.path.join(src_dir, 'models', 'smpl_mean_params.npz')
SMPLX2SMPL = os.path.join(src_dir, 'models', 'smplx', 'smplx2smpl.pkl')

BODY25_TO_SMPLX_JOINTS = {
    1: JOINT_NAMES.index("neck"),
    2: JOINT_NAMES.index("right_shoulder"),
    3: JOINT_NAMES.index("right_elbow"),
    4: JOINT_NAMES.index("right_wrist"),
    5: JOINT_NAMES.index("left_shoulder"),
    6: JOINT_NAMES.index("left_elbow"),
    7: JOINT_NAMES.index("left_wrist"),
    8: JOINT_NAMES.index("pelvis"),
    9: JOINT_NAMES.index("right_hip"),
    10: JOINT_NAMES.index("right_knee"),
    11: JOINT_NAMES.index("right_ankle"),
    12: JOINT_NAMES.index("left_hip"),
    13: JOINT_NAMES.index("left_knee"),
    14: JOINT_NAMES.index("left_ankle"),
    15: JOINT_NAMES.index("right_eye"),
    16: JOINT_NAMES.index("left_eye"),
    17: JOINT_NAMES.index("right_ear"),
    18: JOINT_NAMES.index("left_ear"),
    19: JOINT_NAMES.index("left_big_toe"),
    20: JOINT_NAMES.index("left_small_toe"),
    21: JOINT_NAMES.index("left_heel"),
    22: JOINT_NAMES.index("right_big_toe"),
    23: JOINT_NAMES.index("right_small_toe"),
    24: JOINT_NAMES.index("right_heel"),
}

class SMPLModel(object):
    def __init__(self, device, model_args={}, eval_args={}):
        self.device = device
        self.person_center = 'head'
        self.fast_gt = os.environ.get("MOVIE3R_FAST_SMPL_GT", "").lower() in {"1", "true", "yes"}
        
        self.patch_size = model_args.get('patch_size', 16)
        self.mhmr_img_res = model_args.get('mhmr_img_res', 896)
        self.bb_patch_size = model_args.get('bb_patch_size', 14)

        # Parametric 3D human models
        if self.fast_gt:
            self.smplx_neutral_11 = None
            self.smplx_neutral_10 = None
        else:
            self.smplx_neutral_11 = smplx.create(
                SMPLX_DIR, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=11).to(self.device)
            self.smplx_neutral_10 = smplx.create(
                SMPLX_DIR, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=10).to(self.device)
        
        # Evaluation
        self.use_fake_K = eval_args.get('use_fake_K', False)
        dataset = eval_args.get('dataset', None)
        if dataset is not None:
            self.smpl = [
                smplx.create(SMPLX_DIR, 'smpl', gender=g).to(self.device) for g in ['neutral', 'male', 'female']]
            self.smpl_faces = {'smpl': self.smpl[0].faces, 'smplx': self.smplx_neutral_11.faces}
            with open(SMPLX2SMPL, 'rb') as f:
                self.smplx2smpl = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(self.device)

            if dataset in ['rich']:
                self.smplx = {
                    g: smplx.create(SMPLX_DIR, 'smplx', gender=g, num_pca_comps=12
                                    ).to(self.device) for g in ['male', 'female']}
            self._setup_dataset_config(dataset)        
        
    def _setup_dataset_config(self, dataset):
        self.j_smpl = self.smpl[0].J_regressor[:24]
        if dataset in ['3dpw']:
            h36m_to_14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9][:14]
            self.j_h36m = torch.Tensor(np.load('src/models/smpl/J_regressor_h36m.npy'))
            self.j_regressor = self.j_h36m[h36m_to_14]
            self.pelvis_idx = [2, 3]
            self.params_type = 'smpl'
        elif dataset in ['bedlam', 'rich']:
            self.j_regressor = self.j_smpl
            self.pelvis_idx = [1, 2]
            self.params_type = 'smplx'
        else:
            self.j_regressor = self.j_smpl
            self.pelvis_idx = [1, 2]
            self.params_type = 'smpl'

    def forward_smpl(self, dataset, smpl_dict, smpl_mask):
        nhv = int(smpl_mask.sum())

        if dataset in ['bedlam']:
            out = self.smplx_neutral_11(
                global_orient=smpl_dict['smplx_root_pose'][smpl_mask].reshape(-1, 3),
                body_pose=smpl_dict['smplx_body_pose'][smpl_mask].reshape(-1, 21*3),
                jaw_pose=smpl_dict['smplx_jaw_pose'][smpl_mask].reshape(-1, 3),
                leye_pose=smpl_dict['smplx_leye_pose'][smpl_mask].reshape(-1, 3),
                reye_pose=smpl_dict['smplx_reye_pose'][smpl_mask].reshape(-1, 3),
                left_hand_pose=smpl_dict['smplx_left_hand_pose'][smpl_mask].reshape(-1, 15*3),
                right_hand_pose=smpl_dict['smplx_right_hand_pose'][smpl_mask].reshape(-1, 15*3),
                betas=smpl_dict['smplx_shape'][smpl_mask].reshape(-1, 11),
                transl=smpl_dict['smplx_transl'][smpl_mask].reshape(-1, 3),
                expression=self.smplx_neutral_11.expression.repeat(nhv, 1),
            )
            verts = out.vertices.reshape(nhv, -1, 3)

        elif dataset in ['3dpw']:
            smpl_params = {
                'global_orient': smpl_dict['smpl_root_pose'][smpl_mask].reshape(-1,3),
                'body_pose': smpl_dict['smpl_body_pose'][smpl_mask].reshape(-1,23*3),
                'betas': smpl_dict['smpl_shape'][smpl_mask].reshape(-1,10),
                'transl': smpl_dict['smpl_transl'][smpl_mask].reshape(-1,3),
                }
            out = self.smpl[1](**smpl_params)
            verts = out.vertices.reshape(nhv, -1, 3)

            # update verts/joints if this is not the right gender
            if int(smpl_dict['smpl_gender_id'].max()) == 2:
                out_female = self.smpl[2](**smpl_params)
                idx = torch.where(smpl_dict['smpl_gender_id'] == 2)[1]
                verts[idx] = out_female.vertices.reshape(nhv, -1, 3)[idx]
                
        elif dataset in ['emdb', 'emdb1', 'emdb2']:
            gender = smpl_dict['smpl_gender_id'].max()
            out = self.smpl[gender](
                global_orient=smpl_dict['smpl_root_pose_w'][smpl_mask].reshape(-1,3),
                body_pose=smpl_dict['smpl_body_pose'][smpl_mask].reshape(-1,23*3),
                betas=smpl_dict['smpl_shape'][smpl_mask].reshape(-1,10),
                transl=smpl_dict['smpl_transl_w'][smpl_mask].reshape(-1,3),
            )
            verts = out.vertices.reshape(nhv, -1, 3) # world space
                
        elif dataset in ['rich']:
            gender = {1: 'male', 2: 'female'}[int(smpl_dict['smplx_gender_id'].max())]
            out = self.smplx[gender](
                global_orient=smpl_dict['smplx_global_orient'][smpl_mask].reshape(-1,3),
                body_pose=smpl_dict['smplx_body_pose'][smpl_mask].reshape(-1,21*3),
                jaw_pose=torch.zeros([nhv, 3]),
                leye_pose=torch.zeros([nhv, 3]),
                reye_pose=torch.zeros([nhv, 3]),
                left_hand_pose=torch.zeros([nhv, 12]),
                right_hand_pose=torch.zeros([nhv, 12]),
                betas=smpl_dict['smplx_betas'][smpl_mask].reshape(-1,10),
                transl=smpl_dict['smplx_transl'][smpl_mask].reshape(-1,3),
                expression=torch.zeros([nhv, 10]),    
            )
            verts = out.vertices.reshape(nhv, -1, 3)

        if self.params_type == 'smplx':
            verts = self.smplx2smpl @ verts
        jts = self.j_regressor @ verts

        if "smplx_world_scale" in smpl_dict:
            world_scale = smpl_dict["smplx_world_scale"][smpl_mask].reshape(-1, 1, 1)
            world_scale = world_scale.to(device=verts.device, dtype=verts.dtype)
            verts = verts * world_scale
            jts = jts * world_scale

        return verts, jts

    def update_smpl_gt_fast(self, views):
        target = {}
        batch_size = views[0]["img"].shape[0]

        smpl_keys = [k for k in views[0].keys() if 'smpl' in k]
        smpl_dict = {
            k: (stacked := torch.stack(
                [view.pop(k) for view in views], dim=0)).view(-1, *stacked.shape[2:])
            for k in smpl_keys
        }
        smpl_mask = smpl_dict['smpl_mask']
        idx_h = torch.where(smpl_mask)
        K = torch.stack([view['camera_intrinsics'] for view in views], dim=0)
        K = K.view(-1, *K.shape[2:])
        nhv = int(smpl_mask.sum())
        if nhv == 0:
            target['has_smpl'] = False
            return target

        imgs = torch.stack([view["img"] for view in views], dim=0)
        imgs = imgs.view(-1, *imgs.shape[2:])
        K_mhmr = resize_camera_intrinsics(K, *imgs.shape[2:], self.mhmr_img_res)
        imgs_mhmr = pad_image(imgs, self.mhmr_img_res)

        human_params_are_world = None
        if "human_params_are_world" in views[0]:
            human_params_are_world = torch.stack(
                [view["human_params_are_world"] for view in views], dim=0
            ).view(-1).bool()

        T_w2c = None
        if "T_w2c" in views[0]:
            T_w2c = torch.stack([view["T_w2c"] for view in views], dim=0)
            T_w2c = T_w2c.view(-1, *T_w2c.shape[2:])

        transl = smpl_dict["smplx_transl"][smpl_mask]
        head = transl.clone()
        pelvis = transl.clone()

        precomputed = None
        if "smplx_has_precomputed_keypoints" in smpl_dict:
            precomputed = smpl_dict["smplx_has_precomputed_keypoints"][smpl_mask].reshape(-1) > 0.5

        if "smplx_head_world" in smpl_dict:
            head_src = smpl_dict["smplx_head_world"][smpl_mask].to(
                device=head.device, dtype=head.dtype
            )
            if precomputed is not None:
                head[precomputed] = head_src[precomputed]
            else:
                head = head_src
        if "smplx_pelvis_world" in smpl_dict:
            pelvis_src = smpl_dict["smplx_pelvis_world"][smpl_mask].to(
                device=pelvis.device, dtype=pelvis.dtype
            )
            if precomputed is not None:
                pelvis[precomputed] = pelvis_src[precomputed]
            else:
                pelvis = pelvis_src

        smpl_world_selected = None
        T_w2c_selected = None
        if human_params_are_world is not None and T_w2c is not None:
            smpl_world_selected = human_params_are_world[idx_h[0]]
            T_w2c_selected = T_w2c[idx_h[0]]

        def world_to_cam(points):
            if (
                smpl_world_selected is None
                or T_w2c_selected is None
                or not smpl_world_selected.any()
            ):
                return points
            out = points.clone()
            T_sel = T_w2c_selected[smpl_world_selected].to(
                device=points.device, dtype=points.dtype
            )
            R = T_sel[:, :3, :3]
            t = T_sel[:, :3, 3]
            if points.ndim == 2:
                out[smpl_world_selected] = torch.einsum(
                    "bij,bj->bi", R, points[smpl_world_selected]
                ) + t
            else:
                out[smpl_world_selected] = torch.einsum(
                    "bij,bnj->bni", R, points[smpl_world_selected]
                ) + t[:, None, :]
            return out

        head = world_to_cam(head)
        pelvis = world_to_cam(pelvis)

        num_joints = len(JOINT_NAMES)
        jts_cam = transl.new_zeros(nhv, num_joints, 3)
        head_idx = JOINT_NAMES.index(self.person_center)
        pelvis_idx = JOINT_NAMES.index("pelvis")
        jts_cam[:, head_idx] = head
        jts_cam[:, pelvis_idx] = pelvis

        if precomputed is not None and "smplx_body25_world" in smpl_dict:
            body25 = smpl_dict["smplx_body25_world"][smpl_mask].to(
                device=jts_cam.device, dtype=jts_cam.dtype
            )
            body25 = world_to_cam(body25)
            body25_mask = smpl_dict.get("smplx_body25_mask", None)
            if body25_mask is not None:
                body25_mask = body25_mask[smpl_mask].to(device=jts_cam.device) > 0.5
            for body25_idx, smplx_idx in BODY25_TO_SMPLX_JOINTS.items():
                valid = precomputed
                if body25_mask is not None:
                    valid = valid & body25_mask[:, body25_idx]
                if valid.any():
                    jts_cam[valid, smplx_idx] = body25[valid, body25_idx]

        target['smpl_transl'] = head
        target['smpl_transl_pelvis'] = pelvis
        target['smpl_j3d'] = jts_cam
        target['smpl_j2d'] = perspective_projection(jts_cam, K[idx_h[0]])

        rot_keys = [
            'smplx_root_pose',
            'smplx_body_pose',
            'smplx_left_hand_pose',
            'smplx_right_hand_pose',
            'smplx_jaw_pose',
        ]
        if all(k in smpl_dict for k in rot_keys):
            target['smpl_rotvec'] = torch.cat([smpl_dict[k] for k in rot_keys], 2)[smpl_mask]
            if (
                smpl_world_selected is not None
                and T_w2c_selected is not None
                and smpl_world_selected.any()
            ):
                root_world_rot = target['smpl_rotvec'][smpl_world_selected, 0]
                root_world_mat = roma.rotvec_to_rotmat(root_world_rot)
                R_w2c = T_w2c_selected[smpl_world_selected, :3, :3].to(
                    device=root_world_mat.device, dtype=root_world_mat.dtype
                )
                root_cam_rot = roma.rotmat_to_rotvec(R_w2c @ root_world_mat)
                target['smpl_rotvec'][smpl_world_selected, 0] = root_cam_rot.to(
                    dtype=target['smpl_rotvec'].dtype
                )
            target['smpl_rotmat'] = roma.rotvec_to_rotmat(target['smpl_rotvec'])
        if 'smplx_shape' in smpl_dict:
            target['smpl_shape'] = smpl_dict['smplx_shape'][smpl_mask]

        true_shapes = torch.stack([view["true_shape"] for view in views], dim=0)
        if len(torch.unique(true_shapes, dim=0)) != 1:
            raise NotImplementedError

        pk = target['smpl_transl'].unsqueeze(1)
        pk_loc = perspective_projection(pk, K[idx_h[0]]).squeeze(1)
        n_patch_16, pk_idx_16 = get_patch_uv(true_shapes[0][0], self.patch_size, pk_loc)
        target['smpl_uv_16'] = pk_idx_16[:, [1, 0]]

        pk_loc_mhmr = perspective_projection(pk, K_mhmr[idx_h[0]]).squeeze(1)
        n_patch_14, pk_idx_14 = get_patch_uv(self.mhmr_img_res, self.bb_patch_size, pk_loc_mhmr)
        smpl_mask_14, visible_humans_14, scores_14 = get_score(n_patch_14, pk_idx_14, smpl_mask.clone())
        target['smpl_uv'] = pk_idx_14[:, [1, 0]]

        _target = {}
        num_view = len(views)
        max_humans = smpl_mask_14.shape[1]
        idx_vis = torch.where(visible_humans_14)[0]

        for k, v in target.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:],
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask_14] = v[idx_vis]
            _target[k] = full_out.chunk(num_view, dim=0)

        _target['smpl_scores'] = scores_14.chunk(num_view, dim=0)
        _target['smpl_mask'] = smpl_mask_14.chunk(num_view, dim=0)
        _target['K_mhmr'] = K_mhmr.chunk(num_view, dim=0)
        _target['img_mhmr'] = imgs_mhmr.chunk(num_view, dim=0)

        if "msk" in views[0]:
            msks = torch.stack([view["msk"] for view in views], dim=0)
            msks = msks.view(-1, *msks.shape[2:])
            msks_mhmr = pad_image(msks, self.mhmr_img_res, pad_value=0.0)
            msks_mhmr = (msks_mhmr > 0.1).float()
            _target['msk_mhmr'] = msks_mhmr.chunk(num_view, dim=0)

        for i, v in enumerate(zip(*_target.values())):
            views[i].update(dict(zip(_target.keys(), v)))

        torch.cuda.empty_cache()

    def update_smpl_gt(self, views):
        if self.fast_gt:
            return self.update_smpl_gt_fast(views)

        target = {}

        batch_size = views[0]["img"].shape[0]

        smpl_keys = [k for k in views[0].keys() if 'smpl' in k]
        smpl_dict = {
            k: (stacked := torch.stack(
                [view.pop(k) for view in views], dim=0)).view(-1, *stacked.shape[2:])
            for k in smpl_keys
        }   # Shape: (num_views * batch_size, 10, ...)
        smpl_mask = smpl_dict['smpl_mask']
        idx_h = torch.where(smpl_mask) # frame_idx, batch_idx, human_idx
        K = torch.stack(
            [view['camera_intrinsics'] for view in views], dim=0
        )
        K = K.view(-1, *K.shape[2:])
        nhv = int(smpl_mask.sum())

        # If no valid SMPL humans in this batch, return empty target
        if nhv == 0:
            target['has_smpl'] = False
            return target

        # Get MHMR input image (high-res, square)
        imgs = torch.stack([view["img"] for view in views], dim=0)
        imgs = imgs.view(-1, *imgs.shape[2:])
        K_mhmr = resize_camera_intrinsics(K, *imgs.shape[2:], self.mhmr_img_res)
        imgs_mhmr = pad_image(imgs, self.mhmr_img_res)

        # SMPLX forward - BEDLAM
        has_smplx_params = 1
        out = self.smplx_neutral_11(
            global_orient=smpl_dict['smplx_root_pose'][smpl_mask].reshape(-1, 3),
            body_pose=smpl_dict['smplx_body_pose'][smpl_mask].reshape(-1, 21*3),
            jaw_pose=smpl_dict['smplx_jaw_pose'][smpl_mask].reshape(-1, 3),
            leye_pose=smpl_dict['smplx_leye_pose'][smpl_mask].reshape(-1, 3),
            reye_pose=smpl_dict['smplx_reye_pose'][smpl_mask].reshape(-1, 3),
            left_hand_pose=smpl_dict['smplx_left_hand_pose'][smpl_mask].reshape(-1, 15*3),
            right_hand_pose=smpl_dict['smplx_right_hand_pose'][smpl_mask].reshape(-1, 15*3),
            betas=smpl_dict['smplx_shape'][smpl_mask].reshape(-1, 11),
            transl=smpl_dict['smplx_transl'][smpl_mask].reshape(-1, 3),
            expression=self.smplx_neutral_11.expression.repeat(nhv, 1),
        )
        verts, jts = out.vertices.reshape(nhv, -1, 3), out.joints.reshape(nhv, -1, 3)

        if "smplx_world_scale" in smpl_dict:
            world_scale = smpl_dict["smplx_world_scale"][smpl_mask].reshape(-1, 1, 1)
            world_scale = world_scale.to(device=verts.device, dtype=verts.dtype)
            verts = verts * world_scale
            jts = jts * world_scale

        human_params_are_world = None
        if "human_params_are_world" in views[0]:
            human_params_are_world = torch.stack(
                [view["human_params_are_world"] for view in views], dim=0
            ).view(-1).bool()

        T_w2c = None
        if "T_w2c" in views[0]:
            T_w2c = torch.stack([view["T_w2c"] for view in views], dim=0)
            T_w2c = T_w2c.view(-1, *T_w2c.shape[2:])

        verts_cam, jts_cam = verts, jts
        smpl_world_selected = None
        T_w2c_selected = None
        if human_params_are_world is not None and T_w2c is not None:
            smpl_world_selected = human_params_are_world[idx_h[0]]
            if smpl_world_selected.any():
                verts_cam = verts.clone()
                jts_cam = jts.clone()
                T_w2c_selected = T_w2c[idx_h[0]]
                T_verts = T_w2c_selected[smpl_world_selected].to(
                    device=verts_cam.device, dtype=verts_cam.dtype
                )
                R = T_verts[:, :3, :3]
                t = T_verts[:, :3, 3]
                verts_cam[smpl_world_selected] = torch.einsum(
                    "bij,bnj->bni", R, verts[smpl_world_selected]
                ) + t[:, None, :]
                T_jts = T_w2c_selected[smpl_world_selected].to(
                    device=jts_cam.device, dtype=jts_cam.dtype
                )
                R = T_jts[:, :3, :3]
                t = T_jts[:, :3, 3]
                jts_cam[smpl_world_selected] = torch.einsum(
                    "bij,bnj->bni", R, jts[smpl_world_selected]
                ) + t[:, None, :]

        if "smplx_has_precomputed_mesh" in smpl_dict and "smplx_mesh_world" in smpl_dict:
            precomputed_mesh = smpl_dict["smplx_has_precomputed_mesh"][smpl_mask].reshape(-1) > 0.5
            if precomputed_mesh.any():
                mesh_world = smpl_dict["smplx_mesh_world"][smpl_mask].to(
                    device=verts_cam.device, dtype=verts_cam.dtype
                )
                mesh_cam = mesh_world
                if (
                    smpl_world_selected is not None
                    and T_w2c_selected is not None
                    and smpl_world_selected.any()
                ):
                    mesh_cam = mesh_world.clone()
                    T_mesh = T_w2c_selected[smpl_world_selected].to(
                        device=verts_cam.device, dtype=verts_cam.dtype
                    )
                    R = T_mesh[:, :3, :3]
                    t = T_mesh[:, :3, 3]
                    mesh_cam[smpl_world_selected] = torch.einsum(
                        "bij,bvj->bvi", R, mesh_world[smpl_world_selected]
                    ) + t[:, None, :]

                verts_cam[precomputed_mesh] = mesh_cam[precomputed_mesh]

                mesh_joints = vertices2joints(self.smplx_neutral_11.J_regressor, mesh_cam)
                mesh_joints = self.smplx_neutral_11.vertex_joint_selector(mesh_cam, mesh_joints)
                lmk_faces_idx = self.smplx_neutral_11.lmk_faces_idx.unsqueeze(0).expand(
                    mesh_cam.shape[0], -1
                ).contiguous()
                lmk_bary_coords = self.smplx_neutral_11.lmk_bary_coords.unsqueeze(0).expand(
                    mesh_cam.shape[0], -1, -1
                ).contiguous()
                mesh_landmarks = vertices2landmarks(
                    mesh_cam,
                    self.smplx_neutral_11.faces_tensor,
                    lmk_faces_idx,
                    lmk_bary_coords,
                )
                mesh_joints = torch.cat([mesh_joints, mesh_landmarks], dim=1)
                mesh_joints = mesh_joints.to(device=jts_cam.device, dtype=jts_cam.dtype)
                if mesh_joints.shape[1] == jts_cam.shape[1]:
                    jts_cam[precomputed_mesh] = mesh_joints[precomputed_mesh]

        if "smplx_has_precomputed_keypoints" in smpl_dict:
            precomputed = smpl_dict["smplx_has_precomputed_keypoints"][smpl_mask].reshape(-1) > 0.5
            if precomputed.any():
                body25 = smpl_dict["smplx_body25_world"][smpl_mask].to(
                    device=jts_cam.device, dtype=jts_cam.dtype
                )
                body25_mask = smpl_dict["smplx_body25_mask"][smpl_mask].to(
                    device=jts_cam.device
                ) > 0.5
                head = smpl_dict["smplx_head_world"][smpl_mask].to(
                    device=jts_cam.device, dtype=jts_cam.dtype
                )
                pelvis = smpl_dict["smplx_pelvis_world"][smpl_mask].to(
                    device=jts_cam.device, dtype=jts_cam.dtype
                )

                body25_cam, head_cam, pelvis_cam = body25, head, pelvis
                if (
                    smpl_world_selected is not None
                    and T_w2c_selected is not None
                    and smpl_world_selected.any()
                ):
                    body25_cam = body25.clone()
                    head_cam = head.clone()
                    pelvis_cam = pelvis.clone()
                    T_pre = T_w2c_selected[smpl_world_selected].to(
                        device=jts_cam.device, dtype=jts_cam.dtype
                    )
                    R = T_pre[:, :3, :3]
                    t = T_pre[:, :3, 3]
                    body25_cam[smpl_world_selected] = torch.einsum(
                        "bij,bkj->bki", R, body25[smpl_world_selected]
                    ) + t[:, None, :]
                    head_cam[smpl_world_selected] = torch.einsum(
                        "bij,bj->bi", R, head[smpl_world_selected]
                    ) + t
                    pelvis_cam[smpl_world_selected] = torch.einsum(
                        "bij,bj->bi", R, pelvis[smpl_world_selected]
                    ) + t

                head_idx = JOINT_NAMES.index(self.person_center)
                pelvis_idx = JOINT_NAMES.index("pelvis")
                jts_cam[precomputed, head_idx] = head_cam[precomputed]
                jts_cam[precomputed, pelvis_idx] = pelvis_cam[precomputed]
                for body25_idx, smplx_idx in BODY25_TO_SMPLX_JOINTS.items():
                    valid = precomputed & body25_mask[:, body25_idx]
                    if valid.any():
                        jts_cam[valid, smplx_idx] = body25_cam[valid, body25_idx]

        j2d = perspective_projection(jts_cam, K[idx_h[0]])
        v2d = perspective_projection(verts_cam, K[idx_h[0]])

        # Translation of the primary keypoint
        root_joint_idx = JOINT_NAMES.index(self.person_center)
        target['smpl_transl'] = jts_cam[:,root_joint_idx] # [nhv,3]
        target['smpl_transl_pelvis'] = jts_cam[:,0] # [nhv,3]

        # Fill in target
        target['smpl_v3d'] = verts_cam
        target['smpl_j3d'] = jts_cam
        target['smpl_j2d'] = j2d
        target['smpl_v2d'] = v2d

        if has_smplx_params:
            target['smpl_rotvec'] = torch.cat([smpl_dict['smplx_root_pose'],
                                        smpl_dict['smplx_body_pose'],
                                        smpl_dict['smplx_left_hand_pose'],
                                        smpl_dict['smplx_right_hand_pose'],
                                        smpl_dict['smplx_jaw_pose']],2)[smpl_mask] # [bs,nhmax]
            if (
                smpl_world_selected is not None
                and T_w2c_selected is not None
                and smpl_world_selected.any()
            ):
                root_world_rot = target['smpl_rotvec'][smpl_world_selected, 0]
                root_world_mat = roma.rotvec_to_rotmat(root_world_rot)
                R_w2c = T_w2c_selected[smpl_world_selected, :3, :3].to(
                    device=root_world_mat.device, dtype=root_world_mat.dtype
                )
                root_cam_rot = roma.rotmat_to_rotvec(
                    R_w2c @ root_world_mat
                )
                target['smpl_rotvec'][smpl_world_selected, 0] = root_cam_rot.to(
                    dtype=target['smpl_rotvec'].dtype
                )
            target['smpl_rotmat'] = roma.rotvec_to_rotmat(target['smpl_rotvec'])
            target['smpl_shape'] = smpl_dict['smplx_shape'][smpl_mask]

        
        true_shapes = torch.stack([view["true_shape"] for view in views], dim=0)
        if len(torch.unique(true_shapes, dim=0)) != 1:
            raise NotImplementedError
        
        # Creating the target heatmap for the primary keypoint
        pk = target['smpl_transl'].unsqueeze(1) # (nhv,3)
        
        # For 512 res (CUT3R, patch_size=16)
        pk_loc = perspective_projection(pk, K[idx_h[0]]).squeeze(1) # original pixel uv coordinates (nhv,2): W, H
        n_patch_16, pk_idx_16 = get_patch_uv(true_shapes[0][0], self.patch_size, pk_loc)
        target['smpl_uv_16'] = pk_idx_16[:, [1, 0]]

        # For 896 res (MHMR, patch_size=14)
        pk_loc_mhmr = perspective_projection(pk, K_mhmr[idx_h[0]]).squeeze(1) # original pixel uv coordinates (nhv,2): W, H
        n_patch_14, pk_idx_14 = get_patch_uv(self.mhmr_img_res, self.bb_patch_size, pk_loc_mhmr)
        smpl_mask_14, visible_humans_14, scores_14 = get_score(n_patch_14, pk_idx_14, smpl_mask.clone())
        target['smpl_uv'] = pk_idx_14[:, [1, 0]]

        # Rebatch and Update with visibility indice
        _target = {}
        num_view = len(views)
        max_humans = smpl_mask_14.shape[1]
        idx_vis = torch.where(visible_humans_14)[0]

        for k, v in target.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask_14] = v[idx_vis] # discard unvisible humans due to olccusion
            _target[k] = full_out.chunk(num_view, dim=0) # .view(num_view, batch_size, *full_out.shape[1:])

        _target['smpl_scores'] = scores_14.chunk(num_view, dim=0)
        _target['smpl_mask'] = smpl_mask_14.chunk(num_view, dim=0)
        _target['K_mhmr'] = K_mhmr.chunk(num_view, dim=0)
        _target['img_mhmr'] = imgs_mhmr.chunk(num_view, dim=0)

        if "msk" in views[0]:
            msks = torch.stack([view["msk"] for view in views], dim=0)
            msks = msks.view(-1, *msks.shape[2:])
            msks_mhmr = pad_image(msks, self.mhmr_img_res, pad_value=0.0)  # bs,288,512->bs,896,896
            msks_mhmr = (msks_mhmr > 0.1).float()
            _target['msk_mhmr'] = msks_mhmr.chunk(num_view, dim=0)

        for i, v in enumerate(zip(*_target.values())):
            views[i].update(dict(zip(_target.keys(), v)))

        torch.cuda.empty_cache()
    
    def update_smpl_gt_eval(self, views, dataset):
        from dust3r.utils.geometry import geotrf

        target = {}
        batch_size = views[0]["img"].shape[0]

        smpl_keys = [k for k in views[0].keys() if 'smpl' in k]
        smpl_dict = {
            k: (stacked := torch.stack(
                [view.pop(k) for view in views], dim=0)).view(-1, *stacked.shape[2:])
            for k in smpl_keys
        }   # Shape: (num_views * batch_size, 10, ...)
        smpl_mask = smpl_dict['smpl_mask']
        idx_h = torch.where(smpl_mask) # frame_idx, batch_idx, human_idx
        K = torch.stack([view['camera_intrinsics'] for view in views], dim=0)
        K = K.view(-1, *K.shape[2:])

        # Get MHMR input image (high-res, square)
        imgs = torch.stack([view["img"] for view in views], dim=0)
        imgs = imgs.view(-1, *imgs.shape[2:])
        K_mhmr = resize_camera_intrinsics(K, *imgs.shape[2:], self.mhmr_img_res)
        imgs_mhmr = pad_image(imgs, self.mhmr_img_res)

        verts, jts = self.forward_smpl(dataset, smpl_dict, smpl_mask)

        if dataset in ['emdb', 'emdb1', 'emdb2', 'rich']:
            target['smpl_v3d_w'] = verts
            target['smpl_j3d_w'] = jts
            T_w2c = torch.stack([view['T_w2c'] for view in views], dim=0)
            T_w2c = T_w2c.view(-1, *T_w2c.shape[2:])
            target['smpl_v3d_c'] = geotrf(T_w2c[idx_h[0]], verts)
            target['smpl_j3d_c'] = geotrf(T_w2c[idx_h[0]], jts)
 
        else:
            target['smpl_v3d_c'] = verts
            target['smpl_j3d_c'] = jts
            T_c2w = torch.stack([view['camera_pose'] for view in views], dim=0)
            T_c2w = T_c2w.view(-1, *T_c2w.shape[2:])
            target['smpl_v3d_w'] = geotrf(T_c2w[idx_h[0]], verts)
            target['smpl_j3d_w'] = geotrf(T_c2w[idx_h[0]], jts)

        target['smpl_j2d'] = perspective_projection(target['smpl_j3d_c'], K[idx_h[0]])
        target['smpl_v2d'] = perspective_projection(target['smpl_v3d_c'], K[idx_h[0]])

        # Rebatch and Update with visibility indice
        _target = {}
        num_view = len(views)
        max_humans = smpl_mask.shape[1]
        for k, v in target.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask] = v # discard unvisible humans due to olccusion
            _target[k] = full_out.chunk(num_view, dim=0) # .view(num_view, batch_size, *full_out.shape[1:])

        if self.use_fake_K:
            K_mhmr = get_camera_parameters(self.mhmr_img_res, device=K.device) # if use pseudo K
            K_mhmr = K_mhmr.expand(K.shape[0], -1, -1)

        _target['smpl_mask'] = smpl_mask.chunk(num_view, dim=0)
        _target['K_mhmr'] = K_mhmr.chunk(num_view, dim=0)
        _target['img_mhmr'] = imgs_mhmr.chunk(num_view, dim=0)

        if "msk" in views[0]:
            msks = torch.stack([view["msk"] for view in views], dim=0)
            msks = msks.view(-1, *msks.shape[2:])
            msks_mhmr = pad_image(msks, self.mhmr_img_res, pad_value=0.0)  # bs,288,512->bs,896,896
            msks_mhmr = (msks_mhmr > 0.1).float()
            _target['msk_mhmr'] = msks_mhmr.chunk(num_view, dim=0)

        for i, v in enumerate(zip(*_target.values())):
            views[i].update(dict(zip(_target.keys(), v)))

        torch.cuda.empty_cache()


def get_patch_uv(imgshape, patch_size, pk_loc):
    n_patch = imgshape // patch_size  # H, W
    pk_idx = (pk_loc // patch_size).int()
    return n_patch, pk_idx

def get_score(n_patch, pk_idx, smpl_mask):
    # Scores & updating valid_humans according to occlusion - wap X and Y for scores only
    idx_h = torch.where(smpl_mask)
    nhv = int(smpl_mask.sum())
    bs = smpl_mask.shape[0]
    device = smpl_mask.device

    if isinstance(n_patch, (int, float)):
        patch_h, patch_w = int(n_patch), int(n_patch)
    else:
        patch_h, patch_w = n_patch[0], n_patch[1]

    scores = torch.zeros((bs, patch_h, patch_w)).to(device)
    visible_humans = torch.ones(nhv).to(device) # by default no occlusion so all visible

    for k in range(nhv):
        i = int(idx_h[0][k]) # index of the image
        j = int(idx_h[1][k]) # index of the human in this image
        _x = pk_idx[k,1] # patch center H
        _y = pk_idx[k,0] # patch center W
        # filter out heads out of cropping bounds
        if _x >= 0 and _x < patch_h and _y >= 0 and _y < patch_w:
            if scores[i,_x,_y] == 1:
                smpl_mask[i,j] = 0
                visible_humans[k] = 0
            else:
                scores[i,_x,_y] = 1
        else:
            smpl_mask[i,j] = 0
            visible_humans[k] = 0
    
    return smpl_mask, visible_humans, scores


import torch.nn as nn
from croco.models.blocks import Mlp_flex

class SMPLDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        target_dim=1,
        mlp_ratio=1,
        num_layers=2,
    ):
        super().__init__()
        self.mlp = Mlp_flex(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=target_dim,
            num_layers=num_layers,
            drop=0,
        )

    def forward(
        self,
        feat,
    ):
        """
        feat: BxC
        """

        pred = self.mlp(feat)
        return pred


def regression_mlp(layers_sizes):
    """
    Return a fully connected network.
    """
    assert len(layers_sizes) >= 2
    in_features = layers_sizes[0]
    layers = []
    for i in range(1, len(layers_sizes)-1):
        out_features = layers_sizes[i]
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, layers_sizes[-1]))
    return torch.nn.Sequential(*layers)

def apply_threshold(det_thresh, _scores):
    """ Apply thresholding to detection scores; if stack_K is used and det_thresh is a list, apply to each channel separately """
    if isinstance(det_thresh, list):
        det_thresh = det_thresh[0]
    idx = torch.where(_scores >= det_thresh)
    return idx

def nms(heat, kernel=3):
    """ easy non maximal supression (as in CenterNet) """

    if kernel not in [2, 4]:
        pad = (kernel - 1) // 2
    else:
        if kernel == 2:
            pad = 1
        else:
            pad = 2

    hmax = nn.functional.max_pool2d( heat, (kernel, kernel), stride=1, padding=pad)

    if hmax.shape[2] > heat.shape[2]:
        hmax = hmax[:, :, :heat.shape[2], :heat.shape[3]]

    keep = (hmax == heat).float()

    return heat * keep
