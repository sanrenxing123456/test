import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.misc import NestedTensor, nested_tensor_from_tensor_list
from torchvision.ops import roi_align
from torchvision.transforms import Compose, ToTensor,Normalize
from .transformer import TransformerDecoderLayer, TransformerDecoder
from src.util import box_ops
import numpy as np
import matplotlib.pyplot as plt
from src.util.misc import accuracy, is_dist_avail_and_initialized, get_world_size
from src.models.stip_utils import check_annotation
import time
from mmpose.core.post_processing import get_warp_matrix
import cv2
import matplotlib.pyplot as plt
import PIL.Image as im
import json
import matplotlib.patches as patches
from PIL import Image
from src.models.Graph import Grapher,Stem
class STIP(nn.Module):
    def __init__(self, args, detr, detr_matcher,vit):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("#####33333#333",device)
        self.args = args
        self.detr_matcher = detr_matcher
        # * Instance Transformer ---------------
        self.detr = detr
        self.vitposemodel=vit
        self.graph=Grapher
        self.stem = Stem(input_dim=3, output_dim=256).to(device)
        self.graph=Grapher(in_channels=256,out_channels=512)
        self.graph_fc=nn.Linear(21*256,256)
        if not args.train_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        # --------------------------------------

        # relation feature map
        if self.args.relation_feature_map_from == 'backbone':
            if args.use_high_resolution_relation_feature_map:
                relation_feature_map_dim = 1024
            else:
                relation_feature_map_dim = 2048
        elif self.args.relation_feature_map_from == 'detr_encoder':
            relation_feature_map_dim = self.args.hidden_dim

        # relation proposal
        rel_rep_dim = 1024
        self.coarse_relation_feature_extractor = RelationFeatureExtractor(args, in_channels=relation_feature_map_dim, out_dim=rel_rep_dim)
        # self.union_box_feature_extractor = RelationFeatureExtractor(args, in_channels=relation_feature_map_dim, out_dim=rel_rep_dim)
        self.relation_proposal_mlp = nn.Sequential(
            make_fc(rel_rep_dim, rel_rep_dim // 2), nn.ReLU(),
            make_fc(rel_rep_dim // 2, 1)
        )

        # relation classification
        self.rel_query_pre_proj = make_fc(rel_rep_dim, self.args.hidden_dim)
        if self.args.no_interaction_decoder:
            self.args.hoi_aux_loss = False
        else:
            self.memory_input_proj = nn.Conv2d(relation_feature_map_dim, self.args.hidden_dim, kernel_size=1)
            if args.use_memory_layout_encoding:
                self.layout_embeddings = nn.Embedding(6, self.args.hidden_dim) # 0-pad, 1-image, 2-union, 3-subj, 4-obj, 5-intersection
                self.layout_content_aware_mapping = nn.Sequential(
                    make_fc(self.args.hidden_dim * 2, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim)
                )
            if self.args.use_relation_dependency_encoding:
                self.relation_dependency_embeddings = nn.Embedding(6, self.args.hidden_dim)
                self.relation_dependency_content_aware_mapping = nn.Sequential(
                    make_fc(self.args.hidden_dim * 2, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim)
                )
            if self.args.use_query_fourier_encoding:
                self.fourier_feature_embedding = make_fc(1, self.args.hidden_dim//2) # group=8, group_dim=1
                self.fourier_mlp = nn.Sequential(
                    make_fc(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim // 8)
                )

            decoder_layer = TransformerDecoderLayer(d_model=self.args.hidden_dim, nhead=self.args.hoi_nheads)
            decoder_norm = nn.LayerNorm(self.args.hidden_dim)
            self.interaction_decoder = TransformerDecoder(decoder_layer, self.args.hoi_dec_layers, decoder_norm, return_intermediate=True)
        self.action_embed = nn.Linear(self.args.hidden_dim, self.args.num_actions)

    def forward(self, samples: NestedTensor, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        start_time = time.time()
        samples_tensor=self.recover_from_nested_tensor(samples)#chw
        # samples_tensor= [img.permute(1, 2, 0) for img in samples_tensor]#hwc,list
        # self.visualize_batch_tensor_images(samples_tensor)
        samples_tensor=self.denormalize_to_hwc_tensors(samples_tensor)#hwc
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        # ----------------------------------------------

        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        hs, detr_encoder_outs = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1])
        inst_repr = hs[-1]
        num_nodes = inst_repr.shape[1]

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        # -----------------------------------------------

        det2gt_indices = None
        if self.training:
            detr_outs = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            det2gt_indices = self.detr_matcher(detr_outs, targets)
            gt_rel_pairs = []
            for (ds, gs), t in zip(det2gt_indices, targets):
                gt2det_map = torch.zeros(len(gs)).to(device=ds.device, dtype=ds.dtype)
                gt2det_map[gs] = ds
                gt_rels = gt2det_map[t['relation_map'].sum(-1).nonzero(as_tuple=False)]
                perm = torch.randperm(len(gt_rels))

                gt_rel_pairs.append(gt_rels[perm])
                # if len(gt_rel_pairs[-1]) > self.args.num_hoi_queries:
                #     print(f"imageid={t['image_id']}, gt_relation_count={len(gt_rel_pairs[-1])}")
                #     # check_annotation(samples, targets, rel_num=20, idx=0)

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        pred_rel_exists, pred_rel_pairs, pred_actions = [], [], []
        memory_input, memory_input_mask = features[0].decompose()
        memory_pos = pos[0]
        if self.args.relation_feature_map_from == 'backbone':
            relation_feature_map = features[0]
        elif self.args.relation_feature_map_from == 'detr_encoder':
            relation_feature_map = NestedTensor(detr_encoder_outs, memory_input_mask)
            memory_input = detr_encoder_outs

        if not self.args.no_interaction_decoder:
            memory_input = self.memory_input_proj(memory_input)

        for imgid in range(bs):
            # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
            probs = outputs_class[-1, imgid].softmax(-1)
            inst_scores, inst_labels = probs[:, :-1].max(-1)
            human_instance_ids = torch.logical_and(inst_scores>0.5, inst_labels==1).nonzero(as_tuple=False)
            bg_instance_ids = (probs[:, -1] > 1)
            if self.args.apply_nms_on_detr and not self.training:
                suppress_ids = self.apply_nms(inst_scores, inst_labels, outputs_coord[-1, imgid])
                bg_instance_ids[suppress_ids] = True


            ori_h =  samples_tensor[imgid].shape[0]
            ori_w =  samples_tensor[imgid].shape[1]
            # cx, cy, w,h
            human_box_dict_sample = {
                idx.item(): (
                        outputs_coord[-1][imgid][idx.item()] * torch.tensor([ori_w, ori_h, ori_w, ori_h],
                                                                                  device=outputs_coord[-1].device)
                ).detach().cpu().numpy()
                for idx in human_instance_ids.view(-1)
            }

            human_joint_dict, human_joint_score = self.get_human_joint(samples_tensor[imgid], human_box_dict_sample)
            device=outputs_coord[-1].device

            # self.draw_human_joints(samples_tensor[imgid], human_joint_dict)











            rel_mat = torch.zeros((num_nodes, num_nodes))
            rel_mat[human_instance_ids, ~bg_instance_ids] = 1 # subj is human, obj is not background
            if self.args.dataset_file != 'vcoco': rel_mat.fill_diagonal_(0)
            if self.args.adaptive_relation_query_num:
                if len(rel_mat.nonzero(as_tuple=False)) == 0: rel_mat[0,1] = 1
            else: # ensure enough queries
                if len(rel_mat.nonzero(as_tuple=False)) < self.args.num_hoi_queries:
                    tmp_id = np.random.choice(human_instance_ids.squeeze(1).tolist()) if len(human_instance_ids) > 0 else 0
                    rel_mat[tmp_id] = 1
            if self.training:
                rel_mat[gt_rel_pairs[imgid][:,:1], ~bg_instance_ids] = 1
                rel_mat[gt_rel_pairs[imgid][:,0], gt_rel_pairs[imgid][:, 1]] = 0
                rel_pairs = rel_mat.nonzero(as_tuple=False) # neg pairs

                if self.args.use_hard_mining_for_relation_discovery:
                    # hard negative sampling
                    all_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs], dim=0)
                    gt_pair_count = len(gt_rel_pairs[imgid])
                    all_rel_reps = self.coarse_relation_feature_extractor(all_pairs, relation_feature_map, outputs_coord[-1, imgid].detach(), inst_repr[imgid], idx=imgid,human_joint_dict=human_joint_dict,human_joint_score=human_joint_score, obj_label_logits=outputs_class[-1, imgid])
                    p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps)

                    gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                    _, sort_rel_inds = p_relation_exist_logits[gt_pair_count:].squeeze(1).sort(descending=True)
                    # _, sort_rel_inds = torch.cat([inst_scores[all_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[gt_pair_count:].sort(descending=True)
                    sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds+gt_pair_count])[:self.args.num_hoi_queries]

                    sampled_rel_pairs = all_pairs[sampled_rel_inds]
                    sampled_rel_reps = all_rel_reps[sampled_rel_inds]
                    sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]
                else:
                    # random sampling
                    sampled_neg_inds = torch.randperm(len(rel_pairs))
                    sampled_rel_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs[sampled_neg_inds]], dim=0)[:self.args.num_hoi_queries]
                    sampled_rel_reps = self.coarse_relation_feature_extractor(sampled_rel_pairs, relation_feature_map, outputs_coord[-1, imgid].detach(), inst_repr[imgid], obj_label_logits=outputs_class[-1, imgid], idx=imgid)
                    sampled_rel_pred_exists = self.relation_proposal_mlp(sampled_rel_reps).squeeze(1)
            else:
                rel_pairs = rel_mat.nonzero(as_tuple=False)
                rel_reps = self.coarse_relation_feature_extractor(rel_pairs, relation_feature_map, outputs_coord[-1, imgid].detach(), inst_repr[imgid], idx=imgid,human_joint_dict=human_joint_dict,human_joint_score=human_joint_score, obj_label_logits=outputs_class[-1, imgid])
                p_relation_exist_logits = self.relation_proposal_mlp(rel_reps)

                _, sort_rel_inds = p_relation_exist_logits.squeeze(1).sort(descending=True)
                # _, sort_rel_inds = torch.cat([inst_scores[rel_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1).sort(descending=True)
                sampled_rel_inds = sort_rel_inds[:self.args.num_hoi_queries]

                sampled_rel_pairs = rel_pairs[sampled_rel_inds]
                sampled_rel_reps = rel_reps[sampled_rel_inds]
                sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            query_reps = self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1)
            if self.args.no_interaction_decoder:
                outs = query_reps.unsqueeze(0)
            else:
                query_pos_encoding, relation_dependency_encodings, layout_encodings, memory_union_mask, tgt_mask = None, None, None, None, None
                subj_mask, obj_mask, union_mask, _ = self.generate_layout_masks(sampled_rel_pairs, memory_input_mask, outputs_coord[-1, imgid], idx=imgid)
                if self.args.use_relation_tgt_mask:
                    tgt_mask = (torch.diag(sampled_rel_pred_exists) != 0)
                    attend_ids = sampled_rel_pred_exists.sort(descending=True)[1][:self.args.use_relation_tgt_mask_attend_topk]
                    tgt_mask[:, attend_ids] = True
                    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
                if self.args.use_query_fourier_encoding:
                    query_coords = self.fourier_feature_embedding(outputs_coord[-1, imgid][sampled_rel_pairs].view(len(sampled_rel_pairs), 8, 1)) / np.sqrt(self.args.hidden_dim/2)
                    query_pos_encoding = self.fourier_mlp(torch.cat([torch.cos(query_coords), torch.sin(query_coords)], dim=-1)).view(len(sampled_rel_pairs), -1).unsqueeze(1)
                if self.args.use_relation_dependency_encoding:
                    dependency_map = torch.zeros((len(sampled_rel_pairs), len(sampled_rel_pairs))).to(sampled_rel_reps.device).long() # independent: 0
                    dependency_map[sampled_rel_pairs[:, 0].unsqueeze(1) == sampled_rel_pairs[:, 0].unsqueeze(0)] = 1 # same_subj: 1
                    dependency_map[sampled_rel_pairs[:, 1].unsqueeze(1) == sampled_rel_pairs[:, 1].unsqueeze(0)] = 2 # same_obj: 2
                    dependency_map[sampled_rel_pairs[:, 0].unsqueeze(1) == sampled_rel_pairs[:, 1].unsqueeze(0)] = 3 # subj=obj: 3
                    dependency_map[sampled_rel_pairs[:, 1].unsqueeze(1) == sampled_rel_pairs[:, 0].unsqueeze(0)] = 4 # obj=subj: 4
                    dependency_map.fill_diagonal_(5) # self: 5
                    relation_dependency_encodings = self.relation_dependency_embeddings(dependency_map)
                    relation_dependency_encodings = self.relation_dependency_content_aware_mapping(
                        torch.cat([query_reps.permute(1,0,2).expand(*relation_dependency_encodings.shape), relation_dependency_encodings], dim=-1)
                    ).unsqueeze(2) # (#query, #query, batch size, dim)
                if self.args.use_memory_union_mask:
                    memory_union_mask = union_mask.flatten(1)
                if self.args.use_memory_layout_encoding:
                    layout_map = (~union_mask).long() + (~memory_input_mask[imgid:imgid+1]).long() + (~subj_mask).long() + (~obj_mask).long()*2
                    # plt.imshow(role_map[0].cpu().numpy(), cmap=plt.cm.hot_r); plt.colorbar(); plt.show()
                    layout_encodings = self.layout_embeddings(layout_map)
                    layout_encodings = self.layout_content_aware_mapping(
                        torch.cat([memory_input[imgid:imgid+1].permute(0,2,3,1).expand(*layout_encodings.shape), layout_encodings], dim=-1)
                    ).flatten(start_dim=1, end_dim=2).unsqueeze(2) # (#query, #memory, batch size, dim)
                if self.args.use_graph:
                    obj_instance_ids=torch.unique(sampled_rel_pairs[:,1])
                    # cx, cy, w,h
                    obj_box_dict_samples = {
                        idx.item(): outputs_coord[-1][imgid][idx.item()].detach().cpu().numpy()
                        for idx in obj_instance_ids.view(-1)
                    }
                    obj_box_dict=self.get_obj_box(obj_box_dict_samples,ori_h,ori_w)
                    # self.draw_human_joints(samples_tensor[imgid], obj_box_dict_sample)
                    human_patches = self.graph_extract_patches(samples_tensor[imgid], human_joint_dict)
                    obj_patches = self.graph_extract_patches(samples_tensor[imgid],obj_box_dict)
                    for k in human_patches:
                        human_patches[k] = human_patches[k].to(device)
                    for k in obj_patches:
                        obj_patches[k] = obj_patches[k].to(device)
                    combined_patches=self.combine_patches(human_patches, obj_patches, sampled_rel_pairs,device)#(32, 21, 3, 32, 32) T,P,C,H,W

                    patch_embedding=self.stem(combined_patches.unsqueeze(0))#1,32,21,256
                    graph_encodings = self.graph(patch_embedding.permute(0,1,3,2).unsqueeze(-1))
                    graph_encodings=graph_encodings.squeeze().permute(0,2,1)
                    temp_g=graph_encodings.contiguous().view(32, -1)
                    graph_encodings=self.graph_fc(temp_g).unsqueeze(1)
                    query_reps += graph_encodings
                else:
                    print("not use graph")
                    graph_encodings=None
                # query_reps+=pose_feature

                outs = self.interaction_decoder(tgt=query_reps,
                                                tgt_mask=tgt_mask,
                                                query_pos=query_pos_encoding,
                                                query_structure_encoding=relation_dependency_encodings, # inter-ineraction semantic structure
                                                memory=memory_input[imgid:imgid+1].flatten(2).permute(2,0,1),
                                                memory_key_padding_mask=memory_input_mask[imgid:imgid+1].flatten(1),
                                                memory_mask=memory_union_mask,
                                                pos=memory_pos[imgid:imgid+1].flatten(2).permute(2, 0, 1),
                                                memory_role_embedding=layout_encodings,graph_embedding=None) #  intra-ineraction spatial structure
            action_logits = self.action_embed(outs)

            pred_rel_pairs.append(sampled_rel_pairs)
            pred_actions.append(action_logits)
            pred_rel_exists.append(sampled_rel_pred_exists)




        hoi_recognition_time = time.time() - start_time
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_rel_pairs": pred_rel_pairs,
            "pred_actions": [p[-1].squeeze(1) for p in pred_actions],
            "pred_action_exists": pred_rel_exists,
            "det2gt_indices": det2gt_indices,
            "hoi_recognition_time": hoi_recognition_time,
        }
        if self.args.hoi_aux_loss: out['hoi_aux_outputs'] = self._set_hoi_aux_loss(pred_actions)
        if self.args.train_detr and self.args.aux_loss: out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    def combine_patches(self,human_patches, obj_patches, sampled_rel_pairs,device):
        combined_list = []
        for pair in sampled_rel_pairs:
            human_id, obj_id = pair[0].item(), pair[1].item()

            # 取出对应的patches
            human_patch = human_patches.get(human_id)  # shape: (17, 32, 32, 3)
            obj_patch = obj_patches.get(obj_id)  # shape: (4, 32, 32, 3)

            # 如果对应的id不存在，跳过或用全零代替
            if human_patch is None:
                human_patch = torch.zeros(17, 32, 32, 3, dtype=torch.float32, device=device)
            if obj_patch is None:
                obj_patch = torch.zeros(4, 32, 32, 3, dtype=torch.float32, device=device)

            # 拼接 along 第0维度 (patch数量维度)
            combined_patch = torch.cat([human_patch, obj_patch], dim=0)  # shape: (21, 32, 32, 3)
            combined_patch=combined_patch.permute(0,3,1,2)# shape: (21, 3,32, 32)
            combined_list.append(combined_patch)

        # 拼接所有32对，形成 (32, 21, 3, 32, 32) T,P,C,H,W
        combined_tensor = torch.stack(combined_list, dim=0)
        return combined_tensor
    def get_obj_box(self,  outputs_coord_dict, ori_h, ori_w):
        """
        参数:
            img_tensor: Tensor [3, H, W], 图像张量，值范围为 [0, 1]
            outputs_coord_dict: {idx: [cx, cy, w, h]} 的归一化坐标字典
            ori_h, ori_w: 原图高和宽
        返回:
            obj_box_dict: {idx: ndarray [4,2]}，每个框的四个角点坐标
        """
        obj_box_dict = {}

        for idx, box in outputs_coord_dict.items():
            cx, cy, w, h = box

            # 归一化坐标转为像素坐标
            cx *= ori_w
            cy *= ori_h
            w *= ori_w
            h *= ori_h

            # 四个角点（顺时针）：左上、左下、右下、右上
            top_left     = [cx - w / 2, cy - h / 2]
            bottom_left  = [cx - w / 2, cy + h / 2]
            bottom_right = [cx + w / 2, cy + h / 2]
            top_right    = [cx + w / 2, cy - h / 2]

            corners = np.array([top_left, bottom_left, bottom_right, top_right])
            corners[:, 0] /= ori_w  # 归一化 x
            corners[:, 1] /= ori_h  # 归一化 y
            obj_box_dict[idx] = corners

        # # --- 可视化 ---
        # img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        #
        # fig, ax = plt.subplots()
        # ax.imshow(img_np)
        #
        # for idx, corners in obj_box_dict.items():
        #     rect = patches.Polygon(corners, closed=True, edgecolor='red', linewidth=2, fill=False)
        #     ax.add_patch(rect)
        #     ax.text(corners[0][0], corners[0][1] - 5, f"ID: {idx}", color='yellow', fontsize=8)
        #
        # plt.axis('off')
        # plt.show()

        return obj_box_dict
    def gen_gaussian_kernel(self,kernel_size, sigma):
        """生成 2D 高斯核，返回 shape: (kernel_size, kernel_size)"""
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / np.sum(kernel)
        return kernel

    def graph_extract_patches(self, img_tensor, kpts_dict, kernel_size=128, kernel_sigma=0.3,
                                         scale=1 / 4):
        """
        输入:
            img_tensor: [H, W, 3]，Tensor，像素值范围 [0,1]，图像
            kpts_dict:  {id1: (17, 2), id2: (17, 2), ...}，关键点归一化坐标
        输出:
            patches_dict: {id1: [17, h, w, 3], id2: ...}
        """
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)

        H, W, C = img_tensor.shape
        device = img_tensor.device

        # Pad 图像
        pad = kernel_size
        img_pad = F.pad(img_tensor.permute(2, 0, 1).unsqueeze(0), (pad, pad, pad, pad), mode='constant', value=0)
        img_pad = img_pad.squeeze(0).permute(1, 2, 0)  # [H+2p, W+2p, 3]

        # 高斯核
        kernel_np = self.gen_gaussian_kernel(kernel_size, kernel_size * kernel_sigma)
        kernel = torch.tensor(kernel_np, dtype=torch.float32, device=device).unsqueeze(-1).repeat(1, 1, 3)

        patches_dict = {}

        for pid, kpts in kpts_dict.items():
            if isinstance(kpts, np.ndarray):
                kpts = torch.tensor(kpts, dtype=torch.float32, device=device)

            # 将归一化坐标转换为像素坐标
            kpts_pixel = kpts.clone()
            kpts_pixel[:, 0] *= W
            kpts_pixel[:, 1] *= H

            person_patches = []

            for i in range(kpts_pixel.shape[0]):
                x, y = kpts_pixel[i]
                x += pad
                y += pad

                x1, y1 = int(x - kernel_size // 2), int(y - kernel_size // 2)
                x2, y2 = x1 + kernel_size, y1 + kernel_size

                if x1 < 0 or y1 < 0 or x2 > img_pad.shape[1] or y2 > img_pad.shape[0]:
                    patch = torch.zeros(kernel_size, kernel_size, 3, device=device)
                else:
                    patch = img_pad[y1:y2, x1:x2, :] * kernel

                patch = patch.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                patch = F.interpolate(patch, scale_factor=scale, mode='bilinear', align_corners=False)
                patch = patch.squeeze(0).permute(1, 2, 0)  # [h, w, 3]
                person_patches.append(patch)

            patches_dict[pid] = torch.stack(person_patches, dim=0)  # [17, h, w, 3]

        return patches_dict



        def get_body_part_masks(human_joint_dict, H, W, device):
            """
            Generate 5-part human layout masks from multiple persons' keypoints.
            Returns a tensor of shape (5, H, W) where each slice is a binary mask for one body part.
            """
            part_masks = torch.zeros((5, H, W), device=device)  # 0~4部分：头、躯干、左上肢、右上肢、下肢

            for person_id, joints in human_joint_dict.items():
                kp = torch.tensor(joints, dtype=torch.float32, device=device)  # shape (17, 2)

                def draw_line(mask, p1, p2, width=5):
                    if not (torch.all(p1 >= 0) and torch.all(p2 >= 0)): return
                    p1 = p1.int()
                    p2 = p2.int()
                    steps = 20
                    ys = torch.linspace(p1[1], p2[1], steps).round().long()
                    xs = torch.linspace(p1[0], p2[0], steps).round().long()
                    for y, x in zip(ys, xs):
                        if 0 <= y < H and 0 <= x < W:
                            y1, y2 = max(0, y - width), min(H, y + width)
                            x1, x2 = max(0, x - width), min(W, x + width)
                            mask[y1:y2, x1:x2] = 1

                # 0: Head
                for a, b in [(0, 1), (0, 2), (1, 3), (2, 4)]:
                    draw_line(part_masks[0], kp[a], kp[b])
                # 1: Torso
                for a, b in [(5, 6), (5, 11), (6, 12), (11, 12)]:
                    draw_line(part_masks[1], kp[a], kp[b])
                # 2: Left arm
                for a, b in [(5, 7), (7, 9)]:
                    draw_line(part_masks[2], kp[a], kp[b])
                # 3: Right arm
                for a, b in [(6, 8), (8, 10)]:
                    draw_line(part_masks[3], kp[a], kp[b])
                # 4: Legs
                for a, b in [(11, 13), (13, 15), (12, 14), (14, 16)]:
                    draw_line(part_masks[4], kp[a], kp[b])

            return part_masks.clamp(max=1)  # shape (5, H, W)
    def recover_from_nested_tensor(self,nested_tensor):
        tensors, mask = nested_tensor.decompose()
        recovered = []
        for img, m in zip(tensors, mask):
            h, w = (~m).sum(dim=0).max().item(), (~m).sum(dim=1).max().item()
            recovered.append(img[:, :h, :w])
        return recovered

    def visualize_batch_tensor_images(self, batch_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # 确保是 list，再转成一个 4D Tensor: (B, C, H, W)


        mean = torch.tensor(mean).view(-1, 1, 1).to(batch_tensor[0].device)
        std = torch.tensor(std).view(-1, 1, 1).to(batch_tensor[0].device)

        batch_size = len(batch_tensor)

        for i in range(batch_size):
            img = batch_tensor[i]
            img = img * std + mean  # 反归一化
            img = img.permute(1, 2, 0).cpu().numpy()  # 转成 HWC
            img = (img * 255).clip(0, 255).astype('uint8')

            plt.figure()
            plt.imshow(img)
            plt.title(f"Image {i}")
            plt.axis('off')
            plt.show()

    def denormalize_to_hwc_tensors(self,batch_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        将 (C, H, W) 的 Tensor 列表反归一化，并转换为 (H, W, C) 格式的 Tensor 列表。

        参数:
            batch_tensor: list of torch.Tensor，每个为 (C, H, W)
            mean: list，归一化时的均值
            std: list，归一化时的标准差

        返回:
            list of torch.Tensor，每个为 (H, W, C)，数值范围 [0, 255]，类型为 float32
        """
        device = batch_tensor[0].device
        mean = torch.tensor(mean, device=device).view(-1, 1, 1)
        std = torch.tensor(std, device=device).view(-1, 1, 1)

        result = []
        for img in batch_tensor:
            img = img * std + mean  # 反归一化
            img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
            img = (img * 255).clamp(0, 255)  # 放大到像素值范围
            result.append(img)

        return result
    @torch.jit.unused
    def _set_hoi_aux_loss(self, pred_actions):
        return [{'pred_actions': [p[l].squeeze(1) for p in pred_actions]} for l in range(self.args.hoi_dec_layers - 1)]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": l, "pred_boxes": b} for l, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # merge boxes (NMS)
    def apply_nms(self, inst_scores, inst_labels, cxcywh_boxes, threshold=0.7):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(cxcywh_boxes)
        box_areas = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(-1)
        box_area_sum = box_areas.unsqueeze(1) + box_areas.unsqueeze(0)

        union_boxes = torch.cat([torch.min(xyxy_boxes.unsqueeze(1)[:, :, :2], xyxy_boxes.unsqueeze(0)[:, :, :2]),
                                 torch.max(xyxy_boxes.unsqueeze(1)[:, :, 2:], xyxy_boxes.unsqueeze(0)[:, :, 2:])], dim=-1)
        union_area = (union_boxes[:,:,2:] - union_boxes[:,:,:2]).prod(-1)
        iou = torch.clamp(box_area_sum - union_area, min=0) / union_area
        box_match_mat = torch.logical_and(iou > threshold, inst_labels.unsqueeze(1) == inst_labels.unsqueeze(0))

        suppress_ids = []
        for box_match in box_match_mat:
            group_ids = box_match.nonzero(as_tuple=False).squeeze(1)
            if len(group_ids) > 1:
                max_score_inst_id = group_ids[inst_scores[group_ids].argmax()]
                bg_ids = group_ids[group_ids!=max_score_inst_id]
                suppress_ids.append(bg_ids)
                box_match_mat[:, bg_ids] = False
        if len(suppress_ids) > 0:
            suppress_ids = torch.cat(suppress_ids, dim=0)
        return suppress_ids
    def get_human_joint(self,img_tensor_rgb,human_box_dict):##img_tensor_rgb h,w,3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_size = np.array([192, 256])
        padding = 1.25
        transform = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        human_joint_dict={}
        human_joint_score={}
        h_ori,w_ori,_=img_tensor_rgb.shape




        for human_idx in human_box_dict:
            # (1) 获取每个人体的检测框坐标
            cx, cy, w, h = human_box_dict[human_idx]

            # (2) 图像预处理，裁剪和缩放检测框
            aspect_ratio  = image_size[0] / image_size[1]
            center = np.array([cx,cy], dtype=np.float32)
            scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
            scale = scale * padding
            trans = get_warp_matrix(0, center * 2.0, image_size - 1.0, scale * 200.0)
            img_uint8 = img_tensor_rgb.cpu().numpy().astype(np.uint8)  # 转为 uint8
            img_pil = Image.fromarray(img_uint8)
            processed_img = cv2.warpAffine(np.array(img_pil), trans, (int(image_size[0]), int(image_size[1])),
                                           flags=cv2.INTER_LINEAR)

            # (3) 将图像送入模型进行推理
            input_tensor = transform(processed_img).to(device)
            img_metas = [{'image_file': None, 'center': center, 'scale': scale, 'rotation': 0, 'bbox_score': 1}]
            with torch.no_grad():
                result = self.vitposemodel.forward(input_tensor.unsqueeze(0), img_metas=img_metas, return_loss=False,
                                   return_heatmap=True)

            # (4) 获取模型预测的关节位置
            heatmap = result['output_heatmap'][0]
            preds = result['preds'][0]
            human_joints = preds[:, :2]
            human_joints[:,0]/=w_ori
            human_joints[:,1]/=h_ori
            human_joints_score = np.max(heatmap, axis=(1, 2))
            human_joint_dict[human_idx]=human_joints
            human_joint_score[human_idx]=human_joints_score

        # # ===== 可视化：画框 + 点 =====
        # img_np = img_tensor_rgb.cpu().numpy()
        # if img_np.max() > 1.0:
        #     img_np = img_np / 255.0  # 归一化
        #
        # fig, ax = plt.subplots(figsize=(10, 12))
        # ax.imshow(img_np)
        #
        # for human_idx, (cx, cy, w, h) in human_box_dict.items():
        #     x1 = cx - w / 2
        #     y1 = cy - h / 2
        #     rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(x1, y1 - 5, f'Human {human_idx}', color='lime', fontsize=10)
        #
        #     # 加上关键点
        #     joints = human_joint_dict[human_idx]
        #     for (x, y) in joints:
        #         ax.plot(x, y, 'ro', markersize=4)  # 红色小圆点
        #
        # ax.set_title("Detected Human Boxes and Joints")
        # ax.axis('off')
        # plt.show()

        return human_joint_dict,human_joint_score

    def draw_human_joints(self,img_tensor_rgb, human_joint_dict):
        # 将图像 tensor 转为 numpy
        img_np = img_tensor_rgb.cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        img_draw = img_np.copy()

        h, w, _ = img_draw.shape  # 获取原图尺寸 (H, W)

        # 遍历每个实例，反归一化关键点后绘制
        for human_idx, joints in human_joint_dict.items():
            for joint in joints:
                x = int(joint[0] * w)  # 将归一化坐标转为像素坐标
                y = int(joint[1] * h)
                cv2.circle(img_draw, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # 显示图像
        plt.figure(figsize=(8, 10))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title('Human Keypoints')
        plt.show()

    def generate_layout_masks(self, rel_pairs, feature_masks, boxes, idx):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:,:2], tail_boxes[:,:2]),
            torch.max(head_boxes[:,2:], tail_boxes[:,2:])
        ], dim=1)

        h, w = (~feature_masks[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # mask: image area=False, pad area=True
        scaled_head_boxes = head_boxes * torch.tensor([w,h,w,h]).to(device=head_boxes.device, dtype=head_boxes.dtype).unsqueeze(0)
        scaled_tail_boxes = tail_boxes * torch.tensor([w,h,w,h]).to(device=tail_boxes.device, dtype=tail_boxes.dtype).unsqueeze(0)
        scaled_union_boxes = union_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0)
        bound_upper_inds = (torch.tensor([w,h,w,h])-1).unsqueeze(0).float().to(feature_masks.device)
        rounded_head_boxes = torch.min(scaled_head_boxes.round(), bound_upper_inds).int()
        rounded_tail_boxes = torch.min(scaled_tail_boxes.round(), bound_upper_inds).int()
        rounded_union_boxes = torch.min(scaled_union_boxes.round(), bound_upper_inds).int()

        role_embeddings = None
        # build masks: hit region=False, other region=True
        rel_head_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        rel_tail_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        rel_union_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        for rid in range(len(rel_union_mask)):
            rel_head_mask[rid, rounded_head_boxes[rid,1]:rounded_head_boxes[rid,3]+1,
                               rounded_head_boxes[rid,0]:rounded_head_boxes[rid,2]+1] = False
            rel_tail_mask[rid, rounded_tail_boxes[rid,1]:rounded_tail_boxes[rid,3]+1,
                               rounded_tail_boxes[rid,0]:rounded_tail_boxes[rid,2]+1] = False
            rel_union_mask[rid, rounded_union_boxes[rid,1]:rounded_union_boxes[rid,3]+1,
                                rounded_union_boxes[rid,0]:rounded_union_boxes[rid,2]+1] = False

        return rel_head_mask, rel_tail_mask, rel_union_mask, role_embeddings

class STIPCriterion(nn.Module):
    """ This class computes the loss for STIP.
    1. proposal loss
    2. relation classification loss
    """
    def __init__(self, args, matcher):
        super().__init__()
        self.args = args
        self.matcher = matcher
        self.weight_dict = {
            'loss_proposal': args.proposal_loss_coef,
            'loss_act': args.action_loss_coef
        }
        if args.hoi_aux_loss:
            for i in range(args.hoi_dec_layers - 1):
                self.weight_dict.update({f'loss_act_{i}': self.weight_dict['loss_act']})

        if args.dataset_file == 'vcoco':
            self.invalid_ids = args.invalid_ids
            self.valid_ids = args.valid_ids
        elif args.dataset_file == 'hico-det':
            self.invalid_ids = []
            self.valid_ids = list(range(self.args.num_actions))
            self.hico_valid_obj_ids = torch.tensor(self.args.valid_obj_ids)

        if args.train_detr:
            self.num_classes = args.num_classes
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = args.eos_coef
            self.register_buffer('empty_weight', empty_weight)

            self.detr_losses = ['labels', 'boxes', 'cardinality']
            det_weights = {'loss_ce': 1 * args.finetune_detr_weight, 'loss_bbox': args.bbox_loss_coef * args.finetune_detr_weight, 'loss_giou': args.giou_loss_coef * args.finetune_detr_weight}
            if args.aux_loss:
                aux_weights = {}
                for i in range(args.dec_layers - 1):
                    aux_weights.update({k + f'_{i}': v for k, v in det_weights.items()})
                det_weights.update(aux_weights)
            self.weight_dict.update(det_weights)

    #######################################################################################################################
    # * DETR Losses
    #######################################################################################################################
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, log=False):
        # instance matching
        if outputs['det2gt_indices'] is None:
            outputs_without_aux = {k: v for k, v in outputs.items() if (k != 'aux_outputs' and k != 'hoi_aux_outputs')}
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = outputs['det2gt_indices']

        # generate relation targets
        all_rel_pair_targets = []
        for imgid, (tgt, (det_idxs, gtbox_idxs)) in enumerate(zip(targets, indices)):
            det2gt_map = {int(d): int(g) for d, g in zip(det_idxs, gtbox_idxs)}
            gt_relation_map = tgt['relation_map']
            rel_pairs = outputs['pred_rel_pairs'][imgid]
            rel_pair_targets = torch.zeros((len(rel_pairs), gt_relation_map.shape[-1])).to(gt_relation_map.device)
            for idx, rel in enumerate(rel_pairs):
                if (int(rel[0]) in det2gt_map) and (int(rel[1]) in det2gt_map):
                    rel_pair_targets[idx] = gt_relation_map[det2gt_map[int(rel[0])], det2gt_map[int(rel[1])]]
            all_rel_pair_targets.append(rel_pair_targets)
        all_rel_pair_targets = torch.cat(all_rel_pair_targets, dim=0)

        prior_verb_label_mask = None
        if self.args.dataset_file == 'hico-det':
            # no_interaction_id = self.args.action_names.index('no_interaction')
            # rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) - all_rel_pair_targets[..., no_interaction_id] > 0).float()
            rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()
            if self.args.use_prior_verb_label_mask:
                pred_obj_labels = outputs['pred_logits'][:,:,self.args.valid_obj_ids].argmax(-1)
                tail_obj_ids = [p[:,1] for p in outputs['pred_rel_pairs']]
                tail_obj_labels = torch.cat([l[id] for l, id in zip(pred_obj_labels, tail_obj_ids)])
                prior_verb_label_mask = self.args.correct_mat.transpose(0,1)[tail_obj_labels]
        else:
            rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()

        loss_proposal = self.proposal_loss(torch.cat(outputs['pred_action_exists'], dim=0), rel_proposal_targets)
        loss_action = self.action_loss(torch.cat(outputs['pred_actions'], dim=0)[..., self.valid_ids], all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)

        loss_dict = {'loss_proposal': loss_proposal, 'loss_act': loss_action}
        if 'hoi_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['hoi_aux_outputs']):
                aux_loss = {
                    f'loss_act_{i}': self.action_loss(torch.cat(aux_outputs['pred_actions'], dim=0)[..., self.valid_ids], all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)
                }
                loss_dict.update(aux_loss)

        # jointly train objects and relation decoder
        if self.args.train_detr:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.detr_losses:
                loss_dict.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.detr_losses:
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        loss_dict.update(l_dict)

        return loss_dict

    def proposal_loss(self, inputs, targets):
        # loss = focal_loss(inputs, targets, gamma=self.args.proposal_focal_loss_gamma, alpha=self.args.proposal_focal_loss_alpha)

        ## conventional BCE
        # loss_bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss = loss_bce[targets<0.5].mean() # neg loss
        # if targets.sum() > 0:
        #     loss += loss_bce[targets>0.5].mean() # pos loss

        # focal loss to balance positive/negative
        probs = inputs.sigmoid()
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.proposal_focal_loss_gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.proposal_focal_loss_gamma) * neg_inds
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def action_loss(self, inputs, targets, prior_verb_label_mask=None):
        # loss = focal_loss(inputs, targets, gamma=self.args.action_focal_loss_gamma, alpha=self.args.action_focal_loss_alpha, prior_verb_label_mask=prior_verb_label_mask)
        probs = inputs.sigmoid()

        # focal loss to balance positive/negative
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.action_focal_loss_gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.action_focal_loss_gamma) * neg_inds
        if prior_verb_label_mask is not None: # mask invalid predictions
            pos_loss = pos_loss * prior_verb_label_mask
            neg_loss = neg_loss * prior_verb_label_mask
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

class STIPPostProcess(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, target_sizes, threshold=0, dataset='coco'):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # for relationship post-processing
        h_indices = [p[:, 0] for p in outputs['pred_rel_pairs']]
        o_indices = [p[:, 1] for p in outputs['pred_rel_pairs']]
        if dataset == 'vcoco':
            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            pair_actions = [a.sigmoid() for a in outputs['pred_actions']]
            # pair_actions = outputs['pred_actions'].sigmoid() * outputs['pred_action_exists'].sigmoid() # cls_score ＊　interactiveness score

            results = []
            for batch_idx, (s, l, b)  in enumerate(zip(scores, labels, boxes)):
                h_inds = (l == 1) & (s > threshold)
                o_inds = (s > threshold)

                h_box, h_cat = b[h_inds], s[h_inds]
                o_box, o_cat = b[o_inds], s[o_inds]

                # for scenario 1 in v-coco dataset
                o_inds = torch.cat((o_inds, torch.ones(1).type(torch.bool).to(o_inds.device)))
                o_box = torch.cat((o_box, torch.Tensor([0, 0, 0, 0]).unsqueeze(0).to(o_box.device))) ## add an empty box

                result_dict = {
                    'h_box': h_box, 'h_cat': h_cat,
                    'o_box': o_box, 'o_cat': o_cat,
                    'scores': s, 'labels': l, 'boxes': b
                }

                K = boxes.shape[1]
                n_act = pair_actions[batch_idx].shape[-1]
                score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
                for h_idx, o_idx, pair_action in zip(h_indices[batch_idx], o_indices[batch_idx], pair_actions[batch_idx]):
                    if h_idx == o_idx: o_idx = -1 ## special case: head=tail
                    score[:, h_idx, o_idx] = pair_action

                score = score[:, h_inds, :]
                score = score[:, :, o_inds]

                result_dict.update({
                    'pair_score': score,
                    'hoi_recognition_time': outputs['hoi_recognition_time'],
                })

                results.append(result_dict)
        elif dataset == 'hico-det':
            # tail classification score
            _valid_obj_ids = self.args.valid_obj_ids + [self.args.valid_obj_ids[-1]+1]
            out_obj_logits = outputs['pred_logits'][..., _valid_obj_ids]
            obj_scores, obj_labels = [], []
            for o_ids, lgts in zip(o_indices, out_obj_logits):
                img_obj_scores, img_obj_labels = F.softmax(lgts[o_ids], -1)[..., :-1].max(-1)
                obj_scores.append(img_obj_scores)
                obj_labels.append(img_obj_labels)

            # actions
            out_verb_logits = outputs['pred_actions']
            verb_scores = [l.sigmoid() for l in out_verb_logits]
            # verb_scores = out_verb_logits.sigmoid() * outputs['pred_action_exists'].sigmoid().unsqueeze(-1) # interactiveness

            # accumulate results (iterate through interaction queries)
            results = []
            for batch_idx, (os, ol, vs, box, h_idx, o_idx) in enumerate(zip(obj_scores, obj_labels, verb_scores, boxes, h_indices, o_indices)):
                # label
                sl = torch.full_like(ol, 0) # self.subject_category_id = 0 in HICO-DET
                l = torch.cat((sl, ol))
                # boxes
                sb = box[h_idx, :]
                ob = box[o_idx, :]
                b = torch.cat((sb, ob))

                vs = vs * os.unsqueeze(1)
                ids = torch.arange(b.shape[0])
                res_dict = {
                    'labels': l.to('cpu'),
                    'boxes': b.to('cpu'),
                    'verb_scores': vs.to('cpu'),
                    'sub_ids': ids[:ids.shape[0] // 2],
                    'obj_ids': ids[ids.shape[0] // 2:],
                    'hoi_recognition_time': outputs['hoi_recognition_time'],
                    'orig_size': torch.tensor([img_h[batch_idx], img_w[batch_idx]])
                }
                results.append(res_dict)

        return results

class RelationFeatureExtractor(nn.Module):
    def __init__(self, args, in_channels, resolution=5, out_dim=1024):
        super(RelationFeatureExtractor, self).__init__()
        self.args = args
        self.resolution = resolution

        # head & tail feature (base feature)
        instr_hidden_dim = self.args.hidden_dim
        fusion_dim = instr_hidden_dim*2

        # spatial feature
        if args.use_spatial_feature:
            spatial_in_dim, spatial_out_dim = 8, 64
            self.spatial_proj = make_fc(spatial_in_dim, spatial_out_dim)
            fusion_dim += spatial_out_dim

        # tail semantic feature
        if args.use_tail_semantic_feature:
            semantic_dim = 300
            self.label_embedding = nn.Embedding(self.args.num_classes+1, semantic_dim)
            fusion_dim += semantic_dim

        # union feature
        if args.use_union_feature:
            out_ch, union_out_dim = 256, 256
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=1),
                nn.ReLU(inplace=True),
            ) # reduce channel size before pooling
            self.visual_proj = make_fc(out_ch * (resolution**2), union_out_dim)
            fusion_dim += union_out_dim
        if args.use_human_pose_feature:
            #  pose feature（新增）
            pose_in_dim = 17 * 4  # 17个点 × （x, y, dx, dy）
            pose_out_dim = 256
            self.pose_proj = make_fc(pose_in_dim, pose_out_dim)
            fusion_dim += pose_out_dim
        if args.use_IOB:
            self.box_links = {
                0: (0, 1),
                1: (1, 3),
                2: (2, 4),
                3: (1, 3),
                4: (2, 4),
                5: (5, 7),
                6: (6, 8),
                7: (7, 9),
                8: (8, 10),
                9: (7, 9),
                10: (8, 10),
                11: (11, 13),
                12: (12, 14),
                13: (13, 15),
                14: (14, 16),
                15: (13, 15),
                16: (14, 16),
            }
            IOB_in_dim, IOB_out_dim = 17, 128
            self.IOB_proj = make_fc(IOB_in_dim, IOB_out_dim)
            fusion_dim += IOB_out_dim
        # fusion
        self.fusion_fc = nn.Sequential(
            make_fc(fusion_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )

    def forward(self, rel_pairs, features, boxes, inst_reprs,idx,human_joint_dict=None,human_joint_score=None,  obj_label_logits=None):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (cx_cy_wh, nomalized to 0-1)
            rel_pairs: Nx2
        """
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:,:2], tail_boxes[:,:2]),
            torch.max(head_boxes[:,2:], tail_boxes[:,2:])
        ], dim=1)

        # head & tail features
        head_feats = inst_reprs[rel_pairs[:,0]]
        tail_feats = inst_reprs[rel_pairs[:,1]]
        tail_feats[rel_pairs[:,0]==rel_pairs[:,1]] = 0 # set to 0 when head==tail for VCOCO (i.e., tail overlapped)

        relation_feats = torch.cat([head_feats, tail_feats], dim=-1)

        # spatial layout feats
        if self.args.use_spatial_feature:
            box_layout_feats = self.extract_spatial_layout_feats(xyxy_boxes)
            rel_spatial_feats = self.spatial_proj(box_layout_feats[rel_pairs[:,0], rel_pairs[:,1]])#8映射到64
            relation_feats = torch.cat([relation_feats, rel_spatial_feats], dim=-1)

        # semantic feature
        if self.args.use_tail_semantic_feature:
            semantic_feats = (obj_label_logits.softmax(-1) @ self.label_embedding.weight)[rel_pairs[:,1]]
            relation_feats = torch.cat([relation_feats, semantic_feats], dim=-1)

        # union feature
        if self.args.use_union_feature:
            # H, W = features.tensors.shape[-2:] # stacked image size
            h, w = (~features.mask[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # mask: image area=False, pad area=True
            proj_feature = self.input_proj(features.tensors[idx:idx+1])
            scaled_union_boxes = torch.cat(
                [
                    torch.zeros((len(union_boxes),1)).to(device=union_boxes.device),
                    union_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0),
                ], dim=-1
            )
            union_visual_feats = roi_align(proj_feature, scaled_union_boxes, output_size=self.resolution, sampling_ratio=2)
            union_visual_feats = self.visual_proj(union_visual_feats.flatten(start_dim=1))
            relation_feats = torch.cat([relation_feats, union_visual_feats], dim=-1)
        #humanpose feature
        if self.args.use_human_pose_feature:
            human_feats_tensor = []
            num_joints = 17  # 默认使用 17 个关键点

            for i in range(len(rel_pairs)):
                h_idx, o_idx = rel_pairs[i]
                h_idx = h_idx.item() if isinstance(h_idx, torch.Tensor) else h_idx
                o_idx = o_idx.item() if isinstance(o_idx, torch.Tensor) else o_idx

                if (human_joint_dict is not None and human_joint_score is not None and
                        h_idx in human_joint_dict and h_idx in human_joint_score):
                    joints = human_joint_dict[h_idx]  # (K, 2)
                    # scores = human_joint_score[h_idx]  # (K,)
                    # joints_weighted = joints * scores[:, None]  # (K, 2)

                    obj_box = xyxy_boxes[o_idx]
                    obj_center = ((obj_box[:2] + obj_box[2:]) / 2).cpu().numpy()  # (2,)

                    vecs = joints - obj_center  # (K, 2)
                    # vecs_weighted = vecs * scores[:, None]  # (K, 2)

                    pose_feat = np.concatenate([joints, vecs], axis=1).flatten()  # (K*4,)
                else:
                    # 如果关键点缺失，填充 0 向量（K*4）
                    pose_feat = np.zeros((num_joints * 4,), dtype=np.float32)

                human_feats_tensor.append(torch.tensor(pose_feat, dtype=torch.float32, device=boxes.device))

            human_feats_tensor = torch.stack(human_feats_tensor)  # shape: (N, K*4)
            pose_feat_proj = self.pose_proj(human_feats_tensor)
            relation_feats = torch.cat([relation_feats, pose_feat_proj], dim=-1)
        else:
            print("not use pose")
        if self.args.use_IOB:
            human_boxes_dict=self.compute_keypoint_boxes(human_joint_dict)
            iob_matrix = self.compute_iob_matrix(human_boxes_dict, rel_pairs, xyxy_boxes)
            IOB_feats=self.IOB_proj(iob_matrix.to(boxes.device))
            relation_feats = torch.cat([relation_feats, IOB_feats], dim=-1)

        else:
            print("not use IOB")



        x = self.fusion_fc(relation_feats)
        return x

    def compute_box_area(self,box):
        w = max(0, box[2] - box[0])
        h = max(0, box[3] - box[1])
        return w * h

    def compute_intersection(self,box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return w * h

    def compute_iob_matrix(self,human_boxes_dict, rel_pairs, object_boxes):
        """
        计算每个（人，物体）对中，人17个关键点 box 与物体 box 的 IOB
        :param human_boxes_dict: {human_idx: ndarray (17, 4)}
        :param rel_pairs: Tensor (N, 2) 每行是 [human_idx, object_idx]
        :param object_boxes: Tensor (M, 4)，每个 box 是 [x1, y1, x2, y2]
        :return: Tensor (N, 17)，每个元素是 IOB
        """
        num_pairs = rel_pairs.size(0)
        iob_matrix = torch.zeros((num_pairs, 17), dtype=torch.float32)
        # print(f"---{torch.unique(rel_pairs[:,0])}")
        # print(f"---{human_boxes_dict.keys()}")
        # 提前获取 unique human index + dict keys
        human_indices = rel_pairs[:, 0]
        unique_human_indices = torch.unique(human_indices).tolist()
        for idx in range(num_pairs):
            human_idx, object_idx = rel_pairs[idx].tolist()
            if human_idx not in human_boxes_dict:
                iob_matrix[idx,:]=0
            else:


                human_boxes = human_boxes_dict[human_idx]  # (17, 4)

                object_box = object_boxes[object_idx].cpu().numpy()  # (4,)
                object_area=self.compute_box_area(object_box)
                for k in range(17):
                    human_box = human_boxes[k]
                    inter_area = self.compute_intersection(human_box, object_box)
                    human_area = self.compute_box_area(human_box)+object_area-inter_area
                    iob = inter_area / human_area if human_area > 0 else 0.0
                    iob_matrix[idx, k] = iob

        return iob_matrix
    def compute_keypoint_boxes(self,human_joint_dict):
        """
        输入: human_joint_dict: {人索引: ndarray(17, 2)}
        输出: human_boxes_dict: {人索引: ndarray(17, 4)} 每个关键点 box 是 [x1, y1, x2, y2]
        """
        human_boxes_dict = {}
        for human_idx, joints in human_joint_dict.items():
            boxes = []
            for i in range(17):
                if i not in self.box_links:
                    boxes.append([0, 0, 0, 0])  # 占位
                    continue
                p1, p2 = self.box_links[i]
                joint0 = joints[p1]
                joint1 = joints[p2]
                center = (joint0 + joint1) / 2
                w = abs(joints[i][0] - center[0]) * 2
                h = abs(joints[i][1] - center[1]) * 2
                x1 = joints[i][0] - w / 2
                y1 = joints[i][1] - h / 2
                x2 = joints[i][0] + w / 2
                y2 = joints[i][1] + h / 2
                boxes.append([x1, y1, x2, y2])
            human_boxes_dict[human_idx] = np.array(boxes)
        return human_boxes_dict
    def extract_spatial_layout_feats(self, xyxy_boxes):
        box_center = torch.stack([(xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2, (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2], dim=1)
        dxdy = box_center.unsqueeze(1) - box_center.unsqueeze(0) # distances
        theta = (torch.atan2(dxdy[...,1], dxdy[...,0]) / np.pi).unsqueeze(-1)
        dis = dxdy.norm(dim=-1, keepdim=True)

        box_area = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(dim=1) # areas
        intersec_lt = torch.max(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
        intersec_rb = torch.min(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
        overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        union_lt = torch.min(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
        union_rb = torch.max(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
        union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        spatial_feats = torch.cat([
            dxdy, dis, theta, # dx, dy, distance, theta
            overlap, union, box_area[:,None,None].expand(*union.shape), box_area[None,:,None].expand(*union.shape) # overlap, union, subj, obj
        ], dim=-1)
        return spatial_feats

# conventional focal loss to balance hard/easy
def focal_loss(blogits, target_classes, alpha=0.5, gamma=2, prior_verb_label_mask=None, class_weights=None):
    probs = blogits.sigmoid() # prob(positive)
    loss_bce = F.binary_cross_entropy_with_logits(blogits, target_classes, reduction='none', weight=class_weights)
    p_t = probs * target_classes + (1 - probs) * (1 - target_classes)
    loss_bce = ((1-p_t)**gamma * loss_bce)

    alpha_t = alpha * target_classes + (1 - alpha) * (1 - target_classes)
    loss_focal = alpha_t * loss_bce

    if prior_verb_label_mask is not None:
        loss_focal = loss_focal * prior_verb_label_mask

    loss = loss_focal.sum() / max(target_classes.sum(), 1)
    return loss

def make_fc(dim_in, hidden_dim, a=1):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
        a: negative slope
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=a)
    nn.init.constant_(fc.bias, 0)
    return fc


def make_conv3x3(
    in_channels,
    out_channels,
    padding=1,
    dilation=1,
    stride=1,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
        nn.init.constant_(conv.bias, 0)
    return conv