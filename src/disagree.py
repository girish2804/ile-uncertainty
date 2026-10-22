import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


def _relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., :1] + np.log(relative_top)
    probs_to_filter = sorted_logits < min_thresh
    probs_to_filter[..., :min_tokens_to_keep] = False
    indices_to_remove = probs_to_filter.scatter(1, sorted_indices, probs_to_filter)
    scores_filtered = scores.masked_fill(indices_to_remove, -float("Inf"))
    return scores_filtered, indices_to_remove


def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., :1] + np.log(relative_top)
    probs_to_filter = sorted_logits < min_thresh
    probs_to_filter[..., :min_tokens_to_keep] = False
    indices_to_remove = probs_to_filter.scatter(1, sorted_indices, probs_to_filter)
    return indices_to_remove


def calculate_all_disagreement_metrics(final_logits, premature_logits, 
                                       calculate_metrics=None,
                                       relative_top=0.1,
                                       min_tokens_to_keep=32, store_logits=False,
                                       embeddings=None,
                                       skip_top_neighbors=0):
    """
    Calculate all disagreement and epistemic uncertainty metrics across premature layers vs the final layer.
    
    Args:
        final_logits: Logits from the final layer [batch_size, vocab_size]
        premature_logits: Dictionary/List of logits from earlier layers [batch_size, vocab_size]
        calculate_metrics: List of metrics to calculate (default: all)
        relative_top: The relative threshold for filtering (e.g. 0.1)
        min_tokens_to_keep: Minimum number of tokens to keep
        skip_top_neighbors: Number of immediate top premature layers to skip for neighbor-excluded metrics
    
    Returns:
        Dictionary of aggregated disagreement metrics
    """
    if calculate_metrics is None:
        calculate_metrics = ['jsd', 'kl', 'entropy_mixture', 'mutual_info', 'gated_mutual_info',
                             'epkl', 'cv', 'bhattacharyya', 'tvd',
                             'entropy_mixture_dynamic', 'entropy_mixture_maxprob', 'entropy_mixture_decay', 'sve']
    
    batch_size, vocab_size = final_logits.shape
    premature_logits_list = [premature_logits[x] for x in premature_logits] if isinstance(premature_logits, dict) else premature_logits
    n_layers = len(premature_logits_list)
    
    # Get filter mask based on final layer logits
    filter_mask = ~get_relative_top_filter(final_logits, 
                                           relative_top=relative_top, 
                                           min_tokens_to_keep=min_tokens_to_keep)
    
    # Convert final logits to probabilities and log probabilities
    final_probs = F.softmax(final_logits, dim=-1)
    masked_final_probs = final_probs * filter_mask
    normalization_factor = masked_final_probs.sum(dim=-1, keepdim=True)
    normalized_final_probs = masked_final_probs / (normalization_factor + 1e-10)
    normalized_final_log_probs = torch.log(normalized_final_probs + 1e-10)
    
    # Final layer entropy
    entropy_final = -torch.sum(normalized_final_probs * normalized_final_log_probs, dim=-1)
    
    results = {}
    if store_logits:
        results['unfiltered_final_logits'] = final_logits.clone().detach().to('cpu')
        results['unfiltered_premature_logits'] = [logits.clone().detach().to('cpu') for logits in premature_logits_list]
        results['tokens_kept_count'] = torch.sum(filter_mask[0]).item()
        results['tokens_kept_percentage'] = (torch.sum(filter_mask[0]).item() / vocab_size) * 100
    
    results['entropy_final'] = entropy_final.mean().item()
    metric_values = {metric: [] for metric in calculate_metrics}
    
    # 3. Normalize all premature layers
    premature_probs_list = []
    premature_log_probs_list = []
    premature_entropies_list = []
    
    for p_logits in premature_logits_list:
        premature_probs = F.softmax(p_logits, dim=-1)
        masked_premature_probs = premature_probs * filter_mask
        premature_norm_factor = masked_premature_probs.sum(dim=-1, keepdim=True)
        normalized_premature_probs = masked_premature_probs / (premature_norm_factor + 1e-10)
        normalized_premature_log_probs = torch.log(normalized_premature_probs + 1e-10)
        entropy_premature = -torch.sum(normalized_premature_probs * normalized_premature_log_probs, dim=-1)
        
        premature_probs_list.append(normalized_premature_probs)
        premature_log_probs_list.append(normalized_premature_log_probs)
        premature_entropies_list.append(entropy_premature)

    directional_kl_gains = []
    true_conflict_jsds = []

    for layer_idx in range(n_layers):
        normalized_premature_probs = premature_probs_list[layer_idx]
        normalized_premature_log_probs = premature_log_probs_list[layer_idx]
        entropy_premature = premature_entropies_list[layer_idx]
        
        # 1. Mixture Probability Distribution
        m_probs = 0.5 * (normalized_final_probs + normalized_premature_probs)
        h_m = -torch.sum(m_probs * torch.log(m_probs + 1e-10), dim=-1)
        
        # Delta H: H(premature) - H(final)
        delta_h = entropy_premature - entropy_final
        conflict_mask = (delta_h <= 0.0).float()
        
        for metric in calculate_metrics:
            if metric in ['jsd', 'mutual_info']:
                # Jensen-Shannon Divergence / Pairwise Mutual Information: H(M) - 0.5*(H(P_final) + H(P_prem))
                # Strictly >= 0 by Jensen's inequality; original formula untouched
                jsd = h_m - 0.5 * (entropy_final + entropy_premature)
                jsd = torch.clamp(jsd, min=0.0)
                metric_values[metric].append(jsd.mean().item())
                
                true_conflict_jsds.append((jsd * conflict_mask).mean().item())
                
            elif metric == 'gated_mutual_info':
                # Conflict-Gated Mutual Information: MI * I(H_prem - H_final <= 0)
                jsd = h_m - 0.5 * (entropy_final + entropy_premature)
                jsd = torch.clamp(jsd, min=0.0)
                gated_jsd = jsd * conflict_mask
                metric_values[metric].append(gated_jsd.mean().item())
                
            elif metric == 'kl':
                # Forward KL Divergence: D_KL(P_final || P_premature) = sum_v P_final * (log P_final - log P_premature)
                kl_val = torch.sum(normalized_final_probs * (normalized_final_log_probs - normalized_premature_log_probs), dim=-1)
                kl_val = torch.clamp(kl_val, min=0.0)
                metric_values[metric].append(kl_val.mean().item())
                
                gain = kl_val * torch.clamp(delta_h, min=0.0)
                directional_kl_gains.append(gain.mean().item())
                
            elif metric == 'epkl':
                # Symmetric / Expected Pairwise KL Divergence
                sym_kl = 0.5 * torch.sum((normalized_final_probs - normalized_premature_probs) * (normalized_final_log_probs - normalized_premature_log_probs), dim=-1)
                sym_kl = torch.clamp(sym_kl, min=0.0)
                metric_values[metric].append(sym_kl.mean().item())
                
            elif metric == 'entropy_mixture':
                # Total Predictive Uncertainty of the Mixture
                metric_values[metric].append(h_m.mean().item())
                
            elif metric == 'entropy_mixture_dynamic':
                w_final = torch.exp(-entropy_final.unsqueeze(-1)) / (torch.exp(-entropy_final.unsqueeze(-1)) + torch.exp(-entropy_premature.unsqueeze(-1)) + 1e-10)
                m_dyn_probs = w_final * normalized_final_probs + (1.0 - w_final) * normalized_premature_probs
                h_dyn = -torch.sum(m_dyn_probs * torch.log(m_dyn_probs + 1e-10), dim=-1)
                metric_values[metric].append(h_dyn.mean().item())
                
            elif metric == 'entropy_mixture_maxprob':
                max_final = torch.max(normalized_final_probs, dim=-1, keepdim=True).values
                max_premature = torch.max(normalized_premature_probs, dim=-1, keepdim=True).values
                w_final = max_final / (max_final + max_premature + 1e-10)
                m_max_probs = w_final * normalized_final_probs + (1.0 - w_final) * normalized_premature_probs
                h_max = -torch.sum(m_max_probs * torch.log(m_max_probs + 1e-10), dim=-1)
                metric_values[metric].append(h_max.mean().item())
                
            elif metric == 'entropy_mixture_decay':
                decay_weight = 1.0 - 0.5 * np.exp(-0.15 * (n_layers - layer_idx))
                m_dec_probs = decay_weight * normalized_final_probs + (1.0 - decay_weight) * normalized_premature_probs
                h_dec = -torch.sum(m_dec_probs * torch.log(m_dec_probs + 1e-10), dim=-1)
                metric_values[metric].append(h_dec.mean().item())
                
            elif metric == 'cv':
                stacked_probs = torch.stack([normalized_final_probs, normalized_premature_probs], dim=0)
                mean_probs = torch.mean(stacked_probs, dim=0)
                std_probs = torch.std(stacked_probs, dim=0)
                cv_values = std_probs / (mean_probs + 1e-10)
                masked_cv = cv_values * filter_mask
                values = torch.sum(masked_cv, dim=-1) / (torch.sum(filter_mask, dim=-1) + 1e-10)
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'bhattacharyya':
                bc_coeff = torch.sum(torch.sqrt(normalized_final_probs * normalized_premature_probs + 1e-12), dim=-1)
                bhat_dist = -torch.log(torch.clamp(bc_coeff, min=1e-10, max=1.0))
                metric_values[metric].append(bhat_dist.mean().item())
                
            elif metric == 'tvd':
                tvd_val = 0.5 * torch.sum(torch.abs(normalized_final_probs - normalized_premature_probs), dim=-1)
                metric_values[metric].append(tvd_val.mean().item())
                
            elif metric == 'sve':
                if embeddings is not None:
                    sve_val = calculate_sve(final_logits, embeddings, top_k=10)
                    metric_values[metric].append(sve_val.mean().item())
                else:
                    metric_values[metric].append(0.0)
    
    # Aggregate metrics across all layers (Star Topology)
    for metric in calculate_metrics:
        if metric_values[metric]:
            results[metric] = sum(metric_values[metric]) / len(metric_values[metric])
            results[metric + '_min'] = min(metric_values[metric])
            results[metric + '_max'] = max(metric_values[metric])
            
            # Neighbor-excluded metrics (skip top 2 adjacent layers to eliminate surface-level calibration jitter)
            if n_layers > 2 and metric in ['mutual_info', 'gated_mutual_info', 'jsd']:
                vals_skip2 = metric_values[metric][:-2]
                if vals_skip2:
                    results[f'{metric}_skip2'] = sum(vals_skip2) / len(vals_skip2)
            elif n_layers <= 2 and metric in ['mutual_info', 'gated_mutual_info', 'jsd']:
                results[f'{metric}_skip2'] = results[metric]
                
            if skip_top_neighbors > 0 and n_layers > skip_top_neighbors:
                vals_skip = metric_values[metric][:-skip_top_neighbors]
                if vals_skip:
                    results[f'{metric}_skip_neighbors'] = sum(vals_skip) / len(vals_skip)
            
    # Directional metrics
    if directional_kl_gains:
        results['directional_kl_gain'] = sum(directional_kl_gains) / len(directional_kl_gains)
    if true_conflict_jsds:
        results['true_epistemic_conflict'] = sum(true_conflict_jsds) / len(true_conflict_jsds)

    # =========================================================================
    # Consecutive Chain Metrics (Trajectory Path Length & Turbulence)
    # =========================================================================
    all_sequence_probs = premature_probs_list + [normalized_final_probs]
    all_sequence_log_probs = premature_log_probs_list + [normalized_final_log_probs]
    all_sequence_entropies = premature_entropies_list + [entropy_final]
    
    consecutive_jsd = []
    consecutive_kl = []
    consecutive_tvd = []
    
    for i in range(len(all_sequence_probs) - 1):
        p_i, lp_i, h_i = all_sequence_probs[i], all_sequence_log_probs[i], all_sequence_entropies[i]
        p_next, lp_next, h_next = all_sequence_probs[i + 1], all_sequence_log_probs[i + 1], all_sequence_entropies[i + 1]
        
        # Stepwise JSD
        m_step = 0.5 * (p_i + p_next)
        h_m_step = -torch.sum(m_step * torch.log(m_step + 1e-10), dim=-1)
        step_jsd = torch.clamp(h_m_step - 0.5 * (h_i + h_next), min=0.0).mean().item()
        consecutive_jsd.append(step_jsd)
        
        # Stepwise Forward KL: D_KL(P_{i+1} || P_i)
        step_kl = torch.clamp(torch.sum(p_next * (lp_next - lp_i), dim=-1), min=0.0).mean().item()
        consecutive_kl.append(step_kl)
        
        # Stepwise TVD
        step_tvd = (0.5 * torch.sum(torch.abs(p_next - p_i), dim=-1)).mean().item()
        consecutive_tvd.append(step_tvd)
        
    if consecutive_jsd:
        results['jsd_chain_path_length'] = sum(consecutive_jsd)
        results['jsd_chain_peak_step'] = max(consecutive_jsd)
        results['jsd_chain_mean_velocity'] = sum(consecutive_jsd) / len(consecutive_jsd)
        results['kl_chain_path_length'] = sum(consecutive_kl)
        results['tvd_chain_path_length'] = sum(consecutive_tvd)
        
        # Tortuosity ratio
        star_jsd = results.get('jsd', 0.0)
        results['jsd_tortuosity'] = results['jsd_chain_path_length'] / (star_jsd + 1e-6)
        star_kl = results.get('kl', 0.0)
        results['kl_tortuosity'] = results['kl_chain_path_length'] / (star_kl + 1e-6)
    
    return results


def calculate_sve(logits: torch.FloatTensor, embeddings: torch.FloatTensor, top_k: int = 10) -> torch.FloatTensor:
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    batch_size = logits.shape[0]
    sve_values = torch.zeros(batch_size, device=logits.device)
    
    for i in range(batch_size):
        top_embeds = embeddings[top_indices[i]]
        norm_embeds = F.normalize(top_embeds, p=2, dim=-1)
        sim_matrix = torch.mm(norm_embeds, norm_embeds.t())
        dissim_matrix = 1.0 - sim_matrix
        weights = top_probs[i].unsqueeze(1) * top_probs[i].unsqueeze(0)
        weights = weights / weights.sum()
        sve_values[i] = torch.sum(weights * dissim_matrix)
        
    return sve_values


def compute_all_layers_disagreement_vectorized(final_logits: torch.Tensor,
                                              premature_logits: torch.Tensor,
                                              total_layers: int,
                                              relative_top: float = 0.1,
                                              min_tokens_to_keep: int = 32,
                                              depth_ratio: float = 0.75) -> dict:
    """
    Vectorized computation of all disagreement and epistemic metrics across all layer depths k in 1 ... total_layers - 1.
    Computes all premature layer divergences in a single GPU pass and uses fast slice aggregations.
    """
    if final_logits.dim() == 1:
        final_logits = final_logits.unsqueeze(0)
    if premature_logits.dim() == 2 and premature_logits.shape[0] != total_layers and premature_logits.shape[1] == total_layers:
        premature_logits = premature_logits.t()
    
    # Filter mask from final logits
    filter_mask = ~get_relative_top_filter(final_logits, relative_top=relative_top, min_tokens_to_keep=min_tokens_to_keep)
    
    # Normalized final distribution
    final_probs = F.softmax(final_logits, dim=-1)
    masked_final_probs = final_probs * filter_mask
    norm_final_probs = masked_final_probs / (masked_final_probs.sum(dim=-1, keepdim=True) + 1e-10)
    norm_final_log_probs = torch.log(norm_final_probs + 1e-10)
    entropy_final = -torch.sum(norm_final_probs * norm_final_log_probs, dim=-1).squeeze()
    
    # Vectorized normalization across all premature layers
    prem_probs = F.softmax(premature_logits, dim=-1)
    masked_prem_probs = prem_probs * filter_mask  # [total_layers, vocab_size]
    norm_prem_probs = masked_prem_probs / (masked_prem_probs.sum(dim=-1, keepdim=True) + 1e-10)
    norm_prem_log_probs = torch.log(norm_prem_probs + 1e-10)
    entropy_prem = -torch.sum(norm_prem_probs * norm_prem_log_probs, dim=-1)  # [total_layers]
    
    # Pairwise mixture & metrics with final layer in one vectorized pass
    m_probs = 0.5 * (norm_final_probs + norm_prem_probs)  # [total_layers, vocab_size]
    h_m = -torch.sum(m_probs * torch.log(m_probs + 1e-10), dim=-1)  # [total_layers]
    
    jsd = torch.clamp(h_m - 0.5 * (entropy_final + entropy_prem), min=0.0)  # [total_layers]
    delta_h = entropy_prem - entropy_final
    conflict_mask = (delta_h <= 0.0).float()
    gated_mi = jsd * conflict_mask  # [total_layers]
    
    kl = torch.clamp(torch.sum(norm_final_probs * (norm_final_log_probs - norm_prem_log_probs), dim=-1), min=0.0)
    epkl = torch.clamp(0.5 * torch.sum((norm_final_probs - norm_prem_probs) * (norm_final_log_probs - norm_prem_log_probs), dim=-1), min=0.0)
    tvd = 0.5 * torch.sum(torch.abs(norm_final_probs - norm_prem_probs), dim=-1)
    bc_coeff = torch.sum(torch.sqrt(norm_final_probs * norm_prem_probs + 1e-12), dim=-1)
    bhattacharyya = -torch.log(torch.clamp(bc_coeff, min=1e-10, max=1.0))
    
    # Convert per-layer vectors to CPU numpy for fast cumulative slicing
    h_m_np = h_m.detach().cpu().numpy()
    jsd_np = jsd.detach().cpu().numpy()
    gated_mi_np = gated_mi.detach().cpu().numpy()
    kl_np = kl.detach().cpu().numpy()
    epkl_np = epkl.detach().cpu().numpy()
    tvd_np = tvd.detach().cpu().numpy()
    bhat_np = bhattacharyya.detach().cpu().numpy()
    
    results = {}
    results['entropy_final'] = float(entropy_final.detach().cpu().item())
    
    # Full layer sweep across all depths k in 1 ... total_layers - 1
    for k in range(1, total_layers):
        start_idx = total_layers - k
        results[f'entropy_mixture_top_{k}'] = float(np.mean(h_m_np[start_idx:total_layers]))
        results[f'mutual_info_top_{k}'] = float(np.mean(jsd_np[start_idx:total_layers]))
        results[f'gated_mutual_info_top_{k}'] = float(np.mean(gated_mi_np[start_idx:total_layers]))
        if k > 2:
            results[f'mutual_info_top_{k}_skip2'] = float(np.mean(jsd_np[start_idx:total_layers - 2]))
            results[f'gated_mutual_info_top_{k}_skip2'] = float(np.mean(gated_mi_np[start_idx:total_layers - 2]))
        else:
            results[f'mutual_info_top_{k}_skip2'] = results[f'mutual_info_top_{k}']
            results[f'gated_mutual_info_top_{k}_skip2'] = results[f'gated_mutual_info_top_{k}']

    # Depth 75% normalized metrics
    k_75 = max(1, int(round(depth_ratio * total_layers)))
    start_75 = total_layers - k_75
    results['entropy_mixture_depth_75'] = float(np.mean(h_m_np[start_75:total_layers]))
    results['mutual_info_depth_75'] = float(np.mean(jsd_np[start_75:total_layers]))
    results['gated_mutual_info_depth_75'] = float(np.mean(gated_mi_np[start_75:total_layers]))
    if k_75 > 2:
        results['mutual_info_depth_75_skip2'] = float(np.mean(jsd_np[start_75:total_layers - 2]))
        results['gated_mutual_info_depth_75_skip2'] = float(np.mean(gated_mi_np[start_75:total_layers - 2]))
    else:
        results['mutual_info_depth_75_skip2'] = results['mutual_info_depth_75']
        results['gated_mutual_info_depth_75_skip2'] = results['gated_mutual_info_depth_75']

    # Standard baselines
    k_a = min(2, total_layers - 1)
    k_e = min(16, total_layers - 1)
    results['entropy_mixture'] = results[f'entropy_mixture_top_{k_a}']
    results['mutual_info'] = results[f'mutual_info_top_{k_e}']
    results['gated_mutual_info'] = results[f'gated_mutual_info_top_{k_e}']
    results['jsd'] = float(np.mean(jsd_np[start_75:total_layers]))
    results['kl'] = float(np.mean(kl_np[start_75:total_layers]))
    results['epkl'] = float(np.mean(epkl_np[start_75:total_layers]))
    results['tvd'] = float(np.mean(tvd_np[start_75:total_layers]))
    results['bhattacharyya'] = float(np.mean(bhat_np[start_75:total_layers]))
    
    std_mi = np.std(jsd_np[start_75:total_layers])
    mean_mi = np.mean(jsd_np[start_75:total_layers])
    results['cv'] = float(std_mi / (mean_mi + 1e-10))

    return results
