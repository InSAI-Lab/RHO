
import torch
import torch.nn.functional as F

def multi_positive_infonce_loss(anchor_embeddings, positive_embeddings, temperature=0.1):
    """
    Calculates the InfoNCE loss with multiple positives for each anchor.
    This version correctly separates positive and negative samples.

    Args:
        anchor_embeddings (torch.Tensor): [N, D] - anchor embeddings
        positive_embeddings (torch.Tensor): [N, P, D] - positive embeddings
        N for the numbe of anchors, P for the number of positives per anchor, 
        D for dimension?
        temperature (float): temperature parameter
    
    Returns:
        torch.Tensor: InfoNCE loss
    """
    # Normalize embeddings to use cosine similarity via dot product
    anchor_embeddings = F.normalize(anchor_embeddings, dim=1) # [N, D]
    positive_embeddings = F.normalize(positive_embeddings, dim=2) # [N, P, D]
    
    N, P, D = positive_embeddings.shape
    device = anchor_embeddings.device

    # --- Positive Similarities ---
    # # Similarity between each anchor and its own P positive samples
    # Shape: [N, P]
    positive_sims = torch.einsum('nd,npd->np', anchor_embeddings, positive_embeddings) / temperature

    # --- Negative Similarities ---
    # For anchor i, negatives are:
    # 1. All other anchors (j != i)
    # 2. All positives belonging to other anchors (not anchor i's own positives)
    
    losses = []

    for i in range(N):
        # Current Anchor
        anchor_i = anchor_embeddings[i:i+1] # [1, D]

        # Positives for anchor_i
        positives_i = positive_sims[i] # [P]

        # Negatives for anchor_i:
        # 1. All other anchors (j != i)
        other_anchors = torch.cat([anchor_embeddings[:i], anchor_embeddings[i+1:]], dim=0) # [N-1, D]

        if other_anchors.size(0) > 0:
            anchor_neg_sims = torch.matmul(anchor_i, other_anchors.T) / temperature # [1, N-1]
            anchor_neg_sims = anchor_neg_sims.squeeze(0) # [N-1]
        else:
            anchor_neg_sims = torch.tensor([], device=device)
        
        # 2. Positives from other anchors
        other_positives = torch.cat([positive_embeddings[:i], positive_embeddings[i+1:]], dim=0) # # [N-1, P, D]
        if other_anchors.size(0) > 0:
            other_positives_flat = other_positives.view(-1, D)  # [(N-1)*P, D]
            pos_neg_sims = torch.matmul(anchor_i, other_positives_flat.T) / temperature # [1, (N-1)*P]
            pos_neg_sims = pos_neg_sims.squeeze(0)  # [(N-1)*P]
        else:
            pos_neg_sims = torch.tensor([], device=device)
        
        # Combine all negative similarities:
        if anchor_neg_sims.numel() > 0 and pos_neg_sims.numel() > 0:
            all_neg_sims = torch.cat([anchor_neg_sims, pos_neg_sims], dim=0)
        elif anchor_neg_sims.numel() > 0:
            all_neg_sims = anchor_neg_sims
        elif pos_neg_sims.numel() > 0:
            all_neg_sims = pos_neg_sims
        else:
            all_neg_sims = torch.tensor([], device=device)

        # Compute InfoNCE loss for anchor i
        # Numerator: log-sum-exp of positive similarities
        log_numerator = torch.logsumexp(positives_i, dim=0)

        # Denominator: log-sum-exp of positive + negative similarities
        if all_neg_sims.numel() > 0:
            all_sims_i = torch.cat([positives_i, all_neg_sims], dim=0)
        else:
            all_sims_i = positives_i

        log_denominator = torch.logsumexp(all_sims_i, dim=0)

        # InfoNCE loss: -log(numerator/denominator) = log_denominator - log_numerator
        loss_i = log_denominator - log_numerator
        losses.append(loss_i)

    return torch.stack(losses).mean()

# More efficient vectorized version
def multi_positive_infonce_loss_vectorized(anchor_embeddings, positive_embeddings, temperature=0.1):
    """
    Vectorized version of corrected InfoNCE loss
    """
    # Normalize embeddings
    anchor_embeddings = F.normalize(anchor_embeddings, dim=1)  # [N, D]
    positive_embeddings = F.normalize(positive_embeddings, dim=2)  # [N, P, D]
    
    N, P, D = positive_embeddings.shape
    device = anchor_embeddings.device

    # Create all embeddings: [anchors; all_positives_flattened]
    all_positives_flat = positive_embeddings.view(N * P, D)  # [N*P, D]
    all_embeddings = torch.cat([anchor_embeddings, all_positives_flat], dim=0)  # [N + N*P, D]
    
    # Compute full similarity matrix
    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T) / temperature  # [N+N*P, N+N*P]
    
    losses = []
    
    for i in range(N):
        # Similarities from anchor i to all embeddings
        anchor_sims = sim_matrix[i]  # [N + N*P]
        
        # Positive indices for anchor i: N + i*P to N + (i+1)*P
        pos_start = N + i * P
        pos_end = N + (i + 1) * P
        pos_indices = list(range(pos_start, pos_end))
        
        # Get positive similarities
        pos_sims = anchor_sims[pos_indices]  # [P]
        
        # Create mask for negatives (exclude self and own positives)
        neg_mask = torch.ones(N + N * P, dtype=torch.bool, device=device)
        neg_mask[i] = False  # Exclude self (anchor)
        neg_mask[pos_indices] = False  # Exclude own positives
        
        # Get negative similarities
        neg_sims = anchor_sims[neg_mask]  # [N-1 + (N-1)*P]
        
        # Compute InfoNCE loss
        log_numerator = torch.logsumexp(pos_sims, dim=0)
        all_sims_i = torch.cat([pos_sims, neg_sims], dim=0)
        log_denominator = torch.logsumexp(all_sims_i, dim=0)
        
        loss_i = log_denominator - log_numerator
        losses.append(loss_i)
    
    return torch.stack(losses).mean()

def debug_multi_positive_infonce_loss(descriptors_dict, positive_pairs, 
                                    temperature=0.07, debug=True):
    """
    Debug version of multi-positive InfoNCE loss.
    
    Args:
        descriptors_dict: dict with keys as image identifiers, values as embeddings
        positive_pairs: list of tuples, each tuple contains identifiers of positive pairs
        temperature: temperature parameter for softmax
        debug: whether to print debug information
    """
    
    if debug:
        print(f"\n🔍 DEBUG InfoNCE Loss")
        print(f"Temperature: {temperature}")
        print(f"Num descriptors: {len(descriptors_dict)}")
        print(f"Num positive pairs: {len(positive_pairs)}")
    
    # Extract embeddings
    keys = list(descriptors_dict.keys())
    embeddings = torch.stack([descriptors_dict[k] for k in keys])  # [N, D]
    
    if debug:
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings mean: {embeddings.mean().item():.6f}")
        print(f"Embeddings std: {embeddings.std().item():.6f}")
    
    # Check for collapse before normalization
    if debug:
        first_embed = embeddings[0:1]
        differences = torch.norm(embeddings - first_embed, dim=1)
        max_diff = differences.max().item()
        print(f"Max difference between embeddings: {max_diff:.6f}")
        if max_diff < 1e-6:
            print("🚨 EMBEDDINGS ARE IDENTICAL BEFORE NORMALIZATION!")
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
    
    if debug:
        print(f"After normalization - mean: {embeddings.mean().item():.6f}")
        print(f"After normalization - std: {embeddings.std().item():.6f}")
    
    # Create key to index mapping
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t()) / temperature
    
    if debug:
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Similarity matrix diagonal mean: {similarity_matrix.diag().mean().item():.6f}")
        
        # Check off-diagonal similarities
        mask = ~torch.eye(len(keys), dtype=torch.bool, device=similarity_matrix.device)
        off_diag = similarity_matrix[mask]
        print(f"Off-diagonal similarities - mean: {off_diag.mean().item():.6f}")
        print(f"Off-diagonal similarities - std: {off_diag.std().item():.6f}")
        print(f"Off-diagonal similarities - max: {off_diag.max().item():.6f}")
    
    total_loss = 0.0
    num_valid_pairs = 0
    
    for anchor_key, positive_key in positive_pairs:
        if anchor_key not in key_to_idx or positive_key not in key_to_idx:
            continue
            
        anchor_idx = key_to_idx[anchor_key]
        positive_idx = key_to_idx[positive_key]
        
        # Get similarities for this anchor
        anchor_similarities = similarity_matrix[anchor_idx]  # [N]
        
        # Create mask for positive samples (exclude self)
        positive_mask = torch.zeros_like(anchor_similarities, dtype=torch.bool)
        positive_mask[positive_idx] = True
        
        # Compute loss for this anchor
        numerator = torch.logsumexp(anchor_similarities[positive_mask], dim=0)
        denominator = torch.logsumexp(anchor_similarities, dim=0)
        loss = denominator - numerator
        
        if debug and num_valid_pairs < 3:  # Print first few losses
            print(f"Pair {num_valid_pairs}: anchor={anchor_key}, positive={positive_key}")
            print(f"  Numerator: {numerator.item():.6f}")
            print(f"  Denominator: {denominator.item():.6f}")
            print(f"  Loss: {loss.item():.6f}")
        
        total_loss += loss
        num_valid_pairs += 1
    
    if num_valid_pairs == 0:
        if debug:
            print("🚨 NO VALID PAIRS FOUND!")
        return torch.tensor(0.0, requires_grad=True)
    
    avg_loss = total_loss / num_valid_pairs
    
    if debug:
        print(f"Total valid pairs: {num_valid_pairs}")
        print(f"Average loss: {avg_loss.item():.6f}")
        
        if avg_loss.item() < 1e-6:
            print("🚨 LOSS IS ESSENTIALLY ZERO!")
            print("   This suggests embedding collapse or temperature issues")
    
    return avg_loss

# Wrapper for your existing function
def safe_multi_positive_infonce_loss_vectorized(descriptors_dict, positive_pairs, 
                                              temperature=0.07):
    """Safe wrapper that adds debugging and prevents common issues."""
    
    # Add small epsilon to temperature to prevent division issues
    temperature = max(temperature, 1e-8)
    
    # Call debug version periodically
    import random
    debug_this_call = random.random() < 0.01  # Debug 1% of calls
    
    return debug_multi_positive_infonce_loss(
        descriptors_dict, positive_pairs, temperature, debug=debug_this_call
    )


def test_loss_sanity():
    """Test if loss calculation behaves correctly"""
    N, P, D = 4, 2, 128
    
    # Test 1: Identical embeddings should give loss ≈ log(N + N*P - 1)
    anchor_emb = torch.ones(N, D)
    pos_emb = torch.ones(N, P, D)
    loss = multi_positive_infonce_loss_vectorized(anchor_emb, pos_emb)
    expected = torch.log(torch.tensor(N + N*P - 1 - P, dtype=torch.float))  # Total negatives
    print(f"Identical embeddings loss: {loss:.4f}, expected ≈ {expected:.4f}")
    
    # Test 2: Perfect separation should give very low loss
    anchor_emb = torch.randn(N, D) * 10  # Strong anchors
    pos_emb = anchor_emb.unsqueeze(1).expand(-1, P, -1) + 0.01 * torch.randn(N, P, D)  # Very similar positives
    loss = multi_positive_infonce_loss_vectorized(anchor_emb, pos_emb)
    print(f"Well-separated loss: {loss:.4f} (should be low)")



'''
============================================================================================================
InfoNCE DEBUG Mode
============================================================================================================
'''

def debug_multi_positive_infonce_loss(descriptors_dict, positive_pairs, 
                                    temperature=0.07, debug=True):
    """
    Debug version of multi-positive InfoNCE loss.
    
    Args:
        descriptors_dict: dict with keys as image identifiers, values as embeddings
        positive_pairs: list of tuples, each tuple contains identifiers of positive pairs
        temperature: temperature parameter for softmax
        debug: whether to print debug information
    """
    
    if debug:
        print(f"\n🔍 DEBUG InfoNCE Loss")
        print(f"Temperature: {temperature}")
        print(f"Num descriptors: {len(descriptors_dict)}")
        print(f"Num positive pairs: {len(positive_pairs)}")
    
    # Extract embeddings
    keys = list(descriptors_dict.keys())
    embeddings = torch.stack([descriptors_dict[k] for k in keys])  # [N, D]
    
    if debug:
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings mean: {embeddings.mean().item():.6f}")
        print(f"Embeddings std: {embeddings.std().item():.6f}")
    
    # Check for collapse before normalization
    if debug:
        first_embed = embeddings[0:1]
        differences = torch.norm(embeddings - first_embed, dim=1)
        max_diff = differences.max().item()
        print(f"Max difference between embeddings: {max_diff:.6f}")
        if max_diff < 1e-6:
            print("🚨 EMBEDDINGS ARE IDENTICAL BEFORE NORMALIZATION!")
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
    
    if debug:
        print(f"After normalization - mean: {embeddings.mean().item():.6f}")
        print(f"After normalization - std: {embeddings.std().item():.6f}")
    
    # Create key to index mapping
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t()) / temperature
    
    if debug:
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Similarity matrix diagonal mean: {similarity_matrix.diag().mean().item():.6f}")
        
        # Check off-diagonal similarities
        mask = ~torch.eye(len(keys), dtype=torch.bool, device=similarity_matrix.device)
        off_diag = similarity_matrix[mask]
        print(f"Off-diagonal similarities - mean: {off_diag.mean().item():.6f}")
        print(f"Off-diagonal similarities - std: {off_diag.std().item():.6f}")
        print(f"Off-diagonal similarities - max: {off_diag.max().item():.6f}")
    
    total_loss = 0.0
    num_valid_pairs = 0
    
    for anchor_key, positive_key in positive_pairs:
        if anchor_key not in key_to_idx or positive_key not in key_to_idx:
            continue
            
        anchor_idx = key_to_idx[anchor_key]
        positive_idx = key_to_idx[positive_key]
        
        # Get similarities for this anchor
        anchor_similarities = similarity_matrix[anchor_idx]  # [N]
        
        # Create mask for positive samples (exclude self)
        positive_mask = torch.zeros_like(anchor_similarities, dtype=torch.bool)
        positive_mask[positive_idx] = True
        
        # Compute loss for this anchor
        numerator = torch.logsumexp(anchor_similarities[positive_mask], dim=0)
        denominator = torch.logsumexp(anchor_similarities, dim=0)
        loss = denominator - numerator
        
        if debug and num_valid_pairs < 3:  # Print first few losses
            print(f"Pair {num_valid_pairs}: anchor={anchor_key}, positive={positive_key}")
            print(f"  Numerator: {numerator.item():.6f}")
            print(f"  Denominator: {denominator.item():.6f}")
            print(f"  Loss: {loss.item():.6f}")
        
        total_loss += loss
        num_valid_pairs += 1
    
    if num_valid_pairs == 0:
        if debug:
            print("🚨 NO VALID PAIRS FOUND!")
        return torch.tensor(0.0, requires_grad=True)
    
    avg_loss = total_loss / num_valid_pairs
    
    if debug:
        print(f"Total valid pairs: {num_valid_pairs}")
        print(f"Average loss: {avg_loss.item():.6f}")
        
        if avg_loss.item() < 1e-6:
            print("🚨 LOSS IS ESSENTIALLY ZERO!")
            print("   This suggests embedding collapse or temperature issues")
    
    return avg_loss

# Wrapper for your existing function
def safe_multi_positive_infonce_loss_vectorized(descriptors_dict, positive_pairs, 
                                              temperature=0.07):
    """Safe wrapper that adds debugging and prevents common issues."""
    
    # Add small epsilon to temperature to prevent division issues
    temperature = max(temperature, 1e-8)
    
    # Call debug version periodically
    import random
    debug_this_call = random.random() < 0.01  # Debug 1% of calls
    
    return debug_multi_positive_infonce_loss(
        descriptors_dict, positive_pairs, temperature, debug=debug_this_call
    )