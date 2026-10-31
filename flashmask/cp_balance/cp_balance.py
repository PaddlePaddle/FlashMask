# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

import heapq
import numpy as np
from .cp_balance_cuda_kernels import scanMaxMinChunkedKernel, reduce_workload, indices_to_chunks_cuda, indices_rerank_cuda, cp_balance_ipo_solve
from ..backend import Tensor, get_backend
from ..distributed import get_dist
from typing import List, Tuple, Dict, Optional

def get_q_workload(
    start_row_indices: Tensor,
    q_chunk_size: int,
    m_block_size: int,
    n_block_size: int
) -> Tensor:
    """
    根据稀疏attention的起止索引，估算每个query chunk的计算负载。
    这是负载均衡的第一步，目的是量化每个数据块的计算成本。

    Args:
        start_row_indices (Tensor): 形状为 [B, H, S, 2] 或 [B, H, S, 4] 的张量，
                                           表示每个 query token 需要计算的 key token 的起止范围。
                                           维度4的顺序为 [LTS, LTE, UTS, UTE]。
                                           维度2的顺序为 [LTS, UTE]。
        q_chunk_size (int): Query 侧进行负载均衡分析的块大小。
        m_block_size (int): FlashAttention kernel 中 query 侧的块大小 (Br)。
        n_block_size (int): FlashAttention kernel 中 key 侧的块大小 (Bc)。

    Returns:
        Tensor: 形状为 [1, H, Tchunks, 2] 的张量，
                       其中 Tchunks 是 chunk 的数量。
                       每个 chunk 的信息为 [workload, original_index]，
                       表示该 chunk 的估算工作量和其原始索引。
    """
    be = get_backend(start_row_indices)
    assert start_row_indices is not None, "start_row_indices cannot be None"
    assert q_chunk_size % m_block_size == 0, "q_chunk_size must be divisible by m_block_size"

    # 1. 解析输入的起止索引
    # start_row_indices 可能包含下三角(LT)和上三角(UT)的起止(Start/End)信息
    LTS, LTE, UTS, UTE = None, None, None, None
    if start_row_indices.shape[-1] == 4:
        LTS, LTE, UTS, UTE = be.split(start_row_indices, 4, axis=-1)
        LTS, LTE, UTS, UTE = [t.squeeze(-1) for t in (LTS, LTE, UTS, UTE)]
    elif start_row_indices.shape[-1] == 2:
        LTS, UTE = be.split(start_row_indices, 2, axis=-1)
        LTS, UTE = LTS.squeeze(-1), UTE.squeeze(-1)

    # 2. 获取维度信息
    # 从任意一个非None的张量中获取 Batch, Head, Sequence Length
    valid_tensor = next(t for t in [LTS, LTE, UTS, UTE] if t is not None)
    B, H, S = valid_tensor.shape
    
    # 计算块的数量
    Tr = S // m_block_size  # Query 侧块总数
    Tc = S // n_block_size  # Key 侧块总数
    Tchunks = S // q_chunk_size # 用于负载均衡的 chunk 总数
    assert Tr % Tchunks == 0, "Total row blocks must be divisible by total chunks"
    blocks_per_chunk = Tr // Tchunks

    # 3. 使用自定义CUDA核预计算每个 Key 块内的索引最大/最小值
    # 这一步是关键优化，它将 O(S) 的扫描操作降维到 O(S/Bc)，
    # 极大地加速了后续工作负载的估算。
    def scan_max_min(tensor):
        if tensor is not None:
            return scanMaxMinChunkedKernel(tensor, n_block_size, B, H, S)
        return None, None

    LTStartMax_gpu, LTStartMin_gpu = scan_max_min(LTS)
    LTEndMax_gpu, LTEndMin_gpu = scan_max_min(LTE)
    UTStartMax_gpu, UTStartMin_gpu = scan_max_min(UTS)
    UTEndMax_gpu, UTEndMin_gpu = scan_max_min(UTE)

    # 4. 使用自定义CUDA核计算每个 Query 块的工作负载
    # 这个核模拟了 FlashAttention 的块状计算过程，但只计算需要被激活的块的数量，
    # 而不是执行实际的矩阵乘法，从而高效地估算出工作负载。
    all_indices_max_min = [
        LTStartMax_gpu, LTStartMin_gpu, LTEndMax_gpu, LTEndMin_gpu,
        UTStartMax_gpu, UTStartMin_gpu, UTEndMax_gpu, UTEndMin_gpu
    ]
    workload_per_block = reduce_workload(all_indices_max_min, B, H, Tr, Tc, m_block_size, S)

    # 5. 将每个块的工作负载聚合到 chunk 级别
    workload_grouped = workload_per_block.reshape([B, H, Tchunks, blocks_per_chunk, 1])
    workload_per_chunk = be.sum(workload_grouped, axis=3).sum(axis=0).reshape([1, H, Tchunks])

    # 6. 准备最终输出，包含工作负载和原始索引
    final_res = be.zeros([1, H, Tchunks, 2], dtype='int32')
    final_res[:, :, :, 0] = workload_per_chunk
    final_res[:, :, :, 1] = be.arange(0, Tchunks, dtype="int32")
    
    return final_res


def assign_tasks_heap(
    tasks: np.ndarray, 
    num_buckets: int
) -> Tuple[List[List[Tuple[int, int]]], List[int], int]:
    """
    使用小顶堆的贪心算法，将带有权重和索引的任务列表分配到 M 个桶中，
    以实现负载均衡。

    Args:
        tasks (np.ndarray): 形状为 (N, 2) 的任务数组，每行是 [weight, index]。
        num_buckets (int): 桶的数量（通常等于通信组的 world size）。

    Returns:
        Tuple:
            - buckets (List[List[Tuple[int, int]]]): 分配结果，每个子列表是一个桶的任务。
            - bucket_weights (List[int]): 每个桶的总权重。
            - cuts (int): 数据切分次数，衡量数据重排后的连续性。
    """
    n = len(tasks)
    if n == 0:
        return [[] for _ in range(num_buckets)], [0] * num_buckets, 0
    
    # 每个桶的期望任务数量
    batch_size = n // num_buckets

    # 按权重降序排序任务，优先分配最重的任务
    tasks_sorted = sorted(tasks, key=lambda x: -x[0])

    # 初始化桶和记录每个桶当前状态的变量
    buckets = [[] for _ in range(num_buckets)]
    bucket_weights = [0] * num_buckets
    bucket_counts = [0] * num_buckets

    # 初始化小顶堆，用于快速找到当前总权重最小的桶
    # 堆中元素为 (current_weight, bucket_index)
    heap = [(0, i) for i in range(num_buckets)]

    # 贪心分配：依次将最重的任务分配给当前总权重最小的、且未满的桶
    for weight, idx in tasks_sorted:
        # 找到一个可以放入任务的桶
        temp_popped = []
        found_bucket = False
        while heap:
            bucket_sum, bucket_idx = heapq.heappop(heap)
            if bucket_counts[bucket_idx] < batch_size:
                # 找到桶，更新状态并放回堆中
                buckets[bucket_idx].append((weight, idx))
                bucket_weights[bucket_idx] += weight
                bucket_counts[bucket_idx] += 1
                heapq.heappush(heap, (bucket_weights[bucket_idx], bucket_idx))
                found_bucket = True
                break
            else:
                # 该桶已满，暂存起来，继续寻找下一个
                temp_popped.append((bucket_sum, bucket_idx))
        
        # 将之前因为满了而弹出的桶重新放回堆中
        for item in temp_popped:
            heapq.heappush(heap, item)
            
        if not found_bucket:
            # 如果所有桶都满了（通常在 n % num_buckets != 0 时发生）
            # 将剩余的任务分配给当前总权重最小的桶
            bucket_sum, bucket_idx = heapq.heappop(heap)
            buckets[bucket_idx].append((weight, idx))
            bucket_weights[bucket_idx] += weight
            bucket_counts[bucket_idx] += 1
            heapq.heappush(heap, (bucket_weights[bucket_idx], bucket_idx))


    # （可选）按任务原始序号对每个桶内部进行排序，方便调试
    for i in range(num_buckets):
        buckets[i] = sorted(buckets[i], key=lambda x: x[1])

    # 统计切分次数：衡量重排后数据块的连续性
    all_assigned_indices = sorted([idx for bucket in buckets for _, idx in bucket])
    cuts = sum(1 for i in range(1, len(all_assigned_indices)) if all_assigned_indices[i] != all_assigned_indices[i-1] + 1)

    return buckets, bucket_weights, cuts


def assign_tasks_ipo(
    tasks: np.ndarray,
    num_buckets: int
) -> Tuple[List[List[Tuple[int, int]]], List[int], int]:
    """
    使用 IPO (Iterative Pairwise/Triple Optimal) 最优求解器分配任务。
    接口与 assign_tasks_heap 完全一致。

    当 N > 512 或 N % num_buckets != 0 时自动 fallback 到 assign_tasks_heap。

    Args:
        tasks (np.ndarray): 形状为 (N, 2) 的任务数组，每行是 [weight, index]。
        num_buckets (int): 桶的数量。

    Returns:
        与 assign_tasks_heap 相同的三元组 (buckets, bucket_weights, cuts)。
    """
    # 兼容 Paddle tensor 输入（与 assign_tasks_heap 行为对齐）
    if not isinstance(tasks, np.ndarray):
        tasks = tasks.cpu().numpy() if hasattr(tasks, 'cpu') else np.asarray(tasks)

    n = len(tasks)
    if n == 0 or n > 512 or n % num_buckets != 0:
        return assign_tasks_heap(tasks, num_buckets)

    K = n // num_buckets
    weights = tasks[:, 0].astype(np.int32)

    # 调用 C++ IPO solver
    # assign_matrix: (num_buckets, K)，每个元素是 item index (0..N-1)
    assign_matrix, _ = cp_balance_ipo_solve(weights, num_buckets)

    buckets = []
    bucket_weights = []
    for j in range(num_buckets):
        bucket = []
        bw = 0
        for t in range(K):
            idx = int(assign_matrix[j, t])
            w = int(tasks[idx][0])
            chunk_idx = int(tasks[idx][1])
            bucket.append((w, chunk_idx))
            bw += w
        bucket.sort(key=lambda x: x[1])
        buckets.append(bucket)
        bucket_weights.append(bw)

    # 统计切分次数
    all_idx = sorted([idx for b in buckets for _, idx in b])
    cuts = sum(1 for i in range(1, len(all_idx)) if all_idx[i] != all_idx[i - 1] + 1)

    return buckets, bucket_weights, cuts


# --- 数据通信与重排辅助函数 ---

def get_send_dict(buckets: List[List[Tuple[int, int]]], cp_size: int, rank: int) -> Dict[int, List[int]]:
    """
    根据负载均衡分配结果，为当前 rank 生成 all-to-all 通信的发送字典。

    Args:
        buckets (List): 所有 rank 的任务分配结果。
        cp_size (int): 通信组大小。
        rank (int): 当前进程的 rank。

    Returns:
        Dict[int, List[int]]: 发送字典。key 是目标 rank，value 是要发送给该 rank 的本地 chunk 索引列表。
    """
    send_dict = {i: [] for i in range(cp_size)}
    # 遍历所有桶（即所有目标 rank 的任务列表）
    for target_rank, bucket in enumerate(buckets):
        for _, chunk_idx in bucket:
            # 如果某个 chunk 的原始属主是当前 rank，则需要将其发送
            if chunk_idx // cp_size == rank:
                # chunk_idx % cp_size 得到的是在当前 rank 上的局部索引
                send_dict[target_rank].append(chunk_idx % cp_size)
    return send_dict

def get_recv_dict(bucket: List[Tuple[int, int]], cp_size: int) -> Dict[int, List[int]]:
    """
    根据当前 rank 的任务分配结果，生成 all-to-all 通信的接收字典。

    Args:
        bucket (List): 当前 rank 分配到的任务列表。
        cp_size (int): 通信组大小。

    Returns:
        Dict[int, List[int]]: 接收字典。key 是源 rank，value 是从该 rank 接收的数据块
                               应该被放置到的本地位置索引列表。
    """
    recv_dict = {i: [] for i in range(cp_size)}
    # 遍历分配给我的所有任务
    for local_pos, (_, chunk_idx) in enumerate(bucket):
        # chunk_idx.item() // cp_size 得到的是这个 chunk 原始所在的 rank
        source_rank = chunk_idx.item() // cp_size
        recv_dict[source_rank].append(local_pos)
    return recv_dict

def balance_alltoall(
    input_tensor: Tensor,
    cp_size: int,
    cp_group,
    chunk_size: int,
    send_dict: Dict[int, List[int]],
    recv_dict: Dict[int, List[int]]
) -> Tensor:
    """
    执行 all-to-all 通信，根据 send/recv 字典对 `input_tensor` 进行数据重排。
    此函数已重构，可统一处理不同维度的张量。

    Args:
        input_tensor (Tensor): 待重排的张量，如 Q, K, V。
        cp_size (int): 通信组大小。
        cp_group (dist.Group): Paddle 分布式通信组。
        chunk_size (int): 数据块的大小。
        send_dict (Dict): 发送字典。
        recv_dict (Dict): 接收字典。

    Returns:
        Tensor: 重排后的张量。
    """
    be = get_backend(input_tensor)
    dist = get_dist(be.name)
    original_shape = input_tensor.shape
    B, S = original_shape[0], original_shape[1]
    
    # 将输入张量统一 reshape 为 3D (B, S, -1) 以便统一处理
    tensor_3d = input_tensor.reshape((B, S, -1))
    HD = tensor_3d.shape[-1]
    
    # 1. 准备发送数据 (Gather)
    # 根据 send_dict，从本地张量中收集需要发送给其他 rank 的数据块
    send_list = []
    for target_rank in range(cp_size):
        indices_to_send = send_dict[target_rank]
        if indices_to_send:
            # 将所有要发往同一个 rank 的数据块拼接在一起
            data_to_send = be.concat(
                [tensor_3d[:, idx * chunk_size:(idx + 1) * chunk_size, :] for idx in indices_to_send],
                axis=1
            )
            send_list.append(data_to_send)
        else:
            # 注意：NCCL alltoall 不支持大小为 0 的张量，因此发送一个虚拟的、
            # 非常小的张量作为占位符。接收方也需对应接收。
            send_list.append(be.zeros((B, 1, HD), dtype=input_tensor.dtype))

    # 2. 准备接收缓冲区 (Scatter)
    # 根据 recv_dict，为从其他 rank 接收的数据准备相应大小的空缓冲区
    recv_list = []
    for source_rank in range(cp_size):
        num_chunks_to_recv = len(recv_dict[source_rank])
        if num_chunks_to_recv > 0:
            recv_list.append(
                be.empty((B, chunk_size * num_chunks_to_recv, HD), dtype=input_tensor.dtype)
            )
        else:
            # 对应发送方的虚拟张量，接收一个同样大小的虚拟缓冲区
            recv_list.append(be.empty((B, 1, HD), dtype=input_tensor.dtype))
            
    # 3. 执行 All-to-All 通信
    dist.all_to_all(recv_list, send_list, group=cp_group)

    # 4. 将接收到的数据重新组装成最终张量
    final_res_3d = be.empty_like(tensor_3d)
    for source_rank in range(cp_size):
        local_positions = recv_dict[source_rank]
        if local_positions:
            received_data = recv_list[source_rank]
            # 将从 source_rank 接收到的数据块，放置到它们在本地应该在的位置
            for i, local_pos in enumerate(local_positions):
                start_s = local_pos * chunk_size
                end_s = (local_pos + 1) * chunk_size
                data_start = i * chunk_size
                data_end = (i + 1) * chunk_size
                final_res_3d[:, start_s:end_s, :] = received_data[:, data_start:data_end, :]
    
    # 恢复原始形状
    return final_res_3d.reshape(original_shape)


# --- 主流程函数 ---

def balance_flashmask_input(
    startend_row_indices: Tensor,
    cp_size: int,
    cp_rank: int,
    balance_chunk_size: int = 2048,
    q_block_size: int = 128,
    k_block_size: int = 128,
    use_ipo: bool = False
) -> Tuple[Tensor, List[List[Tuple[int, int]]]]:
    """
    FlashMask 输入数据的负载均衡主流程。
    该函数协调整个过程：估算工作负载 -> 任务分配 -> 生成通信计划 -> 数据重排。

    Args:
        startend_row_indices (Tensor): 稀疏 attention 的原始起止索引。
        cp_size (int): 通信组大小。
        cp_rank (int): 当前进程的 rank。
        balance_chunk_size (int): 用于负载均衡分析和数据移动的块大小。
        q_block_size (int): FlashAttention kernel 的 query 块大小。
        k_block_size (int): FlashAttention kernel 的 key 块大小。
        use_ipo (bool): 是否使用 IPO 最优求解器替代 LPT 贪心。
                        N > 512 或 N % cp_size != 0 时自动 fallback 到 LPT。

    Returns:
        Tuple:
            - local_startend_row_indices (Tensor): 经过负载均衡和重排后，
              当前 rank 需要处理的局部起止索引。
            - buckets (List[List[Tuple[int, int]]]): 全局的任务分配方案，用于后续
              对 Q, K, V 等张量进行同样的重排。
    """
    be = get_backend(startend_row_indices)
    dist = get_dist(be.name)
    # 步骤 1: 估算每个 chunk 的工作负载
    dist.nvtx_range_push("get_q_workload")
    workload = get_q_workload(startend_row_indices, balance_chunk_size, q_block_size, k_block_size)
    dist.nvtx_range_pop()

    # 步骤 2: 任务分配（IPO 最优求解 或 LPT 贪心）
    dist.nvtx_range_push("assign_tasks")
    # 将 workload tensor 转换成 numpy 数组
    tasks_np = workload.reshape([-1, 2]).cpu().numpy()
    if use_ipo:
        buckets, _, _ = assign_tasks_ipo(tasks_np, cp_size)
    else:
        buckets, _, _ = assign_tasks_heap(tasks_np, cp_size)
    dist.nvtx_range_pop()

    # 步骤 5: 根据全局分配方案 `buckets`，对原始索引张量进行重排 (Gather)
    # 这一步创建了一个全局视角下、数据块被重新排列后的 `startend_row_indices`。
    dist.nvtx_range_push("startend_row_indices_rerank")
    # 将 `buckets` 展平，得到一个新的 chunk 顺序
    rerank_indices = np.array([idx for bucket in buckets for _, idx in bucket], dtype=np.int32)
    indices_tensor = be.to_tensor(rerank_indices, dtype='int32')
    
    # 使用 CUDA 核高效地执行 gather 操作
    startend_row_indices_rerank = indices_rerank_cuda(startend_row_indices, indices_tensor)
    dist.nvtx_range_pop()

    # 步骤 6: 从重排后的全局索引中，计算出当前 rank 的局部索引 (Localize)
    # 这一步将全局索引（可能跨越整个序列长度S）转换为相对于本地数据块的局部索引。
    dist.nvtx_range_push("indices_to_chunks")
    local_bucket_indices = [x[1] for x in buckets[cp_rank]]
    local_indices_tensor = be.to_tensor(local_bucket_indices, dtype='int32')
    
    # 使用 CUDA 核高效地执行索引的 clipping 和 offsetting
    local_startend_row_indices = indices_to_chunks_cuda(
        startend_row_indices_rerank, local_indices_tensor, balance_chunk_size
    )
    dist.nvtx_range_pop()

    return local_startend_row_indices, buckets