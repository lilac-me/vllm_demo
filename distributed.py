import torch


_TP: GroupCoordinator | None = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


# kept for backward compatibility
get_context_model_parallel_group = get_dcp_group

_PP: GroupCoordinator | None = None


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


_DP: GroupCoordinator | None = None


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


_EP: GroupCoordinator | None = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, "expert parallel group is not initialized"
    return _EP


_PCP: GroupCoordinator | None = None


def get_pcp_group() -> GroupCoordinator:
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP

_DCP: GroupCoordinator | None = None


def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None, "decode context model parallel group is not initialized"
    return _DCP


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
        backend: name of torch distributed communication backend.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    data_parallel_size = 1
    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    if config is not None:
        data_parallel_size = config.parallel_config.data_parallel_size

    # the layout order is: ExternalDP x DP x PP x TP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    # DP is the data parallel group that is part of the model,
    # all the ranks in the same DP group should generate simultaneously,
    # i.e. the `generate` call in the same DP group should be called together,
    # otherwise it will cause deadlock.
    # to get group_ranks for each dimension, transpose that dimension to the
    # last dimension, then reshape to 2D, then unbind the last dimension
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )  # noqa

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # Build the DCP model-parallel groups.
    global _DCP
    assert _DCP is None, "decode context model parallel group is already initialized"
    # Note(hc): In the current implementation of decode context parallel,
    # dcp_size must not exceed tp_size, because the world size does not
    # change by DCP, it simply reuses the GPUs of TP group, and split one
    # TP group into tp_size//dcp_size DCP groups.
    group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DCP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="dcp",
    )

    global _PCP
    assert _PCP is None, "prefill context parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _PCP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pcp"
    )

    # Build the pipeline model-parallel groups.
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

    global _DP
    assert _DP is None, "data parallel group is already initialized"
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="dp"
    )

    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            data_parallel_size
            * prefill_context_model_parallel_size
            * tensor_model_parallel_size,
        )
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _EP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="ep"
    )

    logger.info_once(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, PCP rank %s, "
        "TP rank %s, EP rank %s",
        rank,
        world_size,
        _DP.rank_in_group,
        _PP.rank_in_group,
        _PCP.rank_in_group,
        _TP.rank_in_group,
        _EP.rank_in_group,
    )
