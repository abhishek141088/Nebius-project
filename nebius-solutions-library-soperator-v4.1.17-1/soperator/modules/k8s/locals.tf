locals {
  node_ssh_access = {
    enabled = length(var.node_ssh_access_users) > 0
  }

  node_cloud_init = {
    enabled = length(var.node_ssh_access_users) > 0 || length(var.nvidia_config_lines) > 0
    cloud_init_data = templatefile("${path.module}/templates/cloud_init.yaml.tftpl", {
      ssh_users                  = var.node_ssh_access_users
      nvidia_config_lines        = var.nvidia_config_lines
      local_nvme_enabled         = false
      local_nvme_mount_path      = "/mnt/local-nvme"
      local_nvme_filesystem_type = "ext4"
    })
    cloud_init_data_no_nvidia = templatefile("${path.module}/templates/cloud_init.yaml.tftpl", {
      ssh_users                  = var.node_ssh_access_users
      nvidia_config_lines        = []
      local_nvme_enabled         = false
      local_nvme_mount_path      = "/mnt/local-nvme"
      local_nvme_filesystem_type = "ext4"
    })
  }

  node_group_gpu_present = {
    worker = [
      for worker in var.node_group_workers :
      (module.resources.by_platform[worker.resource.platform][worker.resource.preset].gpus > 0 ? true : false)
    ]
  }

  node_group_gpu_cluster_compatible = {
    worker = [for worker in var.node_group_workers :
      module.resources.by_platform[worker.resource.platform][worker.resource.preset].gpu_cluster_compatible
    ]
  }

  node_group_workload_label = {
    worker = [for worker in local.node_group_gpu_present.worker :
      (worker ? module.labels.label_workload_gpu : module.labels.label_workload_cpu)
    ]
  }

  # V2 workers (for nodesets)
  node_group_gpu_present_v2 = {
    worker = [
      for worker in var.node_group_workers_v2 :
      (module.resources.by_platform[worker.resource.platform][worker.resource.preset].gpus > 0 ? true : false)
    ]
  }

  node_group_gpu_cluster_compatible_v2 = {
    worker = [for worker in var.node_group_workers_v2 :
      module.resources.by_platform[worker.resource.platform][worker.resource.preset].gpu_cluster_compatible
    ]
  }

  node_group_workload_label_v2 = {
    worker = [for worker in local.node_group_gpu_present_v2.worker :
      (worker ? module.labels.label_workload_gpu : module.labels.label_workload_cpu)
    ]
  }

  # GPU presets that don't support a gpu_cluster block (gpu_cluster_compatible == false,
  # see gpu_fabric_validation.tf) have no InfiniBand fabric, so Nebius never attaches the
  # fabric-derived node labels that Soperator's topology reconciler reads
  # (topologyconf.slurm.nebius.ai/tier-1). Without that label the reconciler's own
  # node predicate drops the node entirely, so the topology-node-labels ConfigMap it
  # maintains is never populated for these nodes - workers then hang forever waiting
  # for topology data that was never going to arrive.
  #
  # Fix: set a static tier-1 label ourselves on exactly these nodes, so the stock
  # reconciler (unmodified) does the rest. "unknown" matches the single static switch
  # name the chart already renders into topology.conf for non-fabric nodes
  # (SwitchName=unknown), so the dynamically-registered topology stays consistent with
  # the statically-rendered one.
  node_group_topology_fallback_label_v2 = {
    worker = [
      for i, worker in var.node_group_workers_v2 :
      (local.node_group_gpu_present_v2.worker[i] && !local.node_group_gpu_cluster_compatible_v2.worker[i]
        ? { "topologyconf.slurm.nebius.ai/tier-1" = "unknown" }
        : {}
      )
    ]
  }

  # Normalize optional NVLink IDs to strings. "" means the worker should not get
  # an nvlink block or NVLink label.
  node_group_nvl_instance_group_id_v2 = {
    worker = [
      for worker in var.node_group_workers_v2 :
      try(trimspace(worker.nvl_instance_group_id), "")
    ]
  }

  # Convert NVLink IDs into label maps that can be merged unconditionally.
  # Example: ["nvl-1", ""] becomes
  # [{ "nebius.com/nvlink-instance-group" = "nvl-1" }, {}].
  node_group_nvl_instance_group_label_v2 = {
    worker = [
      for nvl_instance_group_id in local.node_group_nvl_instance_group_id_v2.worker :
      nvl_instance_group_id != "" ? tomap({
        (module.labels.key_nebius_nvlink_instance_group) = nvl_instance_group_id
      }) : tomap({})
    ]
  }

  # Normalize optional placement-policy node lists. [] means the provider
  # placement_policy block should be omitted for that worker.
  node_group_placement_policy_nodes_v2 = {
    worker = [
      for worker in var.node_group_workers_v2 :
      coalesce(try(worker.placement_policy_nodes, null), [])
    ]
  }

  context_name = join(
    "-",
    [
      "nebius",
      replace(lower(var.company_name), " ", "-"),
      "slurm"
    ]
  )
}
