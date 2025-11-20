# Conversion function from ProjTensorTrain to SubDomainMPS
function SubDomainMPS(::Type{T}, projtt::ProjTensorTrain{T}, sites) where {T}
    # Convert ProjTensorTrain to TensorTrain (T4AITensorCompat)
    tt_itensor = _convert_TCI_TensorTrain_to_ITensorTensorTrain(projtt.data, sites, T)
    # Convert T4AAdaptivePatchedTCI.Projector to T4APartitionedMPSs.Projector
    partitioned_projector = _convert_projector(projtt.projector, sites)
    # Create SubDomainMPS
    return SubDomainMPS(tt_itensor, partitioned_projector)
end

# Convert T4AAdaptivePatchedTCI.Projector to T4APartitionedMPSs.Projector
function _convert_projector(proj::Projector, sites::AbstractVector{<:AbstractVector})
    proj_dict = Dict{Index,Int}()
    for (site_idx, site_vec) in enumerate(sites)
        for (leg_idx, index) in enumerate(site_vec)
            proj_val = proj.data[site_idx][leg_idx]
            if proj_val > 0
                proj_dict[index] = proj_val
            end
        end
    end
    return PartitionedProjector(proj_dict)
end

# Helper function to convert TCI.TensorTrain to T4AITensorCompat.TensorTrain
function _convert_TCI_TensorTrain_to_ITensorTensorTrain(
    tt_tci::TCI.TensorTrain{T,3}, sites::AbstractVector{<:AbstractVector}, ::Type{T}
) where {T}
    N = length(tt_tci)
    length(sites) == N || error("Length mismatch: sites ($(length(sites))) != TensorTrain ($N)")
    
    # Create link indices
    linkdims = TCI.linkdims(tt_tci)
    links = [Index(ld, "Link,l=$l") for (l, ld) in enumerate(linkdims)]
    
    # Convert each tensor core to ITensor
    tensors = ITensor[]
    for n in 1:N
        core = tt_tci[n]
        if n == 1
            # First tensor: (1, physical..., link[1])
            tensor = ITensor(core, sites[n]..., links[1])
        elseif n == N
            # Last tensor: (link[end], physical...)
            tensor = ITensor(core, links[end], sites[n]...)
        else
            # Middle tensors: (link[n-1], physical..., link[n])
            tensor = ITensor(core, links[n-1], sites[n]..., links[n])
        end
        push!(tensors, tensor)
    end
    
    return ITensorTensorTrain(tensors)
end

# Conversion Functions
# MPS(subdmps::SubDomainMPS) is already defined in T4APartitionedMPSs
# subdmps.data is already T4AITensorCompat.TensorTrain = MPS

function ProjTensorTrain{T}(subdmps::SubDomainMPS) where {T}
    # Convert SubDomainMPS.data (TensorTrain from T4AITensorCompat) to TCI.TensorTrain
    tt_tci = _convert_ITensorTensorTrain_to_TCI_TensorTrain(subdmps.data, T)
    # Convert T4APartitionedMPSs.Projector to T4AAdaptivePatchedTCI.Projector
    # This requires sites information, which we can get from siteinds
    sites = siteinds(subdmps)
    proj = _convert_partitioned_projector_to_projector(subdmps.projector, sites)
    return ProjTensorTrain{T}(tt_tci, proj)
end

# Convert T4APartitionedMPSs.Projector to T4AAdaptivePatchedTCI.Projector
function _convert_partitioned_projector_to_projector(
    part_proj::PartitionedProjector, sites::AbstractVector{<:AbstractVector}
)
    data = Vector{Vector{Int}}()
    sitedims = Vector{Vector{Int}}()
    for site_vec in sites
        proj_vec = Int[]
        dim_vec = Int[]
        for index in site_vec
            push!(dim_vec, dim(index))
            proj_val = get(part_proj.data, index, 0)
            push!(proj_vec, proj_val)
        end
        push!(data, proj_vec)
        push!(sitedims, dim_vec)
    end
    return Projector(data, sitedims)
end

function ProjTTContainer{T}(partmps::PartitionedMPS) where {T}
    projtt_vec = ProjTensorTrain{T}[]
    for subdmps in values(partmps.data)
        push!(projtt_vec, ProjTensorTrain{T}(subdmps))
    end
    return ProjTTContainer(projtt_vec)
end

# Helper function to convert T4AITensorCompat.TensorTrain to TCI.TensorTrain
function _convert_ITensorTensorTrain_to_TCI_TensorTrain(
    tt_itensor::ITensorTensorTrain, ::Type{T}
) where {T}
    # Extract sites and convert ITensors to arrays
    sites = siteinds(tt_itensor)
    tensors = Array{T,3}[]
    links = linkinds(tt_itensor)
    
    for (n, tensor) in enumerate(tt_itensor)
        site_inds = sites[n]
        # Convert ITensor to array
        if n == 1
            inds_list = vcat(site_inds, [links[1]])
            arr = Array(tensor, inds_list...)
            push!(tensors, reshape(arr, 1, :, last(size(arr))))
        elseif n == length(tt_itensor)
            inds_list = vcat([links[end]], site_inds)
            arr = Array(tensor, inds_list...)
            push!(tensors, reshape(arr, first(size(arr)), :, 1))
        else
            inds_list = vcat([links[n-1]], site_inds, [links[n]])
            arr = Array(tensor, inds_list...)
            push!(tensors, reshape(arr, first(size(arr)), :, last(size(arr))))
        end
    end
    
    return TCI.TensorTrain{T,3}(tensors)
end

# Utility Functions
function permutesiteinds(Ψ::MPS, sites::AbstractVector{<:AbstractVector})
    links = linkinds(Ψ)
    tensors = Vector{ITensor}(undef, length(Ψ))
    tensors[1] = permute(Ψ[1], vcat(sites[1], links[1]))
    for n in 2:(length(Ψ) - 1)
        tensors[n] = permute(Ψ[n], vcat(links[n - 1], sites[n], links[n]))
    end
    tensors[end] = permute(Ψ[end], vcat(links[end], sites[end]))
    return MPS(tensors)  # T4AITensorCompat.MPS = TensorTrain
end

function project(tensor::ITensor, projsiteinds::Dict{K,Int}) where {K}
    slice = Union{Int,Colon}[
        idx ∈ keys(projsiteinds) ? projsiteinds[idx] : Colon() for idx in inds(tensor)
    ]
    data_org = Array(tensor, inds(tensor)...)
    data_trim = zero(data_org)
    data_trim[slice...] .= data_org[slice...]
    return ITensor(data_trim, inds(tensor)...)
end

function project(oldprojector::Projector, sites, projsiteinds::Dict{Index{T},Int}) where {T}
    newprojdata = deepcopy(oldprojector.data)
    for (siteind, projind) in projsiteinds
        pos = find_nested_index(sites, siteind)
        if pos === nothing
            error("Site index not found: $siteind")
        end
        newprojdata[pos[1]][pos[2]] = projind
    end
    return Projector(newprojdata, oldprojector.sitedims)
end

function project(subdmps::SubDomainMPS, projsiteinds::Dict{Index{T},Int}) where {T}
    # Convert Dict to T4APartitionedMPSs.Projector
    projector_data = Dict{Index,Int}(projsiteinds)
    new_projector = PartitionedProjector(projector_data)
    # Use SubDomainMPS project method (from T4APartitionedMPSs)
    result = T4APartitionedMPSs.project(subdmps, new_projector)
    if result === nothing
        error("Projection resulted in nothing - projectors may not overlap")
    end
    return result
end

function asTT3(::Type{T}, Ψ::MPS, sites; permdims=true)::TensorTrain{T,3} where {T}
    Ψ2 = permdims ? permutesiteinds(Ψ, sites) : Ψ
    tensors = Array{T,3}[]
    links = linkinds(Ψ2)
    push!(tensors, reshape(Array(Ψ2[1], sites[1]..., links[1]), 1, :, dim(links[1])))
    for n in 2:(length(Ψ2) - 1)
        push!(
            tensors,
            reshape(
                Array(Ψ2[n], links[n - 1], sites[n]..., links[n]),
                dim(links[n - 1]),
                :,
                dim(links[n]),
            ),
        )
    end
    push!(
        tensors, reshape(Array(Ψ2[end], links[end], sites[end]...), dim(links[end]), :, 1)
    )
    return TensorTrain{T,3}(tensors)
end

function _check_projector_compatibility(projector::Projector, subdmps::SubDomainMPS)
    # SubDomainMPS already checks compatibility in constructor
    # This is a compatibility function for the old API
    return true
end

function find_nested_index(data::Vector{Vector{T}}, target::T) where {T}
    for (i, subvector) in enumerate(data)
        j = findfirst(x -> x == target, subvector)
        if j !== nothing
            return (i, j)
        end
    end
    return nothing  # Not found
end

# T4AQuantics extension functions are already defined in T4APartitionedMPSs and T4AQuantics
# No need to redefine them - they should work automatically

# Miscellaneous Functions
# Base.show, ITensors.prime, Base.isapprox are already defined in T4APartitionedMPSs
# We can add type aliases for backward compatibility if needed

function ITensors.prime(Ψ::PartitionedMPS, args...; kwargs...)
    return T4APartitionedMPSs.prime(Ψ, args...; kwargs...)
end

# Base.isapprox for SubDomainMPS is already defined in T4APartitionedMPSs

# Make PartitionedMPS work as ProjectableEvaluator for evaluation
# Support both MultiIndex (Vector{Int}) and MMultiIndex (Vector{Vector{Int}})
function (obj::PartitionedMPS)(multiidx::MultiIndex)
    # Convert MultiIndex to MMultiIndex
    sitedims_ = sitedims(obj)
    mmultiidx = multii(sitedims_, multiidx)
    return obj(mmultiidx)
end

function (obj::PartitionedMPS)(mmultiidx::MMultiIndex)
    # Sum over all SubDomainMPS in the PartitionedMPS
    # Each SubDomainMPS corresponds to a different projector
    if isempty(obj.data)
        error("Cannot evaluate empty PartitionedMPS")
    end
    # Get element type from first tensor in first SubDomainMPS
    first_subdmps = first(values(obj.data))
    first_tensor = first(first_subdmps.data)
    # Get element type from ITensor's storage
    T = eltype(ITensors.storage(first_tensor))
    result = zero(T)
    for subdmps in values(obj.data)
        # Convert SubDomainMPS to ProjTensorTrain for evaluation
        # ProjTensorTrain{T}(subdmps) already gets sites from siteinds(subdmps)
        ptt = ProjTensorTrain{T}(subdmps)
        result += ptt(mmultiidx)
    end
    return result
end

# Add fulltensor method for PartitionedMPS
function fulltensor(obj::PartitionedMPS; fused::Bool=false)
    # Sum over all SubDomainMPS in the PartitionedMPS
    if isempty(obj.data)
        error("Cannot compute fulltensor for empty PartitionedMPS")
    end
    result = nothing
    # Get element type from first tensor in first SubDomainMPS
    first_subdmps = first(values(obj.data))
    first_tensor = first(first_subdmps.data)
    # Get element type from ITensor's storage
    T = eltype(ITensors.storage(first_tensor))
    for subdmps in values(obj.data)
        ptt = ProjTensorTrain{T}(subdmps)
        if result === nothing
            result = fulltensor(ptt; fused=fused)
        else
            result .+= fulltensor(ptt; fused=fused)
        end
    end
    return result
end

# Add sitedims property access for PartitionedMPS (needed for ProjectableEvaluator compatibility)
function sitedims(obj::PartitionedMPS)
    if isempty(obj.data)
        error("Cannot get sitedims for empty PartitionedMPS")
    end
    # Get sitedims from first SubDomainMPS
    first_subdmps = first(values(obj.data))
    sites = siteinds(first_subdmps)
    return [collect(dim.(s)) for s in sites]
end
