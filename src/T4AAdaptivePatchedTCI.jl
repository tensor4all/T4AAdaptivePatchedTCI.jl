module T4AAdaptivePatchedTCI

import T4ATensorCI as TCI
import T4ATensorCI: TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2
import T4APartitionedTT: SubDomainTT, PartitionedTT, Projector as PartitionedProjector
using T4APartitionedTT: T4APartitionedTT

using OrderedCollections: OrderedDict, OrderedSet
using Distributed
using EllipsisNotation
import LinearAlgebra as LA

const MMultiIndex = Vector{Vector{Int}}
const TensorTrainState{T} = TensorTrain{T,3} where {T}

include("util.jl")
include("projector.jl")
include("blockstructure.jl")
include("projectable_evaluator.jl")
include("projtensortrain.jl")
include("container.jl")
include("distribute.jl")
include("tree.jl")
include("patching.jl")
include("crossinterpolate.jl")

end
