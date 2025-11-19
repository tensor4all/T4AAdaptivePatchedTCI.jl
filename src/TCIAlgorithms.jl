module TCIAlgorithms

import T4ATensorCI as TCI
import T4ATensorCI: TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2

using TCIITensorConversion
using ITensors
using ITensorMPS: ITensorMPS
using ITensorMPS: MPO, MPS, linkdims, linkinds
using Quantics

using OrderedCollections: OrderedDict, OrderedSet
using Distributed
using EllipsisNotation
using TCIITensorConversion
import LinearAlgebra as LA

import FastMPOContractions as FMPOC

const MMultiIndex = Vector{Vector{Int}}
const TensorTrainState{T} = TensorTrain{T,3} where {T}

include("util.jl")
include("projector.jl")
include("blockstructure.jl")
include("projectable_evaluator.jl")
include("projtensortrain.jl")
include("container.jl")
include("mul.jl")
include("distribute.jl")
include("tree.jl")
include("patching.jl")
include("crossinterpolate.jl")
include("adaptivematmul.jl")

# ITensor interface
include("itensor.jl")

end
