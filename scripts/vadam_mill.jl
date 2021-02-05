using DrWatson
@quickactivate


using LinearAlgebra
using Distributions
using Base.Iterators
using Flux
using Mill


function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

function csv2mill(problem)
	x=readdlm("$(problem)/data.csv",'\t',Float32)
	bagids = readdlm("$(problem)/bagids.csv",'\t',Int)[:]
	bags = seqids2bags(bagids)
	y = readdlm("$(problem)/labels.csv",'\t',Int)
	y = map(b -> maximum(y[b]), bags)
	(samples = BagNode(ArrayNode(x), bags), labels = y)
end


data = "/home/smidl/GitHub/MIProblems/Musk1"
(x,y)=csv2mill(data)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)

# create the model
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.tanh)),                      # model on the level of Flows
    SegmentedMeanMax(10),                                       # aggregation
    ArrayModel(Chain(Dense(20, 10, Flux.tanh), Dense(10, 2))))  # model on the level of bags

# define loss function
loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

# the usual way of training
opt = Flux.ADAM()
Flux.train!(loss, params(model), repeated((x, y_oh), 1000), opt)
q=vtrain_ardvb!(loss, params(model), repeated((x, y_oh), 10000), opt)
