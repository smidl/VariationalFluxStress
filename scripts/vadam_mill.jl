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

function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); 
	seed=nothing, contamination::Real=0.0)

	# split the normal data, add some anomalies to the train set and divide
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	nd = ndims(data_normal.data.data) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:length(data_normal.bags.bags)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	indices_anomalous = 1:length(data_anomalous.bags.bags)
	na_tr = floor(Int, length(split_inds[1])*contamination/(1-contamination))
	tr = na_tr/length(indices_anomalous) # training ratio
    vtr = (1 - tr) # validation/test ratio
    rat_v = ratios[2]/(ratios[2]+ratios[3])
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (tr, vtr*rat_v, vtr*(1-rat_v)); seed=seed)

	tr_n, val_n, tst_n = map(is -> data_normal[is], split_inds)
	tr_a, val_a, tst_a = map(is -> data_anomalous[is], split_inds_anomalous)

	# cat it together
	tr_x = cat(tr_n, tr_a, dims = nd)
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)

	# now create labels
	tr_y = vcat(zeros(Float32, length(tr_n)), ones(Float32, length(tr_a)))
	val_y = vcat(zeros(Float32, length(val_n)), ones(Float32, length(val_a)))
	tst_y = vcat(zeros(Float32, length(tst_n)), ones(Float32, length(tst_a)))

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end

import Base.length
Base.length(B::BagNode)=length(B.bags.bags)


function train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)
    (sum(ratios) == 1 && length(ratios) == 3) ? nothing :
    	error("ratios must be a vector of length 3 that sums up to 1")

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # set number of samples in individual subsets
    n = length(indices)
    ns = cumsum([x for x in floor.(Int, n .* ratios)])

    # scramble indices
    _indices = sample(indices, n, replace=false)

    # restart seed
    (seed == nothing) ? nothing : Random.seed!()

    # return the sets of indices
    _indices[1:ns[1]], _indices[ns[1]+1:ns[2]], _indices[ns[2]+1:ns[3]]
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

data=train_val_test_split(x[y.==0],x[y.==1],(0.8,0.0,0.2);contamination=0.5)
# the usual way of training
opt = Flux.ADAM()
xt = data[1][1];
yt = Flux.onehotbatch((data[1][2].+1)[:],1:2)
Flux.train!(loss, params(model), repeated((xt, yt), 5000), opt)
xte = data[3][1];
yte = Flux.onehotbatch((data[3][2].+1)[:],1:2)

q=vtrain_ardvb!(loss, params(model), repeated((xt, yt), 10000), opt)
