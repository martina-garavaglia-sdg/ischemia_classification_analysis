train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:,1]
y_test = test[:,1]



Plots.plot(data_normal[1,2:end], legend = false, linecolor=:gray, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, ylim=(-4,4), xlab="Samples", ylab="mV")
for i in 2:67
    Plots.plot!(data_normal[i,2:end], legend = false, linecolor=:gray, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, ylim=(-4,4))
end


savefig("visualization/data/normal_gray.png")

# Compute mean for each series
data = train
data = vcat(data, test)
train_normal = train[train[:,1] .== -1.0,:]
test_normal = test[test[:,1] .== -1.0,:]
data_normal = vcat(train_normal, test_normal)

train_abnormal = train[train[:,1] .== 1.0,:]
test_abnormal = test[test[:,1] .== 1.0,:]
data_abnormal = vcat(train_abnormal, test_abnormal)
















means_normal = Float64[]
stds_normal = Float64[]

for i in 2:97
    push!(means_normal, mean(data_normal[:,i]))
    push!(stds_normal, std(data_normal[:,i]))
end

plot!(means_normal, lw = 2, ribbon = stds_normal, fa=0.3, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, lab = "normal heartbeats", xlab="Samples")
savefig("visualization/data/ischemia_normal_mean_stds.png")


means_abnormal = Float64[]
stds_abnormal = Float64[]
for i in 2:97
    push!(means_abnormal, mean(data_abnormal[:,i]))
    push!(stds_abnormal, std(data_abnormal[:,i]))
end

plot!(means_abnormal, lw=2, ribbon = stds_abnormal, fa=0.3, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, lab = "myocardial ischemia", xlab="Samples")
savefig("visualization/data/ischemia_abnormal_means_std.png")


plot(data_abnormal[1,2:end], legend = false)
savefig("visualization/data/p2.png")