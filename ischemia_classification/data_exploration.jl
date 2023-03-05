train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:,1]
y_test = test[:,1]



plot(data_abnormal[1,2:end], legend = false)
for i in 2:133
    plot!(data_abnormal[i,2:end], legend = false)
end


savefig("visualization/data/abnormal.png")

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

plot(means_normal, lw = 2, ribbon = stds_normal, fa=0.3, lab = "normal heartbeats")
#savefig("visualization/data/mean_normal.png")


means_abnormal = Float64[]
stds_abnormal = Float64[]
for i in 2:97
    push!(means_abnormal, mean(data_abnormal[:,i]))
    push!(stds_abnormal, std(data_abnormal[:,i]))
end

plot!(means_abnormal, lw=2, ribbon = stds_abnormal, fa=0.3, lab = "myocardial ischemia")
savefig("visualization/data/ischemia_means_std.png")


plot(data_abnormal[1,2:end], legend = false)
savefig("visualization/data/p2.png")