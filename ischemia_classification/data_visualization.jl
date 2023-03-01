train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:,1]
y_test = test[:,1]



plot(train[2,2:end,:], legend = false)
for i in 3:100
    if train[i,1] == 1.0
        plot!(train[i,2:end,:], legend = false)
    end

end
savefig("visualization/data/abnormal.png")


