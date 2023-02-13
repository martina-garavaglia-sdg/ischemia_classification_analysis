using Flux: onecold, OneHotMatrix

# Accuracy
function accuracy(y_true::OneHotMatrix, y_pred::Any)
    if size(y_true) == size(y_pred)
        a = onecold(y_true, 0:(size(y_true)[1]-1))
        b = onecold(y_pred, 0:(size(y_pred)[1]-1))
        l = size(a)[1]
        return sum(a .== b) / l
    else
        error("Error: true labels and predicted labels must have the same size")
    end
end


# Best model parameters evaluating losses
function is_best(old_loss, new_loss)
    return old_loss > new_loss
end